"""collect_shielded_demos.py — round-0 BC demos from π0.5 + CBF, across the whole SafeLIBERO grid.

The teacher here is the SHIELDED POLICY, not the scripted controller. On the AEGIS-matched
baseline π0.5+CBF runs at TSR 79.2% / CAR 87.5% — roughly a 69% clean-episode yield, against ~42%
for the classical expert (and 0% on several scenes). It is also a far easier BC target: the
classical expert emits saturated P-control/QP deltas, a different action distribution from π0.5's
flow-matched chunks, so imitating it drags the policy off its pretrained prior. Shielded π0.5
actions are π0.5's own actions minus a small safety projection — a much smaller delta to learn.

Two selection criteria, both of which matter:

  CLEAN     succeeded AND did not move the obstacle. Uses the RAW collision flag, not the
            culprit-attribution one: the contact-graph attribution is a documented LOWER BOUND
            (it misses delayed / indirect pushes), so filtering on it would admit demos that
            actually collided. For a safety demo set, err strict.

  SHIELD-ACTIVE   the CBF actually intervened at least once (`--min-cbf-acts`). A clean episode
            where the shield never fired is just base π0.5 behaviour and teaches nothing about
            safety — imitating it is how distillation degenerates into copying the base policy.
            This is SafeDAgger's actual insight, and it is what stops round 0 being a no-op.

Only demos passing the filters are written, so `flow_bc_train --round <out> --success-only` can
consume the output directly; its manifest filter then acts as a redundant safety net.

  # round 0: base π0.5 + shield over the grid
  $PY -m experiments.collect_shielded_demos --checkpoint $CKPT --out results_distill/shielded_r0

  # round N: same command against the FINE-TUNED checkpoint = SafeDAgger
  # (data now comes from the student's own state distribution)
  $PY -m experiments.collect_shielded_demos --checkpoint <round_n_ckpt> \
      --out results_distill/shielded_r1

GPU-bound and single-process: one A40 holds one π0.5, so unlike the classical collector this
cannot be sharded across workers. Budget ~1-2 min per rollout at --num-steps 10; --num-steps 4
roughly halves it. Scale --episodes to the time you have.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

SUITES = ["safelibero_spatial", "safelibero_object", "safelibero_goal"]
LEVELS = ["I", "II"]
TASKS = [0, 1, 2, 3]

FIELDS = ["trace_path", "r_success", "robot_caused_collision", "collision_raw",
          "cbf_activations", "cbf_activation_rate", "cbf_mean_correction",
          "suite", "level", "task", "episode", "queries"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pi05_libero_cbf")
    ap.add_argument("--checkpoint", required=True,
                    help="base π0.5 for round 0; the previous round's LoRA checkpoint for DAgger rounds")
    ap.add_argument("--suites", nargs="+", default=SUITES)
    ap.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    ap.add_argument("--tasks", type=int, nargs="+", default=TASKS)
    ap.add_argument("--episodes", type=int, nargs="+", default=list(range(20)),
                    help="init indices to roll out. DEFAULT 0-19; keep 35-49 held out for eval.")
    ap.add_argument("--out", default="results_distill/shielded_r0")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=10, help="flow denoising steps (4 ≈ 2× faster)")
    ap.add_argument("--noise-level", type=float, default=0.0,
                    help="0 = deterministic ODE. Round 0 wants the policy's best behaviour, not "
                         "exploration; raise it only if you need rollout diversity.")
    ap.add_argument("--sde-type", default="cps", choices=["cps", "sde"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-cbf-acts", type=int, default=1,
                    help="require the shield to have fired at least this many times. 0 keeps every "
                         "clean episode (including ones where the shield never engaged, which carry "
                         "no safety signal).")
    ap.add_argument("--keep-all", action="store_true",
                    help="write every trace regardless of the filters (for analysis, not training)")
    args = ap.parse_args()

    from openpi.policies.policy_logprob import PolicyWithLogprob
    from openpi.training import config as _config
    from experiments.libero_runner import make_libero_env, run_libero_trial
    from experiments.load_policy import create_policy_partial
    from experiments.rl_rollout_local import build_policy_fn
    from experiments.policy_trace import save_episode_trace

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "collect.log", "a")

    def say(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    say(f"=== shielded π0.5 demo collection  {time.strftime('%Y-%m-%d %H:%M')} ===")
    say(f"    checkpoint {args.checkpoint}")
    say(f"    loading policy (once — this is GPU-bound and single-process) …")
    train_cfg = _config.get_config(args.config)
    policy = create_policy_partial(train_cfg, args.checkpoint)
    pol_lp = PolicyWithLogprob(policy, num_steps=args.num_steps, noise_level=args.noise_level,
                               sde_type=args.sde_type, seed=args.seed)
    policy_fn = build_policy_fn(pol_lp)

    scenes = [(s, l, t) for s in args.suites for l in args.levels for t in args.tasks]
    total = len(scenes) * len(args.episodes)
    say(f"    {len(scenes)} scenes × {len(args.episodes)} inits = {total} rollouts")
    say(f"    keeping: success AND raw-collision-free"
        + (f" AND cbf_activations >= {args.min_cbf_acts}" if args.min_cbf_acts else "")
        + ("  [--keep-all: filters reported but not applied]" if args.keep_all else ""))
    say("")

    rows: list[dict] = []
    per = defaultdict(lambda: [0, 0, 0, 0])   # scene → [n, success, clean, kept]
    t0 = time.time()
    done = 0

    for suite, level, task in scenes:
        tag = f"{suite}_L{level}_t{task}"
        try:
            env, lang, inits = make_libero_env(task_suite=suite, task_idx=task,
                                               safety_level=level, horizon=args.horizon)
        except Exception as e:
            say(f"  [{tag}] ENV ERROR — skipped ({e})")
            continue

        for ep in args.episodes:
            done += 1
            try:
                m = run_libero_trial(
                    env=env, episode_idx=ep, instruction=lang, initial_states=inits,
                    obstacles=[], goal_pos=None, auto_goal=True, use_geo_success=False,
                    use_cbf=True, vla="pi05", auto_detect_obstacle=True, aegis_faithful=True,
                    replan_steps=args.replan, horizon=args.horizon,
                    policy_fn=policy_fn, record_policy_trace=True, scene_name=tag)
            except Exception as e:
                say(f"  [{done}/{total}] {tag} ep{ep}: ERROR {type(e).__name__}: {e}")
                continue

            s = m.summary()
            ok = bool(s.get("goal_reached"))
            raw_coll = bool(s.get("collision_detected"))       # strict: RAW, not attributed
            acts = int(s.get("cbf_activations", 0) or 0)
            clean = ok and not raw_coll
            keep = clean and acts >= args.min_cbf_acts

            st = per[(suite, level, task)]
            st[0] += 1; st[1] += int(ok); st[2] += int(clean); st[3] += int(keep)

            if keep or args.keep_all:
                tp = str((out / f"{tag}_ep{ep}_trace.npz").resolve())
                save_episode_trace(m.policy_trace, tp)
                rows.append({
                    "trace_path": tp,
                    # flow_bc_train --success-only filters on these two. We already applied a
                    # STRICTER filter above, so set the collision column from the raw flag.
                    "r_success": 1.5 if ok else 0.0,
                    "robot_caused_collision": int(raw_coll),
                    "collision_raw": int(raw_coll),
                    "cbf_activations": acts,
                    "cbf_activation_rate": s.get("cbf_activation_rate", 0.0),
                    "cbf_mean_correction": s.get("cbf_mean_correction_norm", 0.0),
                    "suite": suite, "level": level, "task": task, "episode": ep,
                    "queries": len(m.policy_trace),
                })

            eta = (time.time() - t0) / done * (total - done) / 60.0
            say(f"  [{done}/{total}] {tag} ep{ep}: succ={ok} coll={raw_coll} cbf_acts={acts} "
                f"→ {'KEPT' if keep else 'dropped'}   (eta {eta:.0f} min)")

        try:
            env.close()
        except Exception:
            pass

        # Checkpoint the manifest after every scene so an interrupted run stays usable.
        if rows:
            with open(out / "manifest.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader(); w.writerows(rows)

    # ── report ──────────────────────────────────────────────────────────────
    say("")
    say("=" * 78)
    say(" PER-SCENE YIELD   (kept = clean AND the shield actually intervened)")
    say("=" * 78)
    say(f"{'scene':<40}{'n':>5}{'success':>9}{'clean':>7}{'KEPT':>7}")
    tot = [0, 0, 0, 0]
    for (suite, level, task), (n, sc, cl, kp) in sorted(per.items()):
        say(f"{suite + ' L' + level + ' t' + str(task):<40}{n:>5}{sc:>9}{cl:>7}{kp:>7}")
        tot = [a + b for a, b in zip(tot, (n, sc, cl, kp))]
    say("-" * 78)
    say(f"{'TOTAL':<40}{tot[0]:>5}{tot[1]:>9}{tot[2]:>7}{tot[3]:>7}")
    say("")
    if tot[2] and tot[3] < tot[2]:
        say(f"  note: {tot[2] - tot[3]} clean episode(s) dropped because the shield never fired — "
            "they carry no safety signal. Re-run with --min-cbf-acts 0 to keep them.")
    say(f"  manifest → {out}/manifest.csv   ({len(rows)} demos)")
    say(f"  train:  $PY -m experiments.flow_bc_train --round {out} --success-only --epochs 20")

    (out / "round_summary.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "rollouts": tot[0], "success": tot[1],
        "clean": tot[2], "kept": tot[3], "min_cbf_acts": args.min_cbf_acts,
        "episodes": args.episodes, "minutes": round((time.time() - t0) / 60, 1),
    }, indent=2) + "\n")
    logf.close()


if __name__ == "__main__":
    main()
