"""collect_bc_all.py — collect classical-expert BC demos across the WHOLE SafeLIBERO grid.

Fans `collect_classical_demos` out over worker processes, one shard per (suite, level, task), then
merges the shard manifests into a single round directory that `flow_bc_train --round <dir>
--success-only` consumes directly.

Parallelism is the point. A rollout costs ~100 s on this Mac and the full grid at 35 inits is 840
of them — 23 hours serially, under 3 with 8 workers. MuJoCo and the MPC QP are both single-threaded
per process, so workers scale nearly linearly until they run out of cores.

    # the default: all 3 suites, both levels, inits 0-34 (35-49 stay HELD OUT for evaluation)
    PYTHONPATH=. python -m experiments.collect_bc_all --out results_distill/bc_all

    # re-run: finished shards are skipped, so an interrupted collection resumes
    PYTHONPATH=. python -m experiments.collect_bc_all --out results_distill/bc_all

Each shard writes `<out>/<suite>_L<level>_t<task>/` with its own manifest; the merged
`<out>/manifest.csv` is what the trainer filters on. Keep the eval inits out of `--episodes` —
benchmarking the fine-tuned VLA on inits it was trained on measures memorisation, not learning.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SUITES = ["safelibero_spatial", "safelibero_object", "safelibero_goal"]
LEVELS = ["I", "II"]
TASKS = [0, 1, 2, 3]

MANIFEST_FIELDS = ["trace_path", "r_success", "robot_caused_collision", "suite", "task", "episode"]


def shard_tag(suite: str, level: str, task: int) -> str:
    return f"{suite}_L{level}_t{task}"


def build_shards(args) -> list[tuple[str, str, int]]:
    return [(s, l, t) for s in args.suites for l in args.levels for t in args.tasks]


def shard_done(out: Path, suite: str, level: str, task: int) -> bool:
    """A shard counts as done ONLY if its worker exited cleanly.

    Not "a manifest exists": a worker killed part-way (Ctrl-C, SIGTERM, OOM) can leave a partial
    manifest behind, and treating that as complete would make a resume silently skip real work and
    report an empty collection as finished."""
    return (out / shard_tag(suite, level, task) / ".complete").exists()


def launch(out: Path, suite: str, level: str, task: int, args) -> tuple[subprocess.Popen, Path]:
    sd = out / shard_tag(suite, level, task)
    sd.mkdir(parents=True, exist_ok=True)
    log = sd / "collect.log"
    cmd = [sys.executable, "-m", "experiments.collect_classical_demos",
           "--suite", suite, "--level", level, "--tasks", str(task),
           "--episodes", *[str(e) for e in args.episodes],
           "--out", str(sd), "--horizon", str(args.horizon), "--replan", str(args.replan)]
    cmd += ["--teacher", args.teacher]
    if args.teacher == "planner":
        cmd.append("--no-cbf")      # self-safe teacher; the shield only fights its plan
    if args.randomize_seed is not None:
        cmd += ["--randomize-seed", str(args.randomize_seed)]
    if args.clean_only:
        cmd.append("--clean-only")
    env = dict(os.environ, PYTHONPATH=os.environ.get("PYTHONPATH", "."))
    return subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env), log


def merge(out: Path) -> list[dict]:
    rows: list[dict] = []
    for m in sorted(out.glob("*/manifest.csv")):
        with open(m) as f:
            rows.extend(csv.DictReader(f))
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})
    return rows


def report(rows: list[dict], out: Path, say) -> None:
    """Per-scene clean-demo counts — the number that decides whether BC has anything to learn."""
    from collections import defaultdict
    per = defaultdict(lambda: [0, 0, 0])          # scene → [n, success, clean]
    for r in rows:
        ok = float(r.get("r_success", 0) or 0) > 0
        safe = int(r.get("robot_caused_collision", 0) or 0) == 0
        key = (r["suite"], r["task"])
        per[key][0] += 1
        per[key][1] += int(ok)
        per[key][2] += int(ok and safe)
    say("")
    say("=" * 72)
    say(f" CLEAN DEMOS PER SCENE  (success + collision-free — what BC actually trains on)")
    say("=" * 72)
    say(f"{'scene':<40}{'demos':>8}{'success':>10}{'CLEAN':>9}")
    tot = [0, 0, 0]
    for (suite, task), (n, s, c) in sorted(per.items()):
        say(f"{suite + ' t' + str(task):<40}{n:>8}{s:>10}{c:>9}")
        tot = [tot[0] + n, tot[1] + s, tot[2] + c]
    say("-" * 72)
    say(f"{'TOTAL':<40}{tot[0]:>8}{tot[1]:>10}{tot[2]:>9}")
    say("")
    say(f"  merged manifest → {out}/manifest.csv")
    say(f"  train with:  flow_bc_train --round {out} --success-only")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="+", default=SUITES)
    ap.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    ap.add_argument("--tasks", type=int, nargs="+", default=TASKS)
    ap.add_argument("--episodes", type=int, nargs="+", default=list(range(35)),
                    help="init indices to collect. DEFAULT 0-34; 35-49 are left held out so the "
                         "fine-tuned VLA can be benchmarked on inits it never saw.")
    ap.add_argument("--out", type=Path, default=Path("results_distill/bc_all"))
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--horizon", type=int, default=None,
                    help="control steps per episode. Default depends on the teacher: the\n                          scripted controller finishes within 300, the planner needs ~900\n                          because it tracks a dense waypoint trace.")
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--clean-only", action="store_true", default=True,
                    help="drop traces for failed/colliding episodes (default: on)")
    ap.add_argument("--keep-all", dest="clean_only", action="store_false",
                    help="keep every trace on disk, not just the clean ones")
    ap.add_argument("--teacher", default="planner", choices=["classical", "planner"])
    ap.add_argument("--randomize-seed", type=int, default=None,
                    help="TRAINING-DATA obstacle randomisation. Omit for canonical layouts "
                         "(Student A); set it for the augmented set (Student B). Never for eval.")
    ap.add_argument("--force", action="store_true", help="re-run shards that already have a manifest")
    ap.add_argument("--merge-only", action="store_true", help="just merge existing shards and report")
    args = ap.parse_args()

    if args.horizon is None:
        args.horizon = 900 if args.teacher == "planner" else 300
    if args.teacher == "planner":
        args.replan = 1          # the planner is queried every control step, not every 5

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "collect_bc_all.log", "a")

    def say(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    if args.merge_only:
        report(merge(out), out, say)
        return

    shards = build_shards(args)
    todo = [s for s in shards if args.force or not shard_done(out, *s)]
    skipped = len(shards) - len(todo)

    say(f"=== BC demo collection  {time.strftime('%Y-%m-%d %H:%M')} ===")
    say(f"    {len(shards)} scenes × {len(args.episodes)} inits = {len(shards)*len(args.episodes)} rollouts")
    say(f"    inits {min(args.episodes)}-{max(args.episodes)} collected; "
        f"anything outside stays held out for evaluation")
    say(f"    {args.workers} workers" + (f", {skipped} shard(s) already done — skipping" if skipped else ""))
    say("")

    t0 = time.time()
    running: list[tuple[subprocess.Popen, tuple, Path]] = []
    queue = list(todo)
    interrupted: list[tuple] = []
    done = 0

    def stop_all(*_a):
        """Ctrl-C / SIGTERM must take the WHOLE run down, not just the current workers.

        Without this the parent would see a killed worker, mark that shard finished and launch the
        next one — so killing workers by hand looks like they respawn forever."""
        say("\n  interrupt — stopping workers and exiting (re-run to resume)")
        for pr, _sc, _lg in running:
            pr.terminate()
        for pr, _sc, _lg in running:
            try:
                pr.wait(timeout=10)
            except Exception:
                pr.kill()
        logf.close()
        raise SystemExit(130)

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    while queue or running:
        while queue and len(running) < args.workers:
            sc = queue.pop(0)
            p, log = launch(out, *sc, args)
            running.append((p, sc, log))
            say(f"  ▶ start  {shard_tag(*sc)}  ({len(running)} running, {len(queue)} queued)")
        time.sleep(5)
        for entry in list(running):
            p, sc, log = entry
            if p.poll() is None:
                continue
            running.remove(entry)
            done += 1
            sd = out / shard_tag(*sc)
            n = clean = 0
            mp = sd / "manifest.csv"
            if mp.exists():
                with open(mp) as f:
                    for r in csv.DictReader(f):
                        n += 1
                        clean += int(float(r.get("r_success", 0) or 0) > 0
                                     and int(r.get("robot_caused_collision", 0) or 0) == 0)
            if p.returncode == 0:
                (sd / ".complete").touch()            # only a clean exit marks the shard resumable-skippable
                say(f"  ✔ done   {shard_tag(*sc):<34} {clean} clean / {n} traces   [{done}/{len(todo)}]")
            else:
                why = "killed (SIGTERM)" if p.returncode == -15 else f"exit {p.returncode}"
                say(f"  ✖ FAILED {shard_tag(*sc):<34} {why} — see {log}   [{done}/{len(todo)}]"
                    f"  (re-run to retry this shard)")
                interrupted.append(sc)

    say(f"\nall shards finished in {(time.time()-t0)/60:.1f} min")
    if interrupted:
        say(f"  {len(interrupted)} shard(s) did NOT complete: "
            + ", ".join(shard_tag(*s) for s in interrupted))
        say("  re-run the same command to retry them (completed shards are skipped)")
    report(merge(out), out, say)
    logf.close()


if __name__ == "__main__":
    main()
