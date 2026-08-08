"""uncertainty_monitor.py — does the policy's own generative uncertainty predict collisions?

A safety filter needs obstacle geometry, a barrier, and a QP at every control step. A MONITOR that
only has to say "something is about to go wrong" could be far cheaper — and if the signal already
exists inside the policy, it needs no extra sensing at all.

pi0.5 produces each action by integrating a flow ODE over ~10 denoising steps, and every trace
stores that whole chain (`chain`, `logp_old`, `sigmas`). Nothing has ever used it. The hypothesis:
in a state where safe and unsafe behaviours are both plausible, the denoising trajectory should look
different — longer, less settled — than in an unambiguous state. If that shows up before collisions,
the policy has been carrying a usable risk signal all along.

SIGNALS (per VLA query, all derived from the chain geometry)
  path_len       sum ||x_{t+1} - x_t|| over the denoising chain — total distance travelled
  terminal_step  ||x_T - x_{T-1}|| — how unsettled the last step still is
  chain_spread   std of the chain around its endpoint — how much the sample moved around
  logp_sum       sum of per-step log-probs. DEGENERATE at --noise-level 0 (deterministic ODE, zero
                 variance), which is what the evals used; reported but expect it to carry nothing.

Scored as AUC for predicting whether the episode collided. AUC 0.5 = no signal; >0.7 = usable.

    python -m experiments.uncertainty_monitor --evals results_shielded/eval_base_nocbf \
        results_shielded/eval_r1_nocbf

Reads only the chain keys via lazy np.load, so images are never decompressed.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (no sklearn). labels: 1 = collided."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def episode_signals(path: Path) -> dict | None:
    """Per-episode aggregates of the per-query chain geometry."""
    try:
        with np.load(path, allow_pickle=True) as z:
            if int(z["n_queries"]) == 0:
                return None
            chain = np.asarray(z["chain"], dtype=np.float32)      # (Q, S+1, *ashape)
            logp = np.asarray(z["logp_old"], dtype=np.float32)    # (Q, S)
    except Exception:
        return None

    q = len(chain)
    flat = chain.reshape(q, chain.shape[1], -1)                   # (Q, S+1, D)
    steps = np.linalg.norm(np.diff(flat, axis=1), axis=2)         # (Q, S)
    path_len = steps.sum(axis=1)
    terminal = steps[:, -1]
    spread = np.linalg.norm(flat - flat[:, -1:, :], axis=2).mean(axis=1)
    lp = logp.sum(axis=1)

    # Max over the episode as well as mean: a monitor fires on the WORST moment, not the average.
    return {
        "path_len_mean": float(path_len.mean()), "path_len_max": float(path_len.max()),
        "terminal_mean": float(terminal.mean()), "terminal_max": float(terminal.max()),
        "spread_mean": float(spread.mean()), "spread_max": float(spread.max()),
        "logp_mean": float(lp.mean()), "logp_min": float(lp.min()),
        "_logp_degenerate": bool(np.allclose(logp, 0.0) or not np.isfinite(logp).all()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals", nargs="+", required=True, help="eval dirs holding */manifest.csv")
    ap.add_argument("--max-episodes", type=int, default=400)
    args = ap.parse_args()

    rows = []
    for ed in args.evals:
        for m in sorted(glob.glob(str(Path(ed) / "**" / "manifest.csv"), recursive=True)):
            for r in csv.DictReader(open(m)):
                tp = r.get("trace_path") or ""
                if tp:
                    rows.append((Path(tp), int(r.get("robot_caused_collision", 0) or 0)))
    if not rows:
        raise SystemExit(f"no manifest rows with trace_path under {args.evals}")
    rows = rows[:args.max_episodes]

    print(f"\n  {len(rows)} episodes from {len(args.evals)} eval dir(s)", flush=True)
    feats, labels, degenerate = [], [], 0
    for i, (tp, coll) in enumerate(rows, 1):
        if not tp.exists():
            continue
        s = episode_signals(tp)
        if s is None:
            continue
        degenerate += int(s.pop("_logp_degenerate"))
        feats.append(s); labels.append(coll)
        if i % 50 == 0 or i == len(rows):
            print(f"    [{i}/{len(rows)}] {len(feats)} usable", flush=True)

    if not feats:
        raise SystemExit("no readable traces — check that trace_path resolves on this machine")

    labels = np.asarray(labels)
    keys = list(feats[0].keys())
    print(f"\n  usable episodes : {len(labels)}   collided: {int(labels.sum())} "
          f"({labels.mean():.0%})")
    if degenerate:
        print(f"  NOTE: logp is degenerate in {degenerate}/{len(feats)} episodes — expected at "
              f"--noise-level 0 (deterministic ODE has zero sampling variance). Ignore logp rows.")

    print(f"\n  {'signal':<18}{'AUC':>8}   (0.50 = no signal, >0.70 = usable monitor)")
    results = []
    for k in keys:
        v = np.asarray([f[k] for f in feats], dtype=float)
        if not np.isfinite(v).all() or v.std() == 0:
            print(f"  {k:<18}{'—':>8}   (constant or non-finite)")
            continue
        a = auc(v, labels)
        # A signal that is INVERSELY predictive is just as useful; report the stronger direction.
        results.append((max(a, 1 - a), k, a))
        print(f"  {k:<18}{a:>8.3f}" + ("   <-- inverted (lower = riskier)" if a < 0.5 else ""))

    if results:
        best, name, raw = max(results)
        print(f"\n  BEST: {name}  AUC {raw:.3f}  (|deviation from chance| {abs(raw-0.5):.3f})")
        if best > 0.70:
            print("    -> usable as a barrier-free runtime monitor. Worth pursuing: it needs no")
            print("       obstacle geometry and no extra sensing, only the policy's own sampler.")
        elif best > 0.60:
            print("    -> weak but real. Might work combined with other signals, or per-query")
            print("       rather than per-episode (a monitor fires on a moment, not an average).")
        else:
            print("    -> no usable signal at episode level. Report as a negative result; the")
            print("       per-query pre-collision version is the remaining thing worth testing.")


if __name__ == "__main__":
    main()
