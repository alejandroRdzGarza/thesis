"""dagger_trend.py — tabulate the DAgger distillation progression across rounds.

Reads every round's summaries under a results dir and prints two trends:
  • STUDENT rollout (round{N}_demos): the state distribution DAgger collects on — how safe the
    current student is when it drives itself (noisy, exploratory).
  • no-CBF EVAL (round{N}_eval_nocbf): the headline — deterministic (ODE) success + collision with
    NO shield. This is the number that must move: collision DOWN, success UP across rounds.

  python -m experiments.dagger_trend            # scans results_distill
  python -m experiments.dagger_trend results_distill_v2
"""
import glob
import json
import re
import sys
from pathlib import Path


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _rounds(base, suffix):
    out = {}
    for p in glob.glob(str(Path(base) / f"round*_{suffix}" / "round_summary.json")):
        m = re.search(r"round(\d+)_", Path(p).parent.name)
        if m:
            s = _load(p)
            if s:
                out[int(m.group(1))] = s
    return out


def _row(n, s, succ_key, coll_key):
    if s is None:
        return f"  round {n}: (missing)"
    su = s.get(succ_key, s.get("success_rate"))
    co = s.get(coll_key, s.get("robot_caused_collision_rate"))
    nn = s.get("n_rollouts", "?")
    return (f"  round {n}:  success {su*100:5.1f}%   collision {co*100:5.1f}%   (n={nn})"
            if su is not None and co is not None else f"  round {n}: (no rates)")


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "results_distill"
    evals = _rounds(base, "eval_nocbf")
    demos = _rounds(base, "demos")

    print(f"\n=== DAgger progression under {base}/ ===")
    print("\nno-CBF EVAL (headline — deterministic, no shield):")
    if evals:
        for n in sorted(evals):
            print(_row(n, evals[n], "unshielded_success_rate", "unshielded_collision_rate"))
    else:
        print("  (no round*_eval_nocbf yet)")

    print("\nSTUDENT rollout distribution (round{N}_demos, noisy):")
    if demos:
        for n in sorted(demos):
            print(_row(n, demos[n], "unshielded_success_rate", "unshielded_collision_rate"))
    else:
        print("  (no round*_demos yet)")

    if len(evals) >= 2:
        ks = sorted(evals)
        c0 = evals[ks[0]].get("unshielded_collision_rate")
        c1 = evals[ks[-1]].get("unshielded_collision_rate")
        s0 = evals[ks[0]].get("unshielded_success_rate")
        s1 = evals[ks[-1]].get("unshielded_success_rate")
        if None not in (c0, c1, s0, s1):
            print(f"\ntrend r{ks[0]}→r{ks[-1]}:  collision {c0*100:.0f}%→{c1*100:.0f}%  "
                  f"({'DOWN ✓' if c1 < c0 else 'UP ✗' if c1 > c0 else 'flat'}),  "
                  f"success {s0*100:.0f}%→{s1*100:.0f}%")
    print()


if __name__ == "__main__":
    main()
