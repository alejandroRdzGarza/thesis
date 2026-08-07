"""sweep_eval_stats.py — pool per-task eval manifests into a publication-ready table.

Each policy's eval dir holds task*/manifest.csv (one row per rollout: r_success, r_cbf,
robot_caused_collision). We POOL every rollout across tasks per policy and report, with 95% Wilson
confidence intervals:
  • success rate            (r_success > 0)
  • no-shield collision rate (robot_caused_collision) — the headline safety metric
  • CBF-activation proxy     (mean |r_cbf|, only meaningful for shielded evals)

Wilson intervals (not normal-approx) because rates near 0/1 with modest n need it — 0/40 collisions
is "≤ ~9%", not "exactly 0 ± 0".

  python -m experiments.sweep_eval_stats \
      "base+noCBF=results_sweep/eval_base_nocbf" \
      "distilled+noCBF=results_sweep/eval_distilled_nocbf" \
      "base+CBF=results_sweep/eval_base_cbf" \
      "distilled+CBF=results_sweep/eval_distilled_cbf"
"""
import csv
import glob
import math
import sys
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for k successes in n trials → (lo, hi) as fractions."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def pool(eval_dir: str):
    """Pool all rollout rows across task*/manifest.csv (or a top-level manifest.csv)."""
    paths = glob.glob(str(Path(eval_dir) / "**" / "manifest.csv"), recursive=True)
    rows = []
    for p in sorted(set(paths)):
        with open(p) as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def _rate(rows, pred):
    n = len(rows)
    k = sum(1 for r in rows if pred(r))
    lo, hi = wilson(k, n)
    return k, n, (k / n if n else 0.0), lo, hi


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: sweep_eval_stats.py "label=eval_dir" ["label2=dir2" ...]')
    policies = []
    for arg in sys.argv[1:]:
        if "=" not in arg:
            sys.exit(f"bad arg (need label=dir): {arg}")
        label, d = arg.split("=", 1)
        policies.append((label, d))

    print("\n" + "=" * 104)
    print(f"{'policy':<22}{'n':>5}  {'TSR (95% CI)':>26}  {'collision (95% CI)':>26}  {'cbf|r|':>7}  {'ETS':>7}")
    print("-" * 104)
    results = {}
    for label, d in policies:
        rows = pool(d)
        if not rows:
            print(f"{label:<22}{0:>5}  (no manifests under {d})")
            continue
        sk, sn, sr, slo, shi = _rate(rows, lambda r: float(r.get("r_success", 0) or 0) > 0)
        ck, cn, cr, clo, chi = _rate(rows, lambda r: int(r.get("robot_caused_collision", 0) or 0) == 1)
        cbf = sum(abs(float(r.get("r_cbf", 0) or 0)) for r in rows) / len(rows)
        # ETS = mean control steps to completion, over SUCCEEDED rollouts only (a failure has no
        # completion time, and averaging in the horizon would make a policy look faster the more
        # often it fails). "-" when the manifest predates the ets column.
        _e = [int(r["ets"]) for r in rows
              if str(r.get("ets", "")).lstrip("-").isdigit() and int(r["ets"]) >= 0]
        ets = f"{sum(_e)/len(_e):7.1f}" if _e else f"{'-':>7}"
        results[label] = dict(n=sn, succ=sr, coll=cr, ets=(sum(_e)/len(_e) if _e else None))
        print(f"{label:<22}{sn:>5}  "
              f"{sr*100:5.1f}% [{slo*100:4.1f},{shi*100:5.1f}]   "
              f"{cr*100:5.1f}% [{clo*100:4.1f},{chi*100:5.1f}]   "
              f"{cbf:6.3f}  {ets}")
    print("=" * 104)

    # Which bodies caused the residual collisions. The headline rate says how OFTEN a policy
    # collides; this says WITH WHAT, and it is where a whole-arm teacher should differ from an
    # end-effector shield (which removes gripper and held-object contacts but not arm links).
    cats = ["gripper", "arm_link", "held_object", "scene_object"]
    rows_any = False
    for label, d in policies:
        rows = pool(d)
        cnt = {c: 0 for c in cats}
        other = 0
        for r in rows:
            for c in str(r.get("culprit", "") or "").split("|"):
                c = c.strip()
                if not c:
                    continue
                if c in cnt:
                    cnt[c] += 1
                else:
                    other += 1
        if sum(cnt.values()) + other == 0:
            continue
        if not rows_any:
            print("\n  COLLISION CULPRITS (episodes in which each body touched the obstacle)")
            print(f"  {'policy':<22}" + "".join(f"{c:>14}" for c in cats) + f"{'other':>8}")
            rows_any = True
        print(f"  {label:<22}" + "".join(f"{cnt[c]:>14}" for c in cats) + f"{other:>8}")
    if not rows_any:
        print("\n  (no culprit column in these manifests — re-run the eval to populate it)")

    print("  TSR = env.check_success · collision = raw obstacle displacement >1 mm (AEGIS metric,"
          " CAR = 100 − collision) · ETS = mean steps to success, succeeded rollouts only")

    # Headline deltas vs base, if present.
    base_k = next((l for l in results if l.lower().startswith("base") and "nocbf" in l.lower().replace("-", "").replace("+", "")), None)
    dist_k = next((l for l in results if l.lower().startswith("distilled") and "nocbf" in l.lower().replace("-", "").replace("+", "")), None)
    if base_k and dist_k:
        b, d = results[base_k], results[dist_k]
        print(f"\nno-shield headline (distilled vs base):")
        print(f"  collision: {b['coll']*100:.1f}% → {d['coll']*100:.1f}%  "
              f"({'−' if d['coll']<=b['coll'] else '+'}{abs(d['coll']-b['coll'])*100:.1f} pts)")
        print(f"  success:   {b['succ']*100:.1f}% → {d['succ']*100:.1f}%  "
              f"({'+' if d['succ']>=b['succ'] else '−'}{abs(d['succ']-b['succ'])*100:.1f} pts)")
    print()


if __name__ == "__main__":
    main()
