#!/usr/bin/env python
"""
summarize_results.py — Compile a SafeLIBERO results dir into a markdown report.

Reads *_agg.csv (per-scene metrics) and *_episodes.csv (per-episode, for the
robot-caused collision decomposition) recursively under --dir, and writes a
comprehensive report: per-scene table, per-suite/level aggregates (plain vs cbf),
the collision decomposition (raw vs robot-caused vs physics artifact), and the
culprit breakdown.

Usage:
    python experiments/summarize_results.py --dir results_baseline
    python experiments/summarize_results.py --dir results_baseline --out figures/results_summary.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def parse_scene(scene: str):
    """'safelibero_spatial_t00_LI' -> (suite, task, level)."""
    m = re.match(r"(.+)_t(\d+)_L([I]+)$", scene)
    if not m:
        return scene, "?", "?"
    return m.group(1), int(m.group(2)), m.group(3)


def load_aggs(root: str):
    rows = []
    for f in glob.glob(f"{root}/**/*_agg.csv", recursive=True):
        for r in csv.DictReader(open(f)):
            suite, task, level = parse_scene(r.get("scene", ""))
            rows.append(dict(suite=suite, task=task, level=level, mode=r.get("mode", ""),
                             car=_f(r.get("car_pct")), tsr=_f(r.get("tsr_pct")),
                             coll=_f(r.get("collision_rate_pct")), ets=_f(r.get("ets")),
                             cbf_act=_f(r.get("mean_cbf_activation_rate")),
                             n=int(_f(r.get("n_episodes")))))
    return rows


def load_episodes(root: str):
    """Yield (suite, level, mode, episode_dict) across all *_episodes.csv."""
    for f in glob.glob(f"{root}/**/*_episodes.csv", recursive=True):
        name = Path(f).stem                      # pi05_<suite>_tNN_LX_<mode>_episodes
        m = re.search(r"(safelibero_\w+?)_t\d+_L([I]+)_(plain|cbf|apf)", name)
        if not m:
            continue
        suite, level, mode = m.group(1), m.group(2), m.group(3)
        for r in csv.DictReader(open(f)):
            yield suite, level, mode, r


def collision_decomp(root: str):
    """Per (suite, level, mode): totals, robot-caused vs artifact, culprit counts."""
    agg = defaultdict(lambda: dict(n=0, coll=0, rc=0, art=0, culprit=Counter()))
    for suite, level, mode, r in load_episodes(root):
        d = agg[(suite, level, mode)]
        d["n"] += 1
        if r.get("collision") == "1":
            d["coll"] += 1
            if r.get("collision_robot_caused") == "1":
                d["rc"] += 1
                d["culprit"][r.get("collision_culprit", "?")] += 1
            else:
                d["art"] += 1
    return agg


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    aggs = load_aggs(args.dir)
    decomp = collision_decomp(args.dir)
    if not aggs:
        print(f"No *_agg.csv found under {args.dir}")
        return

    L = ["# SafeLIBERO baseline results", "", f"Source: `{args.dir}`", ""]

    # ── Per-scene table ──────────────────────────────────────────────────────
    L += ["## Per-task results", ""]
    order = sorted({(a["suite"], a["level"], a["task"]) for a in aggs})
    by_scene = {(a["suite"], a["level"], a["task"], a["mode"]): a for a in aggs}
    rows = []
    for suite, level, task in order:
        for mode in ("plain", "cbf"):
            a = by_scene.get((suite, level, task, mode))
            if not a:
                continue
            rows.append([f"{suite.replace('safelibero_','')} L{level} t{task:02d}", mode,
                         f"{a['car']:.0f}", f"{a['tsr']:.0f}", f"{a['coll']:.0f}",
                         f"{a['ets']:.0f}", f"{a['cbf_act']:.3f}"])
    L += [md_table(["scene", "mode", "CAR%↑", "TSR%↑", "Coll%↓", "ETS↓", "cbf_act"], rows), ""]

    # ── Per-suite/level aggregate (mean over tasks) ──────────────────────────
    L += ["## Aggregates (mean over tasks)", ""]
    grp = defaultdict(lambda: defaultdict(list))
    for a in aggs:
        grp[(a["suite"], a["level"])][a["mode"]].append(a)
    rows = []
    for (suite, level) in sorted(grp):
        for mode in ("plain", "cbf"):
            g = grp[(suite, level)].get(mode, [])
            if not g:
                continue
            mean = lambda k: sum(x[k] for x in g) / len(g)
            rows.append([f"{suite.replace('safelibero_','')} L{level}", mode, len(g),
                         f"{mean('car'):.1f}", f"{mean('tsr'):.1f}", f"{mean('coll'):.1f}",
                         f"{mean('ets'):.0f}", f"{mean('cbf_act'):.3f}"])
    L += [md_table(["suite/level", "mode", "#tasks", "CAR%↑", "TSR%↑", "Coll%↓", "ETS↓", "cbf_act"], rows), ""]

    # ── Collision decomposition (the artifact-corrected numbers) ─────────────
    L += ["## Collision decomposition (RAW is primary; robot_caused = attribution lower bound)", "",
          "Raw collision = SafeLIBERO >2mm displacement (comparable to AEGIS/VLSA) and is the "
          "primary safety metric: a still-arm drift test shows the active obstacle is "
          "physically stable, so raw displacement is robot-caused by construction. "
          "`robot_caused` = obstacle reachable from a robot body through the contact graph "
          "within a short window at threshold-crossing; it UNDER-counts delayed/indirect "
          "pushes, so `unattributed` reflects attribution misses, NOT physics artifacts.", ""]
    rows = []
    tot = defaultdict(lambda: dict(n=0, coll=0, rc=0, art=0, culprit=Counter()))
    for (suite, level, mode) in sorted(decomp):
        d = decomp[(suite, level, mode)]
        rows.append([f"{suite.replace('safelibero_','')} L{level}", mode, d["n"],
                     f"{100*d['coll']/max(d['n'],1):.0f}",
                     f"{100*d['rc']/max(d['n'],1):.0f}",
                     f"{100*d['art']/max(d['coll'],1):.0f}"])
        t = tot[mode]
        t["n"] += d["n"]; t["coll"] += d["coll"]; t["rc"] += d["rc"]; t["art"] += d["art"]
        t["culprit"].update(d["culprit"])
    L += [md_table(["suite/level", "mode", "n", "raw Coll%", "real robot-caused%", "unattributed% (attrib-miss)"], rows), ""]

    L += ["### Overall by mode", ""]
    rows = []
    for mode in ("plain", "cbf"):
        t = tot[mode]
        if not t["n"]:
            continue
        rows.append([mode, t["n"], f"{100*t['coll']/t['n']:.0f}",
                     f"{100*t['rc']/t['n']:.0f}", f"{100*t['art']/max(t['coll'],1):.0f}"])
    L += [md_table(["mode", "n", "raw Coll%", "real robot-caused%", "unattributed% (attrib-miss)"], rows), ""]

    L += ["### Robot-caused culprit breakdown", ""]
    for mode in ("plain", "cbf"):
        if tot[mode]["culprit"]:
            items = ", ".join(f"`{k}`×{v}" for k, v in tot[mode]["culprit"].most_common())
            L += [f"- **{mode}**: {items}"]
    L += [""]

    report = "\n".join(L)
    out = args.out or str(Path(args.dir) / "results_summary.md")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(report)
    print(report)
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
