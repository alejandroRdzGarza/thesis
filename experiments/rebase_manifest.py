"""rebase_manifest.py — rewrite absolute trace_path entries for a different machine.

Demo manifests store `trace_path` as an ABSOLUTE path, resolved on the machine that collected them.
That is right for local use and wrong the moment the round is copied anywhere else: rsync a
collection to a GPU box and every row still points at /Users/<you>/..., so the trainer finds zero
traces and fails — or worse, silently trains on nothing if the loader is lenient.

Rewrites the prefix up to and including the round directory, leaving the per-shard structure alone.

    # on the pod, after rsyncing results_distill/planner_A
    python -m experiments.rebase_manifest --round results_distill/planner_A

By default it rebases onto the round directory's own absolute location, which is what you want
after a copy. --prefix overrides that if the manifest is being prepared for a third machine.
Writes a .bak alongside, and is idempotent — running it twice changes nothing.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def rebase(round_dir: Path, new_prefix: Path, manifest_name: str = "manifest.csv") -> tuple[int, int]:
    mp = round_dir / manifest_name
    rows = list(csv.DictReader(open(mp)))
    if not rows:
        return 0, 0
    changed = 0
    for r in rows:
        old = r.get("trace_path", "")
        if not old:
            continue
        # Keep everything after the round-directory name; that is the shard/file structure, which
        # the copy preserved. Only the machine-specific prefix ahead of it is wrong.
        parts = Path(old).parts
        if round_dir.name in parts:
            tail = Path(*parts[parts.index(round_dir.name) + 1:])
        else:
            tail = Path(old).name          # flat layout: just the filename
        new = str((new_prefix / tail).resolve())
        if new != old:
            r["trace_path"] = new
            changed += 1
    shutil.copy2(mp, mp.with_suffix(".csv.bak"))
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    missing = sum(1 for r in rows if not Path(r["trace_path"]).exists())
    return changed, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", required=True, type=Path)
    ap.add_argument("--prefix", type=Path, default=None,
                    help="new absolute path of the round dir (default: where it actually is now)")
    ap.add_argument("--manifest", default="manifest.csv",
                    help="manifest filename inside the round dir (e.g. manifest_clean.csv)")
    args = ap.parse_args()

    rd: Path = args.round.resolve()
    prefix = (args.prefix or rd).resolve()
    changed, missing = rebase(rd, prefix, args.manifest)
    print(f"  {args.manifest}: rewrote {changed} trace_path entries -> {prefix}")
    if missing:
        # Loudly, because the failure mode this guards against is training on an empty set.
        print(f"  WARNING: {missing} trace file(s) still not found on disk. The copy is incomplete "
              f"or --prefix is wrong; training would silently see fewer demos than the manifest "
              f"claims.")
    else:
        print("  all trace files resolve on this machine.")


if __name__ == "__main__":
    main()
