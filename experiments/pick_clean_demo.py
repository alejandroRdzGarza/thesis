"""pick_clean_demo.py — print the first clean (success + collision-free) demo from a round's
manifest as: trace_path<TAB>suite<TAB>task<TAB>episode. Used by run_overfit_sanity.sh to choose
a single demo to hard-overfit (the BC-machinery sanity check).
"""
import csv
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: pick_clean_demo.py <round_dir> [nth]")
    mpath = Path(sys.argv[1]) / "manifest.csv"
    nth = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if not mpath.exists():
        sys.exit(f"no manifest.csv in {sys.argv[1]}")
    clean = []
    with open(mpath) as f:
        for r in csv.DictReader(f):
            if float(r.get("r_success", 0) or 0) > 0 and int(r.get("robot_caused_collision", 0) or 0) == 0:
                clean.append(r)
    if not clean:
        sys.exit("no clean (success+safe) demo in manifest — the expert produced none to overfit")
    r = clean[min(nth, len(clean) - 1)]
    print(f"{r['trace_path']}\t{r['suite']}\t{r['task']}\t{r['episode']}")


if __name__ == "__main__":
    main()
