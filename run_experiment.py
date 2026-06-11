"""
CLI entry point for running VLA safety experiments.

Usage examples
--------------
# Run a single scene, both modes:
  mjpython run_experiment.py --scene direct_block

# Run one specific mode:
  mjpython run_experiment.py --scene narrow_corridor --mode cbf

# Run every preset scene, both modes:
  mjpython run_experiment.py --all

# List available scenes:
  mjpython run_experiment.py --list

Results are written to results/<scene_name>_<mode>_steps.csv
                         and results/<scene_name>_<mode>_summary.csv
"""

import argparse
from experiments.scene_config import ALL_SCENES
from experiments.runner import run_trial


def main():
    parser = argparse.ArgumentParser(
        description="Run VLA safety benchmark experiments."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene", choices=list(ALL_SCENES.keys()),
                       help="Name of the scene to run.")
    group.add_argument("--all",   action="store_true",
                       help="Run every scene in both modes.")
    group.add_argument("--list",  action="store_true",
                       help="List available scenes and exit.")

    parser.add_argument("--mode", choices=["plain", "cbf", "both"], default="both",
                        help="Which mode to run (default: both).")
    parser.add_argument("--results-dir", default="results",
                        help="Directory for output CSVs (default: results/).")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable scenes:")
        for name, cfg in ALL_SCENES.items():
            obs_count = len(cfg.obstacles)
            print(f"  {name:<22} — {cfg.description}  ({obs_count} obstacle{'s' if obs_count != 1 else ''})")
        print()
        return

    scenes_to_run = list(ALL_SCENES.values()) if args.all else [ALL_SCENES[args.scene]]
    modes_to_run  = (
        [False, True]  if args.mode == "both"  else
        [False]        if args.mode == "plain" else
        [True]
    )

    for cfg in scenes_to_run:
        for use_cbf in modes_to_run:
            run_trial(cfg, use_cbf=use_cbf, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
