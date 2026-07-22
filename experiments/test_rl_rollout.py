"""GPU-free checks for rl_rollout.py (run: python -m experiments.test_rl_rollout)."""
import tempfile, csv
from pathlib import Path
from experiments.safe_reward import compute_episode_reward, RewardConfig
from experiments.rl_rollout import (
    Rollout, score_group, write_manifest, collection_summary,
)

cfg = RewardConfig()
ok = True
def check(name, cond):
    global ok; print(f"  [{'PASS' if cond else 'FAIL'}] {name}"); ok = ok and cond

def mk(gid, k, *, success, robot_caused=False, cbf_rate=0.0):
    rb = compute_episode_reward(
        goal_reached=success, collision_detected=robot_caused,
        collision_robot_caused=robot_caused, cbf_activation_rate=cbf_rate, cfg=cfg)
    return Rollout(group_id=gid, rollout_id=k, traj_path=f"g{gid}/r{k}.h5", reward=rb)

# A group of 4: two clean successes, one success+collision, one failure.
group = [
    mk(0, 0, success=True,  robot_caused=False, cbf_rate=0.1),
    mk(0, 1, success=True,  robot_caused=False, cbf_rate=0.3),
    mk(0, 2, success=True,  robot_caused=True,  cbf_rate=0.2),   # collided → lower R
    mk(0, 3, success=False, robot_caused=False, cbf_rate=0.0),   # failed → lowest R
]
score_group(group, temperature=1.0)

# advantages standardised (mean ~ 0)
check("group advantages mean ~ 0", abs(sum(r.advantage for r in group)) < 1e-6)
# best rollout (clean success, low cbf) has highest weight
best = max(group, key=lambda r: r.reward.total)
check("best reward gets top weight", best is max(group, key=lambda r: r.weight))
# the failure has the lowest reward and a below-1 weight (negative advantage)
fail = group[3]
check("failure has lowest reward", fail.reward.total == min(r.reward.total for r in group))
check("failure weight < 1 (neg advantage)", fail.weight < 1.0)
# collided-but-success ranks below clean success
check("collision penalised below clean success",
      group[2].reward.total < group[0].reward.total)

# manifest round-trips with the expected columns
with tempfile.TemporaryDirectory() as d:
    p = write_manifest(group, Path(d) / "manifest.csv")
    rows = list(csv.DictReader(open(p)))
    check("manifest has one row per rollout", len(rows) == 4)
    need = {"group_id","rollout_id","traj_path","reward","advantage","weight",
            "robot_caused_collision"}
    check("manifest columns present", need <= set(rows[0].keys()))
    check("weights persisted as floats", all(float(r["weight"]) >= 0 for r in rows))

# summary aggregates correctly
s = collection_summary(group)
check("summary success_rate = 3/4", abs(s["success_rate"] - 0.75) < 1e-9)
check("summary robot-caused collision rate = 1/4", abs(s["robot_caused_collision_rate"] - 0.25) < 1e-9)

# positive_only filtering zeros the negative-advantage rollouts
score_group(group, temperature=1.0, positive_only=True)
check("positive_only zeros the failure weight", group[3].weight == 0.0)

print("\nALL PASS" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
