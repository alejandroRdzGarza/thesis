"""Quick checks for safe_reward.py (run: python -m experiments.test_safe_reward)."""
from experiments.safe_reward import (
    RewardConfig, compute_episode_reward, is_direct_collision,
    group_advantages, advantage_weights,
)

cfg = RewardConfig()
ok = True
def check(name, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    ok = ok and cond

from experiments.safe_reward import RewardConfig as _RC
cfg_rc = _RC(penalize_raw_collision=False)   # robot_caused/culprit mode (legacy)

# 1. Success, no collision, no CBF use → high reward
r = compute_episode_reward(goal_reached=True, collision_detected=False,
                           cbf_activation_rate=0.0, cfg=cfg)
check("success clean = w_success + w_progress", abs(r.total - (1.5 + 0.3)) < 1e-9)

# 1b. DEFAULT (raw): ANY collision is penalised, regardless of culprit
r_raw = compute_episode_reward(goal_reached=True, collision_detected=True,
                               collision_culprit="scene_object", cfg=cfg)
check("raw mode penalises any collision", r_raw.direct_collision is True
      and abs(r_raw.collision + 1.0) < 1e-9)

# 2. robot_caused mode: indirect (scene_object) collision is NOT penalised
r_ind = compute_episode_reward(goal_reached=True, collision_detected=True,
                               collision_culprit="scene_object", cfg=cfg_rc)
check("scene_object collision not penalised (robot_caused mode)", r_ind.direct_collision is False
      and abs(r_ind.collision) < 1e-9)

# 3. robot_caused mode: direct (arm_link) collision IS penalised
r_dir = compute_episode_reward(goal_reached=True, collision_detected=True,
                               collision_culprit="arm_link|scene_object", cfg=cfg_rc)
check("arm_link collision penalised (robot_caused mode)", r_dir.direct_collision is True
      and abs(r_dir.collision + 1.0) < 1e-9)

# 4. robot_caused mode: collision with no attribution → conservatively direct
r_un = compute_episode_reward(goal_reached=False, collision_detected=True,
                              collision_culprit="", cfg=cfg_rc)
check("unattributed collision treated as direct (robot_caused mode)", r_un.direct_collision is True)

# 5. CBF activation penalty scales with rate and clamps
r_cbf = compute_episode_reward(goal_reached=False, collision_detected=False,
                               cbf_activation_rate=0.5, cfg=cfg)
check("cbf penalty = -w_cbf*rate", abs(r_cbf.cbf + 0.25) < 1e-9)
r_clamp = compute_episode_reward(goal_reached=False, collision_detected=False,
                                 cbf_activation_rate=5.0, cfg=cfg)
check("cbf rate clamped to 1.0", abs(r_clamp.cbf + 0.5) < 1e-9)

# 6. Partial credit: grasp + near goal beats nothing
r_grasp = compute_episode_reward(goal_reached=False, collision_detected=False,
                                 grasp_achieved=True, goal_dist_final=0.0, cfg=cfg)
r_none  = compute_episode_reward(goal_reached=False, collision_detected=False,
                                 grasp_achieved=False, goal_dist_final=0.30, cfg=cfg)
check("grasp+near-goal > nothing", r_grasp.total > r_none.total)

# 6b. robot_caused mode: scene_object BUT robot-caused → penalised
r_chain = compute_episode_reward(goal_reached=True, collision_detected=True,
                                 collision_robot_caused=True,
                                 collision_culprit="scene_object", cfg=cfg_rc)
check("robot-caused domino penalised despite scene_object culprit",
      r_chain.direct_collision is True and abs(r_chain.collision + 1.0) < 1e-9)

# 6c. robot_caused mode: NOT robot-caused → not penalised
r_phys = compute_episode_reward(goal_reached=True, collision_detected=True,
                                collision_robot_caused=False,
                                collision_culprit="scene_object", cfg=cfg_rc)
check("non-robot-caused displacement not penalised (robot_caused mode)",
      r_phys.direct_collision is False and abs(r_phys.collision) < 1e-9)

# 7. Group advantages: standardised, mean ~0
advs = group_advantages([0.0, 1.0, 2.0, 3.0])
check("advantages mean ~ 0", abs(sum(advs) / len(advs)) < 1e-9)
check("advantages ordered", advs[0] < advs[-1])

# 8. Degenerate group (all equal) → all-zero advantages
check("degenerate group → zeros", all(abs(a) < 1e-6 for a in group_advantages([1.0, 1.0, 1.0])))

# 9. Weights: positive_only drops negatives, exp is monotonic
w = advantage_weights([-2.0, 0.5, 2.0], positive_only=True)
check("positive_only zeros negatives", w[0] == 0.0 and w[1] > 0 and w[2] > w[1])

print("\nALL PASS" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
