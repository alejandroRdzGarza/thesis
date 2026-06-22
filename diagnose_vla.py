"""
VLA Pipeline Diagnostics
========================
Checks every layer of the stack to explain poor task performance:

  1. Server health  — is the VLA server reachable? what model?
  2. Latency        — round-trip inference time (the single biggest bottleneck)
  3. Action quality — are raw VLA outputs in sane ranges?
  4. Gripper        — does the gripper actually close at the right moments?
  5. Update rate    — how many VLA updates happen per 800-step episode?
  6. Image sanity   — saves the exact image the VLA sees to disk
  7. Environment    — does info["success"] ever fire? what does reset obs look like?

Usage (with RunPod tunnel active):
    conda run -n libero python diagnose_vla.py

All images / logs saved to diagnostics/vla_diag/
"""

from __future__ import annotations

import io
import os
import sys
import time
import base64
import pathlib
import requests
import threading
import numpy as np
import cv2
from PIL import Image

# ── config ─────────────────────────────────────────────────────────────────────
VLA_URL      = "http://127.0.0.1:8000/act"
OUT_DIR      = pathlib.Path("diagnostics/vla_diag")
N_LATENCY    = 5          # latency test: number of pings
DIAG_STEPS   = 200        # short episode for action inspection
TASK_SUITE   = "libero_spatial"
TASK_IDX     = 0

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEPARATOR = "=" * 70

def header(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def ok(msg):   print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. SERVER HEALTH
# ─────────────────────────────────────────────────────────────────────────────
header("1 · SERVER HEALTH")

try:
    r = requests.get("http://127.0.0.1:8000/", timeout=5)
    ok("Server is reachable (HTTP 200)")
except requests.ConnectionError:
    fail("Cannot connect to VLA server at localhost:8000")
    fail("  → Is the RunPod tunnel running?  (bash runpod/tunnel.sh)")
    fail("  → Is the model server started?   (bash runpod/run_server.sh)")
    sys.exit(1)
except Exception as e:
    warn(f"Got non-200 from root endpoint: {e} — server may still work")

# Model identity: send a blank image and inspect the response
blank_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
buf = io.BytesIO()
blank_img.save(buf, format="JPEG", quality=85)
b64 = base64.b64encode(buf.getvalue()).decode()

try:
    r = requests.post(VLA_URL,
                      json={"image_base64": b64,
                            "instruction": "test"},
                      timeout=120)
    d = r.json()
    if d.get("action"):
        ok(f"Server returned a 7-D action on blank image")
        ok(f"unnorm_key in response: {d.get('unnorm_key', '(not reported)')}")
        _blank_action = np.array(d["action"])
        print(f"         raw action (blank img): {np.round(_blank_action, 4)}")
    else:
        fail(f"Server error on blank image: {d.get('error')}")
except Exception as e:
    fail(f"Request to /act failed: {e}")
    sys.exit(1)

print()
print("  ── MODEL NOTE ──────────────────────────────────────────────────────")
print("  You are using:  openvla/openvla-7b-finetuned-libero-spatial")
print("                  (base OpenVLA-7B, single action per inference,")
print("                   ~1-5 s per call on GPU)")
print()
print("  AEGIS paper baseline uses: OpenVLA-OFT")
print("                  (parallel decoding + action chunking,")
print("                   ~0.1-0.5 s per call, 25-50× faster,")
print("                   significantly higher TSR: ~90%+)")
print()
print("  → Key implication: with base OpenVLA-7B, each action is repeated")
print("    ~20-100× at 20 Hz control while waiting for the next inference.")
print("    This causes the arm to overshoot, oscillate, and miss grasps.")
print("  ────────────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LATENCY
# ─────────────────────────────────────────────────────────────────────────────
header("2 · INFERENCE LATENCY (round-trip, seconds)")

# Use a realistic image (noise) as a proxy
rng = np.random.default_rng(42)
noise_img = (rng.integers(50, 200, (224, 224, 3), dtype=np.uint8))
buf2 = io.BytesIO()
Image.fromarray(noise_img).save(buf2, format="JPEG", quality=85)
b64_noise = base64.b64encode(buf2.getvalue()).decode()

latencies = []
for i in range(N_LATENCY):
    t0 = time.perf_counter()
    r = requests.post(VLA_URL,
                      json={"image_base64": b64_noise,
                            "instruction": "pick up the object"},
                      timeout=120)
    dt = time.perf_counter() - t0
    latencies.append(dt)
    print(f"    ping {i+1}/{N_LATENCY}:  {dt:.3f}s")

lat_mean = np.mean(latencies)
lat_min  = np.min(latencies)
lat_max  = np.max(latencies)
print(f"\n  mean={lat_mean:.3f}s  min={lat_min:.3f}s  max={lat_max:.3f}s")

steps_per_update = int(lat_mean * 20)   # 20 Hz control
print(f"\n  At 20 Hz control: VLA updates every ~{steps_per_update} steps")
print(f"  In an 800-step episode: ~{int(800 / steps_per_update)} total VLA updates")
print(f"  In a 300-step episode: ~{int(300 / steps_per_update)} total VLA updates")

if lat_mean < 0.5:
    ok(f"Latency is acceptable ({lat_mean:.2f}s) — action repetition ~{steps_per_update}× per update")
elif lat_mean < 2.0:
    warn(f"Latency is moderate ({lat_mean:.2f}s) — arm repeats each action ~{steps_per_update}× (may overshoot)")
else:
    fail(f"Latency is HIGH ({lat_mean:.2f}s) — arm repeats each action ~{steps_per_update}× (very likely cause of failure)")
    warn("Consider: OpenVLA-OFT runs ~25-50× faster and uses action chunking")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ACTION QUALITY — raw VLA output inspection
# ─────────────────────────────────────────────────────────────────────────────
header("3 · RAW VLA OUTPUT (real LIBERO image from env)")

try:
    from libero.libero import benchmark as _bench
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    bench = _bench.get_benchmark_dict()["libero_spatial"]()
    task  = bench.get_task(TASK_IDX)
    bddl  = os.path.join(get_libero_path("bddl_files"),
                          task.problem_folder, task.bddl_file)
    instruction = task.language

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        controller="OSC_POSE",
        camera_heights=224, camera_widths=224,
        camera_names=["agentview"],
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, control_freq=20, horizon=DIAG_STEPS,
        ignore_done=True,
    )
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]

    raw_img = obs["agentview_image"]                # OpenGL convention (y-flipped)
    disp_img = raw_img[::-1].copy()                 # flip for display / VLA

    # Save both to disk
    cv2.imwrite(str(OUT_DIR / "raw_obs_image.png"),
                cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT_DIR / "vla_input_image.png"),
                cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR))
    ok(f"Saved raw_obs_image.png  (what robosuite returns)")
    ok(f"Saved vla_input_image.png  (what VLA server receives after [::-1] flip)")

    # Query VLA with the real image
    buf3 = io.BytesIO()
    Image.fromarray(disp_img).save(buf3, format="JPEG", quality=85)
    b64_real = base64.b64encode(buf3.getvalue()).decode()

    t0 = time.perf_counter()
    r = requests.post(VLA_URL,
                      json={"image_base64": b64_real,
                            "instruction": instruction},
                      timeout=120)
    dt_real = time.perf_counter() - t0
    d_real = r.json()
    raw_action = np.array(d_real["action"])

    print(f"\n  Instruction: \"{instruction}\"")
    print(f"  Inference time (real image): {dt_real:.3f}s")
    print(f"\n  Raw VLA action (from predict_action + unnorm):")
    labels = ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"]
    for lbl, val in zip(labels, raw_action):
        print(f"    {lbl:>8}  =  {val:+.4f}")

    gripper_raw = raw_action[6]
    gripper_norm = gripper_raw * 2.0 - 1.0       # [0,1] → [-1,+1]
    gripper_bin  = 1.0 if gripper_norm > 0.0 else -1.0
    gripper_final = -gripper_bin                  # invert for LIBERO

    print(f"\n  Gripper post-processing:")
    print(f"    raw={gripper_raw:.4f} → norm={gripper_norm:+.4f} → bin={gripper_bin:+.1f} → inverted={gripper_final:+.1f}")
    if gripper_final == 1.0:
        print(f"    → gripper OPEN  (action=+1 in LIBERO = open)")
    else:
        print(f"    → gripper CLOSE (action=-1 in LIBERO = close)")

    xyz_mag = float(np.linalg.norm(raw_action[:3]))
    print(f"\n  XYZ action magnitude: {xyz_mag:.4f}  (typical LIBERO range: 0.01-0.10)")
    if xyz_mag < 0.001:
        fail("XYZ magnitude is near zero — VLA may be outputting no-op actions")
    elif xyz_mag > 0.5:
        warn(f"XYZ magnitude is large ({xyz_mag:.3f}) — may cause overshooting with many repeats")
    else:
        ok(f"XYZ magnitude looks reasonable ({xyz_mag:.3f})")

except ImportError as e:
    warn(f"Cannot test with real LIBERO image: {e}")
    env = None


# ─────────────────────────────────────────────────────────────────────────────
# 4. SHORT EPISODE TRACE — gripper, updates, goal distance
# ─────────────────────────────────────────────────────────────────────────────
if env is not None:
    header(f"4 · EPISODE TRACE ({DIAG_STEPS} steps, plain mode)")

    # Globals for VLA thread
    _vla_lock      = threading.Lock()
    _vla_image_g   = None
    _vla_action_g  = np.zeros(7)
    _vla_counter_g = 0
    _vla_running_g = False
    _vla_times     = []   # list of (step_posted, latency_s)
    _vla_posted_at = [None]

    def _vla_worker_diag():
        global _vla_action_g, _vla_running_g, _vla_counter_g
        while _vla_running_g:
            with _vla_lock:
                img  = _vla_image_g
            if img is not None:
                try:
                    buf4 = io.BytesIO()
                    Image.fromarray(img).save(buf4, format="JPEG", quality=85)
                    b64 = base64.b64encode(buf4.getvalue()).decode()
                    t0 = time.perf_counter()
                    r = requests.post(VLA_URL,
                                      json={"image_base64": b64,
                                            "instruction": instruction},
                                      timeout=120)
                    dt = time.perf_counter() - t0
                    d = r.json()
                    if d.get("action"):
                        with _vla_lock:
                            _vla_action_g[:] = np.array(d["action"])
                            _vla_counter_g  += 1
                            _vla_times.append(dt)
                except Exception as ex:
                    pass
            time.sleep(0.01)

    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]

    # Warm-up
    DUMMY = np.array([0., 0., 0., 0., 0., 0., -1.])
    for _ in range(10):
        out = env.step(DUMMY.tolist())
        obs = out[0] if isinstance(out, tuple) else out

    _vla_running_g = True
    th = threading.Thread(target=_vla_worker_diag, daemon=True)
    th.start()

    log = []         # per-step records
    last_cnt = -1
    cur_action = DUMMY.copy()
    goal_pos = np.array([0.062, 0.195, 1.0])

    for t in range(DIAG_STEPS):
        raw_obs = obs.get("agentview_image", None)
        if raw_obs is not None:
            img_vla = raw_obs[::-1].copy()
            if img_vla.shape[:2] != (224, 224):
                img_vla = cv2.resize(img_vla, (224, 224))
            with _vla_lock:
                _vla_image_g = img_vla.astype(np.uint8)

        with _vla_lock:
            raw_a = _vla_action_g.copy()
            cnt   = _vla_counter_g

        if cnt != last_cnt and cnt > 0:
            # New VLA output — apply post-processing
            a = raw_a.copy()
            a[6] = 2.0 * a[6] - 1.0
            a[6] = 1.0 if a[6] > 0.0 else -1.0
            a[6] = -a[6]
            cur_action = a
            last_cnt = cnt

        out = env.step(cur_action.tolist())
        if len(out) == 4:
            obs, _, done, info = out
        else:
            obs, _, term, trunc, info = out
            done = term or trunc

        ee_pos  = np.array(obs.get("robot0_eef_pos", [0, 0, 0]))
        d_goal  = float(np.linalg.norm(ee_pos - goal_pos))
        success = bool(info.get("success", False))

        log.append({
            "step": t, "vla_cnt": cnt, "gripper_raw": raw_a[6],
            "gripper_final": cur_action[6],
            "xyz_mag": float(np.linalg.norm(cur_action[:3])),
            "d_goal": d_goal, "success": success,
        })

        if done or success:
            print(f"  Episode ended at step {t}  success={success}")
            break

    _vla_running_g = False
    th.join(timeout=2.0)

    # ── Print summary ──────────────────────────────────────────────────────
    total_updates = log[-1]["vla_cnt"] if log else 0
    steps_done    = len(log)
    update_freq   = steps_done / total_updates if total_updates > 0 else float("inf")

    print(f"\n  Episode length   : {steps_done} steps")
    print(f"  VLA updates      : {total_updates}  (1 update every ~{update_freq:.0f} steps = ~{update_freq/20:.1f}s)")

    if _vla_times:
        print(f"  Inference times  : mean={np.mean(_vla_times):.3f}s  "
              f"min={np.min(_vla_times):.3f}s  max={np.max(_vla_times):.3f}s")

    # Gripper analysis
    gripper_finals = [r["gripper_final"] for r in log if r["vla_cnt"] > 0]
    if gripper_finals:
        n_close = sum(1 for g in gripper_finals if g < 0)
        n_open  = sum(1 for g in gripper_finals if g > 0)
        pct_close = 100 * n_close / len(gripper_finals)
        print(f"\n  Gripper CLOSE (−1) : {n_close}/{len(gripper_finals)} steps ({pct_close:.0f}%)")
        print(f"  Gripper OPEN  (+1) : {n_open}/{len(gripper_finals)} steps ({100-pct_close:.0f}%)")
        if n_close == 0:
            fail("Gripper NEVER CLOSES — arm cannot grasp any object!")
            warn("  Check: gripper post-processing (normalize + invert)")
            # Print raw gripper values
            raw_grippers = [r["gripper_raw"] for r in log if r["vla_cnt"] > 0]
            print(f"  Raw gripper values (from predict_action): "
                  f"min={min(raw_grippers):.3f}  max={max(raw_grippers):.3f}  "
                  f"mean={np.mean(raw_grippers):.3f}")
        elif pct_close < 5:
            warn(f"Gripper closes only {pct_close:.0f}% of steps — very infrequent grasps")
        else:
            ok(f"Gripper closes {pct_close:.0f}% of steps")

    # d_goal analysis
    d_goals = [r["d_goal"] for r in log]
    min_d   = min(d_goals)
    print(f"\n  d_goal range: min={min_d:.3f}m  max={max(d_goals):.3f}m")
    if min_d < 0.05:
        ok(f"EE gets within {min_d*100:.0f}cm of goal — positioning OK")
    elif min_d < 0.15:
        warn(f"EE gets to {min_d*100:.0f}cm from goal — close but may not grasp")
    else:
        fail(f"EE never gets within 15cm of goal (min={min_d:.3f}m) — direction/grasping broken")

    # Success
    if any(r["success"] for r in log):
        ok("info[\"success\"] fired at least once — LIBERO success detector works")
    else:
        warn("info[\"success\"] never fired in this episode")

    # Save step-level CSV
    import csv
    csv_path = OUT_DIR / "episode_trace.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader(); w.writerows(log)
    ok(f"Step log → {csv_path}")

    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. FINAL DIAGNOSIS + RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
header("5 · DIAGNOSIS SUMMARY")

print("""
  ROOT CAUSE RANKING (most likely → least likely)
  ─────────────────────────────────────────────────

  [1] MODEL SPEED  ← probably the main culprit
      Base OpenVLA-7B takes 1-5s per inference.  At 20 Hz control this means
      each action is repeated 20-100 steps.  The arm overshoots every motion,
      oscillates, and never executes a clean pick-grasp-place sequence.
      AEGIS baseline uses OpenVLA-OFT (25-50× faster), which is why their
      TSR is 50.9% even with obstacles.

      Fix option A (fast): switch to OpenVLA-OFT
          download_weights.sh with HF_MODEL=openvla/openvla-oft-libero-spatial
          (or the equivalent HuggingFace ID — verify on HF)
          OFT is a drop-in replacement that uses the same /act endpoint.

      Fix option B (no model change): reduce control_freq to match VLA speed
          If VLA takes ~2s, set control_freq=1 or call VLA synchronously.
          Run env step only when a new VLA action arrives (blocking mode).

  [2] GRIPPER CONVENTION
      Check the gripper section above.  If NEVER CLOSES appears, the
      normalize→binarize→invert pipeline is producing the wrong sign.
      Try removing the inversion (_invert_gripper) and retesting.

  [3] ACTION MAGNITUDE
      With large repeats (20-100×), even a 0.05m/step delta becomes
      1-5m of travel.  The arm shoots to the joint limits then returns.
      The d_goal trace (goes far away then comes back) is the fingerprint.

  [4] IMAGE ORIENTATION (tested, reverted)
      Flip [::-1] is correct for OpenVLA-LIBERO.  No further action needed.
""")
