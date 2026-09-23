import os
import re
import time

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from test_env import DroneLandingEnv


# ============================================================
# SETTINGS
# ============================================================

NUM_EPISODES = 10

starting_position = [0.5, 0.5, 0.5]
hover_target = [0.5, -0.5, 1.0]


# ============================================================
# FIND ALL TRAINED MODELS
# ============================================================

model_dir = "ppo_models"

versions = []
for filename in os.listdir(model_dir):
    match = re.match(r"ppo_drone_landing_v(\d+)\.zip$", filename)
    if match:
        versions.append(int(match.group(1)))

if not versions:
    raise FileNotFoundError("No trained models found in ppo_models/")

versions.sort()

print()
print("=" * 60)
print("Available models:")
print("=" * 60)

for i, v in enumerate(versions, start=1):
    model_file = os.path.join(
        model_dir, f"ppo_drone_landing_v{v}.zip"
    )
    stat = os.stat(model_file)
    mtime = time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)
    )
    size_kb = stat.st_size / 1024
    print(f"  {i:2d}. v{v:<3d}   {size_kb:8.1f} KB   saved {mtime}")

print()

while True:
    raw = input(
        f"Choose a version (1-{len(versions)}, "
        f"or 'q' to quit): "
    ).strip()

    if raw.lower() == "q":
        print("Exiting.")
        raise SystemExit

    try:
        choice = int(raw)
        if 1 <= choice <= len(versions):
            chosen_version = versions[choice - 1]
            break
        else:
            print(f"Please enter a number between 1 and {len(versions)}.")
    except ValueError:
        print("Please enter a number, or 'q' to quit.")

model_path = os.path.join(
    model_dir, f"ppo_drone_landing_v{chosen_version}"
)
vec_normalize_path = model_path + "_vecnormalize.pkl"

print()
print(f"Loading model: {model_path}.zip")
print(f"Loading normalization stats: {vec_normalize_path}")
print()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = PPO.load(model_path, device="cpu")


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def make_env():
    return DroneLandingEnv()

vec_env = DummyVecEnv([make_env])

if not os.path.exists(vec_normalize_path):
    print()
    print(f"WARNING: {vec_normalize_path} not found.")
    print("Skipping VecNormalize — observations will NOT be")
    print("normalized. The model may behave strangely if it was")
    print("trained with normalization enabled.")
    print()
    input("Press Enter to continue anyway, or Ctrl-C to abort...")
else:
    vec_env = VecNormalize.load(vec_normalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

env = vec_env.envs[0]


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print(f"      HOVER EVALUATION — v{chosen_version} — {NUM_EPISODES} EPISODES")
print("=" * 70)
print(f"Starting position: {starting_position}")
print(f"Hover target:      {hover_target}")
print("=" * 70)
print()


# ============================================================
# LAUNCH VIEWER ONCE, RUN MULTIPLE EPISODES
# ============================================================

all_distances = []
all_thrusts = []
all_rewards = []
all_lengths = []
all_reasons = []

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    env.viewer = viewer

    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -20.0

    for episode_idx in range(NUM_EPISODES):

        # ------------------------------------------------
        # RESET + OVERRIDE SPAWN
        # ------------------------------------------------
        obs = vec_env.reset()

        env.data.qpos[
            env.model.jnt_qposadr[env.drone_joint_id]:
            env.model.jnt_qposadr[env.drone_joint_id] + 3
        ] = starting_position

        mujoco.mj_forward(env.model, env.data)

        if vec_normalize_path and os.path.exists(vec_normalize_path):
            obs = vec_env.normalize_obs(
                env._get_observation().reshape(1, -1)
            )
        else:
            obs = env._get_observation().reshape(1, -1)

        print(f"--- Episode {episode_idx + 1} / {NUM_EPISODES} ---")

        # ------------------------------------------------
        # EPISODE LOOP
        # ------------------------------------------------
        distances = []
        thrust_values = []
        episode_step = 0
        last_print_step = 0
        episode_reward = 0.0

        while viewer.is_running():

            action, _ = model.predict(obs, deterministic=True)

            thrust = (action[0, 0] + 1.0) / 2.0 * env.max_thrust
            thrust_values.append(float(thrust))

            obs, reward, done, info = vec_env.step(action)

            episode_step += 1
            episode_reward += float(reward[0])

            drone_position = env.data.xpos[env.drone_id].copy()
            distance = float(np.linalg.norm(
                np.array(drone_position) - np.array(hover_target)
            ))
            distances.append(distance)

            viewer.cam.lookat[:] = drone_position

            # ------------- live status every 50 steps -------------
            if episode_step - last_print_step >= 50:
                last_print_step = episode_step
                tilt_deg = info[0].get("tilt_deg", 0.0)
                print(
                    f"  step {episode_step:5d} | "
                    f"pos=({drone_position[0]:+.3f},"
                    f" {drone_position[1]:+.3f},"
                    f" {drone_position[2]:+.3f}) | "
                    f"dist={distance:.3f} | "
                    f"tilt={tilt_deg:5.1f}° | "
                    f"thrust={thrust:.2f}"
                )

            viewer.sync()

            # ------------- episode ended -------------
            if done[0]:
                termination_reason = info[0].get(
                    "termination_reason", "unknown"
                )

                min_d = min(distances) if distances else float("nan")
                avg_d = (sum(distances) / len(distances)
                         if distances else float("nan"))
                avg_t = (sum(thrust_values) / len(thrust_values)
                         if thrust_values else float("nan"))

                print(
                    f"  end | steps={episode_step} | "
                    f"reason={termination_reason} | "
                    f"reward={episode_reward:+.1f} | "
                    f"min_dist={min_d:.3f} | "
                    f"avg_dist={avg_d:.3f} | "
                    f"avg_thrust={avg_t:.2f}"
                )
                print()

                all_distances.extend(distances)
                all_thrusts.extend(thrust_values)
                all_rewards.append(episode_reward)
                all_lengths.append(episode_step)
                all_reasons.append(termination_reason)

                break

            time.sleep(0.002)

        # ------------------------------------------------
        # VIEWER CLOSED MID-RUN
        # ------------------------------------------------
        if not viewer.is_running():
            print("Viewer closed — ending evaluation early.")
            break


# ============================================================
# AGGREGATE SUMMARY
# ============================================================

print("=" * 70)
print(f"                    AGGREGATE SUMMARY — v{chosen_version}")
print("=" * 70)

if all_lengths:
    print(f"Episodes run:            {len(all_lengths)}")
    print(f"Mean episode length:     {sum(all_lengths) / len(all_lengths):.1f} steps")
    print(f"Mean episode reward:     {sum(all_rewards) / len(all_rewards):+.2f}")
    print(f"Mean min distance:       {min(all_distances):.4f} m (best across all)")
    print(f"Overall avg distance:    {sum(all_distances) / len(all_distances):.4f} m")
    print(f"Mean thrust:             {sum(all_thrusts) / len(all_thrusts):.4f} N")
    print()
    print("Termination breakdown:")
    from collections import Counter
    reason_counts = Counter(all_reasons)
    for reason, count in reason_counts.most_common():
        pct = 100.0 * count / len(all_reasons)
        print(f"  {reason:25s} {count:3d}  ({pct:5.1f}%)")

print("=" * 70)

vec_env.close()