import os
import re
import time
from collections import Counter

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from test_env import DroneLandingEnv


# ============================================================
# SETTINGS
# ============================================================

NUM_EPISODES = 10


# ============================================================
# DISCOVER RUNS
# ============================================================

def find_final_models():
    """Scan ppo_models/ for vXX.zip AND vXX_best.zip files."""
    runs = {}
    model_dir = "ppo_models"
    if not os.path.isdir(model_dir):
        return runs

    for filename in os.listdir(model_dir):
        match = re.match(
            r"ppo_drone_landing_v(\d+)(_best)?\.zip$", filename
        )
        if not match:
            continue
        v = int(match.group(1))
        is_best = match.group(2) == "_best"
        label = "best" if is_best else "final"

        model_path = os.path.join(model_dir, filename[:-4])
        pkl_path = model_path + "_vecnormalize.pkl"

        runs.setdefault(v, {"source": "ppo_models", "models": []})
        runs[v]["models"].append({
            "label": label,
            "model_path": model_path,
            "pkl_path": pkl_path if os.path.exists(pkl_path) else None,
        })
    return runs


def find_log_runs():
    """Scan tensorboard/ for run dirs with checkpoints."""
    runs = []
    log_dir = "tensorboard"
    if not os.path.isdir(log_dir):
        return runs

    for name in sorted(os.listdir(log_dir)):
        run_dir = os.path.join(log_dir, name)
        if not os.path.isdir(run_dir):
            continue

        models = []

        ckpt_dir = os.path.join(run_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            for ckpt in sorted(os.listdir(ckpt_dir)):
                if not ckpt.endswith(".zip"):
                    continue
                base = os.path.join(ckpt_dir, ckpt[:-4])
                pkl_path = base + "_vecnormalize.pkl"
                models.append({
                    "label": ckpt[:-4],
                    "model_path": base,
                    "pkl_path": pkl_path if os.path.exists(pkl_path) else None,
                })

        if models:
            runs.append({"name": name, "dir": run_dir, "models": models})

    return runs


final_runs = find_final_models()
log_runs = find_log_runs()


# ============================================================
# LEVEL 1: CHOOSE A RUN
# ============================================================

print()
print("=" * 70)
print("Available runs:")
print("=" * 70)

run_choices = []

for v in sorted(final_runs.keys()):
    for m in final_runs[v]["models"]:
        stat = os.stat(m["model_path"] + ".zip")
        mtime = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)
        )
        run_choices.append({
            "display": f"v{v:<3d} [{m['label']:<5s}]    {mtime}",
            "models": [m],
        })

for run in log_runs:
    def sort_key(m):
        match = re.search(r"_(\d+)_steps", m["label"])
        if match:
            return int(match.group(1))
        return 0

    run["models"].sort(key=sort_key)

    stat = os.stat(run["models"][0]["model_path"] + ".zip")
    mtime = time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)
    )
    run_choices.append({
        "display": f"{run['name'][:40]:<40} ({len(run['models'])} models)",
        "models": run["models"],
    })

if not run_choices:
    raise FileNotFoundError("No trained models found.")

for i, choice in enumerate(run_choices, start=1):
    print(f"  {i:2d}. {choice['display']}")

print()


def prompt_choice(prompt, n_options):
    while True:
        raw = input(f"{prompt} (1-{n_options}, or 'q'): ").strip()
        if raw.lower() == "q":
            print("Exiting.")
            raise SystemExit
        try:
            c = int(raw)
            if 1 <= c <= n_options:
                return c - 1
            print(f"Please enter a number between 1 and {n_options}.")
        except ValueError:
            print("Please enter a number, or 'q' to quit.")


choice_idx = prompt_choice("Choose a run", len(run_choices))
chosen_run = run_choices[choice_idx]


# ============================================================
# LEVEL 2: CHOOSE A MODEL WITHIN THE RUN
# ============================================================

if len(chosen_run["models"]) == 1:
    chosen_model = chosen_run["models"][0]
    print(f"Only one model in this run: {chosen_model['label']}")
else:
    print()
    print("=" * 70)
    print("Models in this run:")
    print("=" * 70)
    for i, m in enumerate(chosen_run["models"], start=1):
        stat = os.stat(m["model_path"] + ".zip")
        mtime = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)
        )
        size_kb = stat.st_size / 1024
        print(f"  {i:2d}. {m['label']:<30} {size_kb:7.1f} KB   {mtime}")
    print()

    model_idx = prompt_choice(
        "Choose a model", len(chosen_run["models"])
    )
    chosen_model = chosen_run["models"][model_idx]

model_path = chosen_model["model_path"]
vec_normalize_path = chosen_model["pkl_path"]

print()
print(f"Loading model: {model_path}.zip")
if vec_normalize_path:
    print(f"Loading normalization stats: {vec_normalize_path}")
else:
    print("WARNING: no matching VecNormalize stats found.")
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

use_normalization = vec_normalize_path is not None

if not use_normalization:
    print()
    print("Continuing WITHOUT observation normalization.")
    print("The model may behave strangely if it was trained with")
    print("normalization enabled.")
    print()
    input("Press Enter to continue anyway, or Ctrl-C to abort...")
else:
    vec_env = VecNormalize.load(vec_normalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

env = vec_env.envs[0]


# ============================================================
# DIAGNOSTIC
# ============================================================

print()
print("=" * 70)
print("DIAGNOSTIC")
print("=" * 70)

model_obs_shape = model.observation_space.shape
print(f"Model expects obs shape:    {model_obs_shape}")

env_obs_shape = env.observation_space.shape
print(f"Env provides obs shape:     {env_obs_shape}")

if model_obs_shape != env_obs_shape:
    print(">>> MISMATCH — model and env obs dims differ!")
else:
    print("    (obs dims match — good)")

print()
print(f"vec_env type:               {type(vec_env).__name__}")

if isinstance(vec_env, VecNormalize):
    print(f"vec_env.training:           {vec_env.training}")
    print(f"vec_env.norm_obs:           {vec_env.norm_obs}")

    if hasattr(vec_env, "obs_rms"):
        mean = vec_env.obs_rms.mean
        var = vec_env.obs_rms.var
        print(f"obs_rms.mean shape:         {mean.shape}")
        print(f"obs_rms.mean[:5]:           {np.array2string(mean[:5], precision=4)}")
        print(f"obs_rms.var[:5]:            {np.array2string(var[:5], precision=4)}")
        print(f"obs_rms.count:              {vec_env.obs_rms.count}")

print("=" * 70)
print()


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 70)
print(f"      EVALUATION — {chosen_model['label']} — {NUM_EPISODES} EPISODES")
print("=" * 70)
print()


# ============================================================
# RUN EPISODES WITH VIEWER
# ============================================================

all_distances = []
all_thrusts = []
all_rewards = []
all_lengths = []
all_reasons = []

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

    env.viewer = viewer

    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -20.0

    for episode_idx in range(NUM_EPISODES):

        obs = vec_env.reset()
        target = env.target_position.copy()

        print(f"--- Episode {episode_idx + 1} / {NUM_EPISODES} ---")
        print(f"    target = {target}")

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
            distance = float(np.linalg.norm(drone_position - target))
            distances.append(distance)

            viewer.cam.lookat[:] = drone_position

            if episode_step - last_print_step >= 50:
                last_print_step = episode_step
                print(
                    f"  step {episode_step:5d} | "
                    f"pos=({drone_position[0]:+.3f},"
                    f" {drone_position[1]:+.3f},"
                    f" {drone_position[2]:+.3f}) | "
                    f"dist={distance:.3f} | "
                    f"thrust={thrust:.2f}"
                )

            viewer.sync()

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
                    f"reward={episode_reward:+.2f} | "
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

        if not viewer.is_running():
            print("Viewer closed — ending evaluation early.")
            break


# ============================================================
# AGGREGATE SUMMARY
# ============================================================

print("=" * 70)
print(f"          AGGREGATE SUMMARY — {chosen_model['label']}")
print("=" * 70)

if all_lengths:
    print(f"Episodes run:            {len(all_lengths)}")
    print(f"Mean episode length:     {sum(all_lengths) / len(all_lengths):.1f} steps")
    print(f"Mean episode reward:     {sum(all_rewards) / len(all_rewards):+.2f}")
    print(f"Mean min distance:       {min(all_distances):.4f} m")
    print(f"Overall avg distance:    {sum(all_distances) / len(all_distances):.4f} m")
    print(f"Mean thrust:             {sum(all_thrusts) / len(all_thrusts):.4f} N")
    print()
    print("Termination breakdown:")
    reason_counts = Counter(all_reasons)
    for reason, count in reason_counts.most_common():
        pct = 100.0 * count / len(all_reasons)
        print(f"  {reason:25s} {count:3d}  ({pct:5.1f}%)")

print("=" * 70)

vec_env.close()