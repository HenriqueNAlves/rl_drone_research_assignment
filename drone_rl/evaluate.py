import os
import re
import time

import mujoco

from stable_baselines3 import PPO
from test_env import DroneLandingEnv


# ============================================================
# FIND LATEST TRAINED MODEL
# ============================================================

model_dir = "ppo_models"

existing_versions = []

for filename in os.listdir(model_dir):

    match = re.match(
        r"ppo_drone_landing_v(\d+)\.zip$",
        filename
    )

    if match:
        existing_versions.append(int(match.group(1)))


if not existing_versions:
    raise FileNotFoundError(
        "No trained models found in ppo_models/"
    )


latest_version = max(existing_versions)

model_path = os.path.join(
    model_dir,
    f"ppo_drone_landing_v{latest_version}"
)

print(f"Loading model: {model_path}.zip")


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = PPO.load(model_path)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = DroneLandingEnv()


# ============================================================
# TEST STARTING POSITIONS
# ============================================================

starting_positions = [
    [0.0, 0.0, 1.0],
    [0.2, 0.0, 1.0],
    [-0.2, 0.0, 1.0],
    [0.0, 0.2, 1.0],
    [0.0, -0.2, 1.0],
    [0.4, 0.0, 1.0],
    [-0.4, 0.0, 1.0],
    [0.0, 0.4, 1.0],
    [0.0, -0.4, 1.0],
]


# ============================================================
# CREATE VIEWER
# ============================================================

observation, info = env.reset()

with mujoco.viewer.launch_passive(
    env.model,
    env.data
) as viewer:

    env.viewer = viewer

    position_index = 0

    # Set initial drone position
    env.data.qpos[
        env.model.jnt_qposadr[env.drone_joint_id]:
        env.model.jnt_qposadr[env.drone_joint_id] + 3
    ] = starting_positions[position_index]

    mujoco.mj_forward(env.model, env.data)

    observation = env._get_observation()

    print(
        f"\nTESTING STARTING POSITION "
        f"{position_index + 1}/{len(starting_positions)}: "
        f"{starting_positions[position_index]}"
    )

    while viewer.is_running():

        # Ask PPO what action to take
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        # Apply action to our environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Update viewer
        viewer.sync()

        # Print episode result
        if terminated or truncated:

            print(
                "EPISODE ENDED:",
                info["termination_reason"],
                "Reward:",
                reward
            )

            # Move to next starting position
            position_index += 1

            if position_index >= len(starting_positions):
                print("\nFinished testing all starting positions.")
                break

            # Reset environment
            observation, info = env.reset()

            # Set new drone starting position
            env.data.qpos[
                env.model.jnt_qposadr[env.drone_joint_id]:
                env.model.jnt_qposadr[env.drone_joint_id] + 3
            ] = starting_positions[position_index]

            # Zero the drone velocity
            env.data.qvel[
                env.model.jnt_dofadr[env.drone_joint_id]:
                env.model.jnt_dofadr[env.drone_joint_id] + 6
            ] = 0.0

            mujoco.mj_forward(env.model, env.data)

            # Get observation corresponding to new position
            observation = env._get_observation()

            print(
                f"\nTESTING STARTING POSITION "
                f"{position_index + 1}/{len(starting_positions)}: "
                f"{starting_positions[position_index]}"
            )

        time.sleep(0.002)


env.close()