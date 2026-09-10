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
        existing_versions.append(
            int(match.group(1))
        )


if not existing_versions:

    raise FileNotFoundError(
        "No trained models found in ppo_models/"
    )


latest_version = max(existing_versions)

model_path = os.path.join(
    model_dir,
    f"ppo_drone_landing_v{latest_version}"
)

print(
    f"Loading model: {model_path}.zip"
)


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
# EVALUATION STATISTICS
# ============================================================

total_episodes = len(starting_positions)

successful_episodes = 0

episode_results = []


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

    # --------------------------------------------------------
    # SET INITIAL DRONE POSITION
    # --------------------------------------------------------

    env.data.qpos[
        env.model.jnt_qposadr[env.drone_joint_id]:
        env.model.jnt_qposadr[env.drone_joint_id] + 3
    ] = starting_positions[position_index]

    mujoco.mj_forward(
        env.model,
        env.data
    )

    observation = env._get_observation()

    print()
    print("=" * 70)
    print("                     EVALUATION")
    print("=" * 70)
    print(
        f"Testing {total_episodes} starting positions"
    )
    print("=" * 70)

    print(
        f"\nTESTING STARTING POSITION "
        f"{position_index + 1}/{total_episodes}: "
        f"{starting_positions[position_index]}"
    )

    # ========================================================
    # RUN EVALUATION
    # ========================================================

    while viewer.is_running():

        # ----------------------------------------------------
        # ASK PPO WHAT ACTION TO TAKE
        # ----------------------------------------------------

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        # ----------------------------------------------------
        # APPLY ACTION
        # ----------------------------------------------------

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        # ----------------------------------------------------
        # UPDATE VIEWER
        # ----------------------------------------------------

        viewer.sync()

        # ----------------------------------------------------
        # EPISODE ENDED
        # ----------------------------------------------------

        if terminated or truncated:

            termination_reason = info[
                "termination_reason"
            ]

            # ------------------------------------------------
            # CHECK SUCCESS
            # ------------------------------------------------

            success = (
                termination_reason
                == "successful_landing"
            )

            if success:
                successful_episodes += 1

            # Store result
            episode_results.append(
                {
                    "starting_position":
                        starting_positions[position_index],
                    "success":
                        success,
                    "termination_reason":
                        termination_reason,
                    "reward":
                        reward,
                }
            )

            # ------------------------------------------------
            # PRINT EPISODE RESULT
            # ------------------------------------------------

            print()
            print("-" * 70)

            if success:
                print("RESULT: SUCCESS")
            else:
                print("RESULT: FAILURE")

            print(
                "Termination reason:",
                termination_reason
            )

            print(
                "Reward:",
                reward
            )

            print("-" * 70)

            # ------------------------------------------------
            # MOVE TO NEXT STARTING POSITION
            # ------------------------------------------------

            position_index += 1

            if position_index >= total_episodes:

                # ============================================
                # FINAL EVALUATION RESULTS
                # ============================================

                success_rate = (
                    100.0
                    * successful_episodes
                    / total_episodes
                )

                print()
                print("=" * 70)
                print("                 EVALUATION COMPLETE")
                print("=" * 70)

                print(
                    f"Successful landings: "
                    f"{successful_episodes}/{total_episodes}"
                )

                print(
                    f"SUCCESS RATE: "
                    f"{success_rate:.2f}%"
                )

                print("=" * 70)

                break

            # ------------------------------------------------
            # RESET ENVIRONMENT
            # ------------------------------------------------

            observation, info = env.reset()

            # ------------------------------------------------
            # SET NEW DRONE STARTING POSITION
            # ------------------------------------------------

            env.data.qpos[
                env.model.jnt_qposadr[env.drone_joint_id]:
                env.model.jnt_qposadr[env.drone_joint_id] + 3
            ] = starting_positions[position_index]

            # ------------------------------------------------
            # ZERO DRONE VELOCITY
            # ------------------------------------------------

            env.data.qvel[
                env.model.jnt_dofadr[env.drone_joint_id]:
                env.model.jnt_dofadr[env.drone_joint_id] + 6
            ] = 0.0

            mujoco.mj_forward(
                env.model,
                env.data
            )

            # ------------------------------------------------
            # GET NEW OBSERVATION
            # ------------------------------------------------

            observation = env._get_observation()

            print()
            print(
                f"TESTING STARTING POSITION "
                f"{position_index + 1}/{total_episodes}: "
                f"{starting_positions[position_index]}"
            )

        time.sleep(0.002)


# ============================================================
# CLOSE ENVIRONMENT
# ============================================================

env.close()