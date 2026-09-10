import os

import re

import gymnasium as gym

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import BaseCallback

from test_env import DroneLandingEnv


# ============================================================
# TRAINING STATISTICS CALLBACK
# ============================================================

class TrainingStatsCallback(BaseCallback):

    def __init__(self, verbose=0):

        super().__init__(verbose)

        self.success_count = 0

        self.episode_count = 0

        self.termination_reasons = {}


    def _on_step(self) -> bool:

        # Check whether an episode has ended
        dones = self.locals["dones"]

        infos = self.locals["infos"]

        for done, info in zip(dones, infos):

            if done:

                self.episode_count += 1

                # --------------------------------------------
                # TERMINATION REASON
                # --------------------------------------------

                termination_reason = info.get(
                    "termination_reason",
                    "unknown"
                )

                if termination_reason not in self.termination_reasons:

                    self.termination_reasons[
                        termination_reason
                    ] = 0

                self.termination_reasons[
                    termination_reason
                ] += 1


                # --------------------------------------------
                # SUCCESS
                # --------------------------------------------

                if termination_reason == "successful_landing":

                    self.success_count += 1


        # --------------------------------------------
        # LOG STATISTICS
        # --------------------------------------------

        if self.episode_count > 0:

            success_rate = (
                self.success_count
                / self.episode_count
            )

            self.logger.record(
                "rollout/success_rate",
                success_rate
            )


            # --------------------------------------------
            # TERMINATION REASON DISTRIBUTION
            # --------------------------------------------

            for reason, count in self.termination_reasons.items():

                percentage = (
                    count
                    / self.episode_count
                )

                self.logger.record(
                    f"rollout/termination_{reason}",
                    percentage
                )


        return True


# ============================================================
# FOLDER FOR TRAINED MODELS
# ============================================================

model_dir = "ppo_models"

os.makedirs(
    model_dir,
    exist_ok=True
)


# ============================================================
# FIND THE LATEST MODEL VERSION
# ============================================================

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


if existing_versions:

    next_version = max(existing_versions) + 1

else:

    next_version = 1


model_path = os.path.join(
    model_dir,
    f"ppo_drone_landing_v{next_version}"
)

print(
    f"Saving model as: {model_path}.zip"
)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = DroneLandingEnv()


# ============================================================
# CREATE PPO AGENT
# ============================================================

model = PPO(

    "MlpPolicy",

    env,

    verbose=1,

)


# ============================================================
# CREATE TRAINING CALLBACK
# ============================================================

stats_callback = TrainingStatsCallback()


# ============================================================
# TRAIN
# ============================================================

model.learn(

    total_timesteps=2_500_000,

    callback=stats_callback

)


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

model.save(model_path)


# ============================================================
# CLOSE ENVIRONMENT
# ============================================================

env.close()