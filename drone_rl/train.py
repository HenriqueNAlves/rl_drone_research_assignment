import os
import re
import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for done, info in zip(dones, infos):
            if done:
                self.episode_count += 1

                termination_reason = info.get(
                    "termination_reason", "unknown"
                )
                self.termination_reasons[termination_reason] = (
                    self.termination_reasons.get(termination_reason, 0) + 1
                )

                if termination_reason == "hover_success":
                    self.success_count += 1

        if self.episode_count > 0:
            self.logger.record(
                "rollout/success_rate",
                self.success_count / self.episode_count
            )
            for reason, count in self.termination_reasons.items():
                self.logger.record(
                    f"rollout/termination_{reason}",
                    count / self.episode_count
                )

        return True


# ============================================================
# FOLDERS
# ============================================================

model_dir = "ppo_models"
os.makedirs(model_dir, exist_ok=True)

run_name = "sq_dist_defaults"
run_timestamp = (
    f"{run_name}_"
    f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
)
log_dir = os.path.join("logs", run_timestamp)
os.makedirs(log_dir, exist_ok=True)

print(f"Logging this run to: {log_dir}")


# ============================================================
# FIND THE LATEST MODEL VERSION
# ============================================================

existing_versions = []
for filename in os.listdir(model_dir):
    match = re.match(r"ppo_drone_landing_v(\d+)\.zip$", filename)
    if match:
        existing_versions.append(int(match.group(1)))

next_version = max(existing_versions) + 1 if existing_versions else 1

model_path = os.path.join(
    model_dir, f"ppo_drone_landing_v{next_version}"
)
vec_normalize_path = model_path + "_vecnormalize.pkl"

print(f"Saving model as: {model_path}.zip")


# ============================================================
# ENVIRONMENT
# ============================================================

def make_env():
    return Monitor(DroneLandingEnv())

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)


# ============================================================
# PPO — DEFAULTS (no tuning)
# ============================================================

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
)


# ============================================================
# LOGGER
# ============================================================

new_logger = configure(
    log_dir,
    ["stdout", "csv", "tensorboard"]
)
model.set_logger(new_logger)


# ============================================================
# TRAIN — 5 MILLION STEPS
# ============================================================

stats_callback = TrainingStatsCallback()

model.learn(
    total_timesteps=3_000_000,
    callback=stats_callback,
)


# ============================================================
# SAVE
# ============================================================

model.save(model_path)
env.save(vec_normalize_path)

print(f"Saved normalization stats as: {vec_normalize_path}")

env.close()