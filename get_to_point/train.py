import os
import re
import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from test_env import DroneLandingEnv


# ============================================================
# CONFIG
# ============================================================

# Set to True for a single DummyVecEnv (no subprocesses, easier
# to debug). Set to False to use SubprocVecEnv with N_ENVS
# parallel workers.
SINGLE_ENV = False

N_ENVS = 1 if SINGLE_ENV else 8

TOTAL_TIMESTEPS = 5_000_000
N_STEPS = 1024
BATCH_SIZE = 256
N_EPOCHS = 5
LEARNING_RATE = 3e-4
ENT_COEF = 0.005
NET_ARCH = [128, 128]
DEVICE = "cpu"


# ============================================================
# ENV FACTORY
# ============================================================

def make_env():
    return Monitor(DroneLandingEnv())


# ============================================================
# BEST-MEAN-REWARD CALLBACK
# ============================================================

class BestRewardCallback(BaseCallback):
    """
    Tracks the highest rolling-mean episode reward seen so far.
    Saves model + matching VecNormalize stats together.
    """

    def __init__(self, save_path, vec_env, verbose=0, window=100):
        super().__init__(verbose)
        self.save_path = save_path
        self.vec_env = vec_env
        self.best_mean_reward = -np.inf

        self.recent_rewards = []
        self.window = window

        self.episode_count = 0
        self.success_count = 0
        self.termination_reasons = {}

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]

        for done, info, rew in zip(dones, infos, rewards):
            if done:
                self.episode_count += 1
                self.recent_rewards.append(float(rew))
                if len(self.recent_rewards) > self.window:
                    self.recent_rewards.pop(0)

                reason = info.get("termination_reason", "unknown")
                self.termination_reasons[reason] = (
                    self.termination_reasons.get(reason, 0) + 1
                )
                if reason == "success":
                    self.success_count += 1

        if len(self.recent_rewards) >= self.window:
            mean_reward = float(np.mean(self.recent_rewards))

            self.logger.record(
                "rollout/mean_reward_window", mean_reward
            )
            if self.episode_count > 0:
                self.logger.record(
                    "rollout/success_rate",
                    self.success_count / self.episode_count,
                )
                for reason, count in self.termination_reasons.items():
                    self.logger.record(
                        f"rollout/termination_{reason}",
                        count / self.episode_count,
                    )

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
                self.vec_env.save(self.save_path + "_vecnormalize.pkl")
                if self.verbose > 0:
                    print(f"[BEST] New best mean reward: "
                          f"{mean_reward:.2f} at step {self.num_timesteps}")

        return True


# ============================================================
# MAIN
# ============================================================

def main():

    model_dir = "ppo_models"
    os.makedirs(model_dir, exist_ok=True)

    tensorboard_root = "tensorboard"
    os.makedirs(tensorboard_root, exist_ok=True)

    # --- next version ---
    existing_versions = []
    for filename in os.listdir(model_dir):
        match = re.match(r"ppo_drone_landing_v(\d+)(_best)?\.zip$", filename)
        if match:
            existing_versions.append(int(match.group(1)))

    next_version = max(existing_versions) + 1 if existing_versions else 1

    final_model_path = os.path.join(
        model_dir, f"ppo_drone_landing_v{next_version}"
    )
    best_model_path = os.path.join(
        model_dir, f"ppo_drone_landing_v{next_version}_best"
    )

    run_name = f"v{next_version}_point_to_point_vect{N_ENVS}"
    run_timestamp = (
        f"{run_name}_"
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    run_dir = os.path.join(tensorboard_root, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    print(f"TensorBoard run dir:       {run_dir}")
    print(f"Final model will go to:    {final_model_path}.zip")
    print(f"Best  model will go to:    {best_model_path}.zip")
    print(f"Total timesteps:           {TOTAL_TIMESTEPS:,}")

    if SINGLE_ENV:
        print("Env mode:                  SINGLE (DummyVecEnv, 1 worker)")
    else:
        print(f"Env mode:                  PARALLEL (SubprocVecEnv, {N_ENVS} workers)")

    print()
    print(f"    tensorboard --logdir {tensorboard_root}")
    print()

    # --- training env ---
    if SINGLE_ENV:
        env = DummyVecEnv([make_env])
    else:
        env = SubprocVecEnv([make_env for _ in range(N_ENVS)])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # --- PPO ---
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=DEVICE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        ent_coef=ENT_COEF,
        policy_kwargs=dict(net_arch=NET_ARCH),
        tensorboard_log=run_dir,
    )

    new_logger = configure(run_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # --- callbacks ---
    best_callback = BestRewardCallback(
        save_path=best_model_path,
        vec_env=env,
        verbose=1,
        window=100,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix=f"ppo_v{next_version}",
        save_vecnormalize=True,
        verbose=0,
    )

    # --- train ---
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[best_callback, checkpoint_callback],
        tb_log_name=run_name,
    )

    # --- save final ---
    model.save(final_model_path)
    env.save(final_model_path + "_vecnormalize.pkl")

    print(f"Saved final model: {final_model_path}.zip")
    print(f"Saved best  model: {best_model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()