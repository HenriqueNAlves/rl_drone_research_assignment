import os
import re

import gymnasium as gym
from stable_baselines3 import PPO

from test_env import DroneLandingEnv


# Folder for trained models
model_dir = "ppo_models"

os.makedirs(model_dir, exist_ok=True)


# Find the latest model version
existing_versions = []

for filename in os.listdir(model_dir):

    match = re.match(r"ppo_drone_landing_v(\d+)\.zip$", filename)

    if match:
        existing_versions.append(int(match.group(1)))


if existing_versions:
    next_version = max(existing_versions) + 1
else:
    next_version = 1


model_path = os.path.join(
    model_dir,
    f"ppo_drone_landing_v{next_version}"
)

print(f"Saving model as: {model_path}.zip")


# Create environment
env = DroneLandingEnv()


# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
)

# Train
model.learn(total_timesteps=2_500_000)


# Save trained model
model.save(model_path)


# Close environment
env.close()