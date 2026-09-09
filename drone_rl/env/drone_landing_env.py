import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DroneLandingEnv(gym.Env):
    """
    Gymnasium environment for learning to land a quadrotor
    on an inclined platform supported by a potentially unstable tower.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # ---------------------------------------------------------
        # ACTION SPACE
        # ---------------------------------------------------------
        # [collective thrust, roll rate, pitch rate, yaw rate]
        #
        # All actions are normalized to [-1, 1].
        # We will later map these values to the physical
        # CTBR limits of the drone.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # OBSERVATION SPACE
        # ---------------------------------------------------------
        # This is a temporary placeholder.
        # We will define the exact observation vector once
        # we establish the simulator's state representation.
        #
        # For now:
        #   3 -> relative position
        #   3 -> relative velocity
        #   9 -> relative orientation (rotation matrix)
        #   3 -> drone angular velocity
        #   4 -> tower state (placeholder)
        #
        # Total = 22 values.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        """
        Start a new episode.
        """
        super().reset(seed=seed)

        # TODO:
        # Reset the drone
        # Reset the tower
        # Reset the platform
        # Reset simulation time
        # Generate the initial observation

        observation = np.zeros(22, dtype=np.float32)

        info = {}

        return observation, info

    def step(self, action):
        """
        Apply one CTBR action and advance the simulation by one timestep.
        """

        # TODO:
        # 1. Convert normalized action to physical CTBR command
        # 2. Send command to the simulator
        # 3. Advance physics
        # 4. Read new drone/platform/tower state
        # 5. Construct observation
        # 6. Calculate reward
        # 7. Determine whether landing succeeded
        # 8. Determine whether tower collapsed
        # 9. Determine whether episode was truncated

        observation = np.zeros(22, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the current simulation state.
        """
        pass

    def close(self):
        """
        Clean up the simulation.
        """
        pass