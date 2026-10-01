import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import time


class DroneLandingEnv(gym.Env):
    """
    Minimal hover task with attitude-stabilized control.

    Actions:
        action[0]  -> thrust residual around hover thrust
                      (0 = hover, +1 = full up, -1 = full down)
        action[1]  -> desired roll angle   (-1 .. +1)
        action[2]  -> desired pitch angle  (-1 .. +1)
        action[3]  -> desired yaw rate     (-1 .. +1)

    Reward per step:
        - distance_penalty * curr_distance
        + velocity_gain    * radial_velocity
        - angular_penalty  * ||body_rates||

    Terminal:
        success          -> +success_bonus
        ground_collision -> -terminal_penalty
        out_of_bounds    -> -terminal_penalty
        time_limit       -> no bonus, no penalty
    """

    def __init__(self):
        super().__init__()

        # ============================================================
        # TASK POINTS (same point: hover)
        # ============================================================

        self.spawn_position = np.array([0.0, 0.0, 1.0])
        self.target_position = np.array([0.0, 0.0, 1.0])

        # ============================================================
        # EPISODE
        # ============================================================

        self.max_episode_time = 20.0

        # ============================================================
        # SUCCESS
        # ============================================================

        self.success_distance = 0.10
        self.success_required_time = 5.0

        # ============================================================
        # ARENA BOUNDS
        # ============================================================

        self.arena_x_max = 5.0
        self.arena_y_max = 5.0
        self.arena_z_max = 5.0

        # ============================================================
        # DRONE PHYSICAL PROPERTIES
        # ============================================================

        self.drone_mass = 0.16
        self.gravity = 9.81
        self.hover_thrust = self.drone_mass * self.gravity   # 1.5696 N

        self.max_thrust = 3.52
        self.max_body_rate = 8.73
        self.rate_gain_K = 3.0
        self.rate_limit = 26.18

        # thrust range around hover: action[0] = +1 -> hover + scale,
        # action[0] = -1 -> hover - scale (clamped at 0)
        self.thrust_scale = 1.5

        # ============================================================
        # ATTITUDE CONTROLLER
        # ============================================================

        self.max_roll_angle = np.radians(30.0)
        self.max_pitch_angle = np.radians(30.0)
        self.max_yaw_rate = 2.0

        self.attitude_Kp = 4.0

        # ============================================================
        # REWARD WEIGHTS
        # ============================================================

        self.reward_distance_penalty = 1.0
        self.reward_velocity_gain = 0.5
        self.reward_angular_penalty = 0.01

        # ============================================================
        # TERMINAL REWARDS / PENALTIES
        # ============================================================

        self.reward_success_bonus = 100.0
        self.penalty_terminal = 10.0

        # ============================================================
        # MUJOCO MODEL
        # ============================================================

        self.model = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
            <worldbody>

                <geom name="ground_geom" type="plane" size="5 5 0.1"
                      rgba="0.3 0.3 0.3 1"/>

                <body name="drone" pos="0 0 1.0">
                    <freejoint/>
                    <geom name="drone_geom" type="box"
                          size="0.065 0.065 0.04"
                          mass="{self.drone_mass}"
                          rgba="0.2 0.6 0.9 1"/>
                </body>

                <body name="goal_marker" pos="{self.target_position[0]}
                                              {self.target_position[1]}
                                              {self.target_position[2]}">
                    <geom type="sphere" size="0.10"
                          rgba="0.1 0.9 0.2 0.5"
                          contype="0" conaffinity="0"/>
                </body>

            </worldbody>
        </mujoco>
        """)

        self.data = mujoco.MjData(self.model)

        # ============================================================
        # IDS
        # ============================================================

        self.drone_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.drone_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "drone_geom")
        self.ground_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground_geom")

        self.drone_joint_id = self.model.body_jntadr[self.drone_id]
        self.drone_dofadr = self.model.jnt_dofadr[self.drone_joint_id]

        # ============================================================
        # EPISODE STATE
        # ============================================================

        self.episode_steps = 0
        self.success_steps = 0

        # ============================================================
        # SPACES
        # ============================================================

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.viewer = None

    # ================================================================
    # OBSERVATION
    # ================================================================

    def _get_observation(self):
        drone_position = self.data.xpos[self.drone_id].copy()
        drone_quat = self.data.xquat[self.drone_id].copy()

        drone_velocity = self.data.qvel[
            self.drone_dofadr:self.drone_dofadr + 3
        ].copy()

        to_target = self.target_position - drone_position

        observation = np.concatenate([
            drone_position,
            drone_velocity,
            drone_quat,
            self.target_position,
            to_target,
        ])
        return observation.astype(np.float32)

    # ================================================================
    # ATTITUDE
    # ================================================================

    def _get_roll_pitch(self):
        R = self.data.xmat[self.drone_id].reshape(3, 3)
        bz = R[:, 2]
        pitch = np.arctan2(bz[0], bz[2])
        roll = np.arctan2(bz[1], bz[2])
        return roll, pitch

    # ================================================================
    # REWARD
    # ================================================================

    def _calculate_reward(self, curr_distance, drone_velocity):

        to_target = self.target_position - self.data.xpos[self.drone_id]
        dist = np.linalg.norm(to_target)
        if dist > 1e-6:
            direction = to_target / dist
        else:
            direction = np.zeros(3)

        radial_velocity = float(np.dot(drone_velocity, direction))

        body_rates = self.data.qvel[
            self.drone_dofadr + 3:self.drone_dofadr + 6
        ]
        angular_speed = float(np.linalg.norm(body_rates))

        reward = (
            - self.reward_distance_penalty * curr_distance
            + self.reward_velocity_gain * radial_velocity
            - self.reward_angular_penalty * angular_speed
        )
        return float(reward)

    # ================================================================
    # CHECK TERMINATION
    # ================================================================

    def _check_termination(self):

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if ((g1 == self.drone_geom_id and g2 == self.ground_geom_id) or
                    (g2 == self.drone_geom_id and g1 == self.ground_geom_id)):
                return True, "ground_collision"

        drone_position = self.data.xpos[self.drone_id]
        distance = float(np.linalg.norm(drone_position - self.target_position))

        if distance < self.success_distance:
            self.success_steps += 1
        else:
            self.success_steps = 0

        required = max(1, int(self.success_required_time
                              / self.model.opt.timestep))
        if self.success_steps >= required:
            return True, "success"

        if (abs(drone_position[0]) > self.arena_x_max or
            abs(drone_position[1]) > self.arena_y_max or
            drone_position[2] > self.arena_z_max or
            drone_position[2] < 0.0):
            return True, "out_of_bounds"

        elapsed = self.episode_steps * self.model.opt.timestep
        if elapsed >= self.max_episode_time:
            return False, "time_limit"

        return False, None

    # ================================================================
    # RESET
    # ================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[
            self.model.jnt_qposadr[self.drone_joint_id]:
            self.model.jnt_qposadr[self.drone_joint_id] + 3
        ] = self.spawn_position

        self.data.qpos[
            self.model.jnt_qposadr[self.drone_joint_id] + 3:
            self.model.jnt_qposadr[self.drone_joint_id] + 7
        ] = np.array([1.0, 0.0, 0.0, 0.0])

        mujoco.mj_forward(self.model, self.data)

        self.episode_steps = 0
        self.success_steps = 0

        observation = self._get_observation()
        return observation, {}

    # ================================================================
    # STEP
    # ================================================================

    def step(self, action):

        # thrust is a residual around hover thrust.
        # action[0] = 0  -> hover thrust
        # action[0] = +1 -> hover + thrust_scale
        # action[0] = -1 -> hover - thrust_scale (clamped at 0)
        thrust = self.hover_thrust + action[0] * self.thrust_scale
        thrust = float(np.clip(thrust, 0.0, self.max_thrust))

        desired_roll = action[1] * self.max_roll_angle
        desired_pitch = action[2] * self.max_pitch_angle
        desired_yaw_rate = action[3] * self.max_yaw_rate

        current_roll, current_pitch = self._get_roll_pitch()

        roll_error = desired_roll - current_roll
        pitch_error = desired_pitch - current_pitch

        desired_roll_rate = self.attitude_Kp * roll_error
        desired_pitch_rate = self.attitude_Kp * pitch_error

        desired_roll_rate = float(np.clip(
            desired_roll_rate, -self.max_body_rate, self.max_body_rate))
        desired_pitch_rate = float(np.clip(
            desired_pitch_rate, -self.max_body_rate, self.max_body_rate))

        desired_body_rates = np.array([
            desired_roll_rate,
            desired_pitch_rate,
            desired_yaw_rate,
        ])

        current_body_rates = self.data.qvel[
            self.drone_dofadr + 3:self.drone_dofadr + 6
        ].copy()

        rate_error = desired_body_rates - current_body_rates
        rate_accel = np.clip(
            self.rate_gain_K * rate_error,
            -self.rate_limit,
            self.rate_limit
        )

        dt = self.model.opt.timestep
        new_body_rates = current_body_rates + rate_accel * dt

        self.data.qvel[
            self.drone_dofadr + 3:self.drone_dofadr + 6
        ] = new_body_rates

        rotation_matrix = self.data.xmat[self.drone_id].reshape(3, 3)
        body_z_axis = rotation_matrix[:, 2]
        thrust_force = thrust * body_z_axis

        self.data.xfrc_applied[self.drone_id, 0:3] = thrust_force
        self.data.xfrc_applied[self.drone_id, 3:6] = 0.0

        mujoco.mj_step(self.model, self.data)

        self.episode_steps += 1
        self.data.xfrc_applied[self.drone_id] = 0.0

        curr_pos = self.data.xpos[self.drone_id].copy()
        curr_distance = float(np.linalg.norm(
            curr_pos - self.target_position))

        drone_velocity = self.data.qvel[
            self.drone_dofadr:self.drone_dofadr + 3
        ].copy()

        reward = self._calculate_reward(curr_distance, drone_velocity)

        terminated, reason = self._check_termination()
        truncated = (reason == "time_limit")

        if terminated:
            if reason == "success":
                reward += self.reward_success_bonus
            elif reason == "ground_collision":
                reward -= self.penalty_terminal
            elif reason == "out_of_bounds":
                reward -= self.penalty_terminal

        observation = self._get_observation()

        info = {
            "distance_to_target": curr_distance,
            "termination_reason": reason,
            "thrust": thrust,
        }

        return observation, reward, terminated, truncated, info

    # ================================================================
    # RENDER / CLOSE
    # ================================================================

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# ====================================================================
# MANUAL TEST
# ====================================================================

if __name__ == "__main__":

    env = DroneLandingEnv()
    env.reset()

    thrust_action_holder = [0.0]
    wx_action_holder = [0.0]
    wy_action_holder = [0.0]
    wz_action_holder = [0.0]

    def key_callback(keycode):
        if keycode in (ord("P"), ord("p")):
            thrust_action_holder[0] = (
                0.5 if thrust_action_holder[0] == 0.0 else 0.0)
        elif keycode == 265:
            wy_action_holder[0] = (
                0.5 if wy_action_holder[0] == 0.0 else 0.0)
        elif keycode == 264:
            wy_action_holder[0] = (
                -0.5 if wy_action_holder[0] == 0.0 else 0.0)
        elif keycode == 263:
            wx_action_holder[0] = (
                -0.5 if wx_action_holder[0] == 0.0 else 0.0)
        elif keycode == 262:
            wx_action_holder[0] = (
                0.5 if wx_action_holder[0] == 0.0 else 0.0)
        elif keycode in (ord("R"), ord("r")):
            env.reset()
            thrust_action_holder[0] = 0.0
            wx_action_holder[0] = 0.0
            wy_action_holder[0] = 0.0
            wz_action_holder[0] = 0.0

    print("MANUAL DRONE CONTROL — arrows = tilt, P = thrust bump, R = reset")
    print("Thrust action 0 = hover. Default drone should hover on its own.")
    print(f"Target: {env.target_position}")

    with mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=key_callback) as viewer:

        env.viewer = viewer
        viewer.cam.lookat[:] = env.spawn_position
        viewer.cam.distance = 3.0

        while viewer.is_running():
            action = np.array([
                thrust_action_holder[0],
                wx_action_holder[0],
                wy_action_holder[0],
                wz_action_holder[0]
            ], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("EPISODE ENDED:", info["termination_reason"],
                      "Reward:", reward,
                      "Distance:", info["distance_to_target"])
                obs, _ = env.reset()
                viewer.sync()
                continue

            viewer.sync()
            time.sleep(0.002)

    env.close()