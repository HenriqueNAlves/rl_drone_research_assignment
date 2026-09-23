import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import time


class DroneLandingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # ============================================================
        # ENVIRONMENT / PLATFORM PARAMETERS
        # ============================================================

        platform_angle = 30
        platform_width = 0.2
        platform_height = 0.1

        height_difference = (
            platform_width * np.tan(np.radians(platform_angle))
        )
        highside_height = platform_height + height_difference

        # ============================================================
        # SUCCESSFUL LANDING PARAMETERS
        # ============================================================

        self.landing_max_velocity = 0.2
        self.landing_max_angle = 20.0
        self.landing_required_time = 0.2

        # ============================================================
        # FAILURE / TERMINATION PARAMETERS
        # ============================================================

        self.ground_height = 0.035
        self.max_episode_time = 10.0

        # ============================================================
        # DRONE PHYSICAL PROPERTIES (identified from flight logs)
        # ============================================================

        self.max_thrust = 3.52        # N
        self.max_body_rate = 8.73     # rad/s (500 deg/s)

        # Rate-loop model fitted from blackbox logs
        self.rate_gain_K = 3.0        # 1/s   (tau ~ 330 ms)
        self.rate_limit = 26.18       # rad/s^2

        # ============================================================
        # TASK
        # ============================================================

        self.task_mode = "hover"

        self.hover_target_position = np.array([0.5, -0.5, 1.0])
        self.spawn_position = np.array([0.5, 0.5, 0.5])

        self.hover_success_distance = 0.20
        self.hover_success_time = 0.5

        # ============================================================
        # REWARD PARAMETERS
        # ============================================================

        self.reward_position_distance = 1.0
        self.reward_approach_gain = 1.0
        self.reward_velocity_penalty = 0.1
        self.reward_ground_contact = -1.0

        self.tilt_safe_angle = 30.0
        self.tilt_dangerimport gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import time


class DroneLandingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # ============================================================
        # ENVIRONMENT / PLATFORM PARAMETERS
        # ============================================================

        platform_angle = 30
        platform_width = 0.2
        platform_height = 0.1

        height_difference = (
            platform_width * np.tan(np.radians(platform_angle))
        )
        highside_height = platform_height + height_difference

        # ============================================================
        # SUCCESSFUL LANDING PARAMETERS
        # ============================================================

        self.landing_max_velocity = 0.2
        self.landing_max_angle = 20.0
        self.landing_required_time = 0.2

        # ============================================================
        # FAILURE / TERMINATION PARAMETERS
        # ============================================================

        self.ground_height = 0.035
        self.max_episode_time = 10.0

        # ============================================================
        # TERMINATION TOGGLES  (#6)
        # ============================================================

        self.terminate_on_ground_collision = True
        self.terminate_on_tower_collision = True

        self.terminate_on_tilt = True
        self.tilt_termination_angle = 90.0   # (#5) single threshold

        # ============================================================
        # DRONE PHYSICAL PROPERTIES (identified from flight logs)
        # ============================================================

        self.max_thrust = 3.52        # N
        self.max_body_rate = 8.73     # rad/s (500 deg/s)

        self.rate_gain_K = 3.0        # 1/s   (tau ~ 330 ms)
        self.rate_limit = 26.18       # rad/s^2

        # ============================================================
        # TASK
        # ============================================================

        self.task_mode = "hover"

        self.hover_target_position = np.array([0.5, -0.5, 1.0])
        self.spawn_position = np.array([0.5, 0.5, 0.5])

        self.hover_success_distance = 0.20    # (#1) widened
        self.hover_success_time = 0.5

        # ============================================================
        # REWARD PARAMETERS
        # ============================================================

        self.reward_position_distance = 1.0
        self.reward_approach_gain = 1.0
        self.reward_velocity_penalty = 0.1
        self.reward_ground_contact = -1.0

        # Exponential tilt penalty (#4). Always applied, no band.
        self.tilt_penalty_coefficient = 0.1
        self.tilt_penalty_scale = 20.0

        # Terminal rewards/penalties
        self.reward_hover_success = 100.0
        self.penalty_ground_collision = 10.0
        self.penalty_tilt_termination = 60.0
        self.penalty_tower_collision = 100.0
        self.reward_tower_hit = 100.0

        self.hover_stable_steps = 0

        # ============================================================
        # MUJOCO MODEL
        # ============================================================

        self.model = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
            <asset>
                <mesh
                    name="landing_platform"
                    vertex="
                        -0.1 -0.1 0
                         0.1 -0.1 0
                         0.1  0.1 0
                        -0.1  0.1 0

                        -0.1 -0.1 {platform_height}
                         0.1 -0.1 {highside_height}
                         0.1  0.1 {highside_height}
                        -0.1  0.1 {platform_height}
                    "
                    face="
                        0 1 2
                        0 2 3
                        4 6 5
                        4 7 6
                        0 4 5
                        0 5 1
                        1 5 6
                        2 6 7
                        2 7 3
                        3 4 0
                    "
                />
            </asset>

            <worldbody>

                <geom name="ground_geom" type="plane" size="5 5 0.1"/>

                <body name="tower1" pos="0 0 0.1">
                    <freejoint/>
                    <geom name="tower1_geom" type="box" size="0.1 0.1 0.1"
                          mass="1" rgba="0.65 0.55 0.1 1"/>
                </body>

                <body name="tower2" pos="0 0 0.3">
                    <freejoint/>
                    <geom name="tower2_geom" type="box" size="0.1 0.1 0.1"
                          mass="1" rgba="0.65 0.55 0.1 1"/>
                </body>

                <body name="tower3" pos="0 0 0.5">
                    <freejoint/>
                    <geom name="tower3_geom" type="box" size="0.1 0.1 0.1"
                          mass="1" rgba="0.65 0.15 0.12 1"/>
                </body>

                <body name="platform" pos="0 0 0.6">
                    <freejoint/>
                    <geom name="platform_geom" type="mesh"
                          mesh="landing_platform" mass="1"
                          rgba="0.65 0.15 0.12 1"/>
                </body>

                <body name="drone" pos="0.5 0.5 0.5">
                    <freejoint/>
                    <geom name="drone_geom" type="box" size="0.065 0.065 0.04"
                          mass="0.16" rgba="0.2 0.6 0.9 1"/>
                </body>

                <!-- (#2) Hover target zone — visual only, no collisions -->
                <body name="hover_marker" pos="0.5 -0.5 1.0">
                    <geom
                        type="sphere"
                        size="0.20"
                        rgba="0.1 0.9 0.2 0.25"
                        contype="0"
                        conaffinity="0"
                    />
                </body>

            </worldbody>
        </mujoco>
        """)

        self.data = mujoco.MjData(self.model)

        # ============================================================
        # BODY / GEOM IDS
        # ============================================================

        self.drone_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.platform_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "platform")
        self.tower_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tower1"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tower2"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tower3"),
        ]

        self.drone_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "drone_geom")
        self.platform_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "platform_geom")
        self.ground_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground_geom")
        self.tower_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tower1_geom"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tower2_geom"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tower3_geom"),
        ]

        self.tower_initial_positions = np.array([
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.5]
        ])

        # ============================================================
        # FREEJOINTS
        # ============================================================

        self.drone_joint_id = self.model.body_jntadr[self.drone_id]
        self.platform_joint_id = self.model.body_jntadr[self.platform_id]

        self.drone_dofadr = self.model.jnt_dofadr[self.drone_joint_id]

        # ============================================================
        # PLATFORM PLANE GEOMETRY (debug only)
        # ============================================================

        self.platform_plane_point_local = np.array([
            0.0, 0.0, platform_height + height_difference / 2.0
        ])
        platform_slope = height_difference / platform_width
        self.platform_slope = platform_slope
        self.platform_half_width = platform_width / 2.0
        self.platform_plane_normal_local = np.array([-platform_slope, 0.0, 1.0])
        self.platform_plane_normal_local /= np.linalg.norm(
            self.platform_plane_normal_local
        )

        # ============================================================
        # EPISODE STATE
        # ============================================================

        self.episode_steps = 0
        self.landing_stable_steps = 0

        # ============================================================
        # SPACES
        # ============================================================

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.viewer = None

    # ================================================================
    # GET OBSERVATION
    # ================================================================

    def _get_observation(self):
        drone_position = self.data.xpos[self.drone_id].copy()
        drone_orientation = self.data.xquat[self.drone_id].copy()

        drone_velocity = self.data.qvel[
            self.drone_dofadr:self.drone_dofadr + 6
        ].copy()
        drone_linear_velocity = drone_velocity[:3]
        drone_angular_velocity = drone_velocity[3:]

        platform_position = self.data.xpos[self.platform_id].copy()
        platform_orientation = self.data.xquat[self.platform_id].copy()

        platform_velocity = self.data.qvel[
            self.model.jnt_dofadr[self.platform_joint_id]:
            self.model.jnt_dofadr[self.platform_joint_id] + 6
        ].copy()
        platform_linear_velocity = platform_velocity[:3]

        relative_position = drone_position - platform_position
        relative_velocity = drone_linear_velocity - platform_linear_velocity

        observation = np.concatenate([
            drone_orientation,
            drone_linear_velocity,
            drone_angular_velocity,
            relative_position,
            relative_velocity,
            platform_orientation
        ])
        return observation.astype(np.float32)

    # ================================================================
    # TILT ANGLE (degrees from level)
    # ================================================================

    def _get_tilt_angle_deg(self):
        rotation_matrix = self.data.xmat[self.drone_id].reshape(3, 3)
        body_up_world = rotation_matrix[:, 2]
        world_up = np.array([0.0, 0.0, 1.0])
        alignment = float(np.clip(np.dot(body_up_world, world_up), -1.0, 1.0))
        return float(np.degrees(np.arccos(alignment)))

    # ================================================================
    # REWARD
    # ================================================================

    def _calculate_reward(self, terminated, termination_reason):

        drone_position = self.data.xpos[self.drone_id]
        drone_velocity = self.data.qvel[
            self.drone_dofadr:self.drone_dofadr + 3
        ]

        reward = 0.0

        if self.task_mode == "hover":
            to_target = self.hover_target_position - drone_position
            distance = float(np.linalg.norm(to_target))
            reward -= self.reward_position_distance * (distance ** 2)

            if distance > 1e-6:
                direction = to_target / distance
                v_toward = float(np.dot(drone_velocity, direction))
            else:
                v_toward = 0.0
            reward += self.reward_approach_gain * v_toward

        elif self.task_mode == "ram":
            tower_target = np.array([0.0, 0.0, 0.5])
            to_target = tower_target - drone_position
            distance = float(np.linalg.norm(to_target))
            reward -= self.reward_position_distance * (distance ** 2)

            if distance > 1e-6:
                direction = to_target / distance
                v_toward = float(np.dot(drone_velocity, direction))
            else:
                v_toward = 0.0
            reward += self.reward_approach_gain * v_toward

        linear_speed = float(np.linalg.norm(drone_velocity))
        reward -= self.reward_velocity_penalty * linear_speed

        # (#4) Exponential tilt penalty — smooth, applies at all angles
        tilt_deg = self._get_tilt_angle_deg()
        reward -= self.tilt_penalty_coefficient * (
            np.exp(tilt_deg / self.tilt_penalty_scale) - 1.0
        )

        # Terminal rewards / penalties (gated by the toggles)
        if terminated:
            if termination_reason == "hover_success":
                reward += self.reward_hover_success
            elif termination_reason == "ground_collision":
                if self.terminate_on_ground_collision:
                    reward -= self.penalty_ground_collision
            elif termination_reason == "tilt_termination":
                if self.terminate_on_tilt:
                    reward -= self.penalty_tilt_termination
            elif termination_reason == "tower_collision":
                if self.terminate_on_tower_collision:
                    if self.task_mode == "hover":
                        reward -= self.penalty_tower_collision
                    else:
                        reward += self.reward_tower_hit

        return reward

    # ================================================================
    # CHECK TERMINATION
    # ================================================================

    def _check_termination(self):

        # --- Tower collision (#6 toggle) ---
        if self.terminate_on_tower_collision:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                g1, g2 = contact.geom1, contact.geom2
                for tower_geom in self.tower_geom_ids:
                    if ((g1 == self.drone_geom_id and g2 == tower_geom) or
                        (g2 == self.drone_geom_id and g1 == tower_geom)):
                        return True, "tower_collision"

        # --- Ground contact (#6 toggle) ---
        if self.terminate_on_ground_collision:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                g1, g2 = contact.geom1, contact.geom2
                if ((g1 == self.drone_geom_id and g2 == self.ground_geom_id) or
                    (g2 == self.drone_geom_id and g1 == self.ground_geom_id)):
                    return True, "ground_collision"

        # --- Tilt termination (#5) ---
        if self.terminate_on_tilt:
            tilt_deg = self._get_tilt_angle_deg()
            if tilt_deg >= self.tilt_termination_angle:
                return True, "tilt_termination"

        # --- Hover success ---
        if self.task_mode == "hover":
            drone_position = self.data.xpos[self.drone_id]
            distance = float(np.linalg.norm(
                drone_position - self.hover_target_position))

            if distance < self.hover_success_distance:
                self.hover_stable_steps += 1
            else:
                self.hover_stable_steps = 0

            required_steps = max(
                1, int(self.hover_success_time / self.model.opt.timestep))
            if self.hover_stable_steps >= required_steps:
                return True, "hover_success"

        # --- Time limit ---
        elapsed_time = self.episode_steps * self.model.opt.timestep
        if elapsed_time >= self.max_episode_time:
            return False, "time_limit"

        return False, None

    # ================================================================
    # RESET
    # ================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.episode_steps = 0
        self.landing_stable_steps = 0
        self.hover_stable_steps = 0

        observation = self._get_observation()
        return observation, {}

    # ================================================================
    # STEP
    # ================================================================

    def step(self, action):

        thrust = (action[0] + 1.0) / 2.0 * self.max_thrust

        desired_body_rates = np.array([
            action[1] * self.max_body_rate,
            action[2] * self.max_body_rate,
            action[3] * self.max_body_rate
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

        observation = self._get_observation()

        terminated, termination_reason = self._check_termination()
        truncated = (termination_reason == "time_limit")

        reward = self._calculate_reward(terminated, termination_reason)

        drone_position = self.data.xpos[self.drone_id]
        distance_to_target = float(np.linalg.norm(
            drone_position - self.hover_target_position))
        tilt_deg = self._get_tilt_angle_deg()

        info = {
            "thrust": thrust,
            "distance_to_target": distance_to_target,
            "tilt_deg": tilt_deg,
            "termination_reason": termination_reason,
        }

        return observation, reward, terminated, truncated, info

    # ================================================================
    # RENDER / CLOSE
    # ================================================================

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
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

    thrust_action = -1.0
    wx_action = 0.0
    wy_action = 0.0
    wz_action = 0.0

    def key_callback(keycode):
        if keycode in (ord("P"), ord("p")):
            thrust_action_holder[0] = (
                1.0 if thrust_action_holder[0] == -1.0 else -1.0)
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
            thrust_action_holder[0] = -1.0
            wx_action_holder[0] = 0.0
            wy_action_holder[0] = 0.0
            wz_action_holder[0] = 0.0

    thrust_action_holder = [thrust_action]
    wx_action_holder = [wx_action]
    wy_action_holder = [wy_action]
    wz_action_holder = [wz_action]

    print("MANUAL DRONE CONTROL — UP/DOWN/LEFT/RIGHT, P=thrust, R=reset")

    with mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=key_callback) as viewer:

        env.viewer = viewer
        viewer.cam.lookat[:] = env.hover_target_position
        viewer.cam.distance = 2.0

        while viewer.is_running():
            action = np.array([
                thrust_action_holder[0],
                wx_action_holder[0],
                wy_action_holder[0],
                wz_action_holder[0]
            ], dtype=np.float32)

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("EPISODE ENDED:", info["termination_reason"],
                      "Reward:", reward,
                      "Tilt:", info["tilt_deg"])
                observation, info = env.reset()
                viewer.sync()
                continue

            viewer.sync()
            time.sleep(0.002)

    env.close()_angle = 60.0
        self.tilt_fatal_angle = 90.0
        self.tilt_penalty_max = 15.0

        self.reward_hover_success = 100.0
        self.penalty_ground_collision = 10.0
        self.penalty_tilt_danger = 30.0
        self.penalty_tilt_fatal = 60.0
        self.penalty_tower_collision = 100.0
        self.reward_tower_hit = 100.0

        self.hover_stable_steps = 0

        # ============================================================
        # MUJOCO MODEL
        # ============================================================

        self.model = mujoco.MjModel.from_xml_string(f"""
        <mujoco>
            <asset>
                <mesh
                    name="landing_platform"
                    vertex="
                        -0.1 -0.1 0
                         0.1 -0.1 0
                         0.1  0.1 0
                        -0.1  0.1 0

                        -0.1 -0.1 {platform_height}
                         0.1 -0.1 {highside_height}
                         0.1  0.1 {highside_height}
                        -0.1  0.1 {platform_height}
                    "
                    face="
                        0 1 2
                        0 2 3
                        4 6 5
                        4 7 6
                        0 4 5
                        0 5 1
                        1 5 6
                        2 6 7
                        2 7 3
                        3 4 0
                    "
                />
            </asset>

            <worldbody>

                <geom name="ground_geom" type="plane" size="5 5 0.1"/>

                <body name="tower1" pos="0 0 0.1">
                    <freejoint/>
                    <geom name="tower1_geom" type="box" size="0.1 0.1 0.1"
                          mass="1" rgba="0.65 0.55 0.1 1"/>
                </body>

                <body name="tower2" pos="0 0 0.3">
                    <freejoint/>
                    <geom name="tower2_geom" type="box" size="0.1 0.1 0.1"
                          mass="1" rgba="0.65 0.55 0.1 1"/>
                </body>

                <body name="tower3" pos="0 0 0.5">
                    <freejoint/>
                    <geom name="tower3_geom" type="box" size="0.1 0.1 0.1"
                          mass="1" rgba="0.65 0.15 0.12 1"/>
                </body>

                <body name="platform" pos="0 0 0.6">
                    <freejoint/>
                    <geom name="platform_geom" type="mesh"
                          mesh="landing_platform" mass="1"
                          rgba="0.65 0.15 0.12 1"/>
                </body>

                <body name="drone" pos="0.5 0.5 0.5">
                    <freejoint/>
                    <geom name="drone_geom" type="box" size="0.065 0.065 0.04"
                          mass="0.16" rgba="0.2 0.6 0.9 1"/>
                </body>

            </worldbody>
        </mujoco>
        """)

        self.data = mujoco.MjData(self.model)

        # ============================================================
        # BODY / GEOM IDS
        # ============================================================

        self.drone_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.platform_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "platform")
        self.tower_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tower1"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tower2"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tower3"),
        ]

        self.drone_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "drone_geom")
        self.platform_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "platform_geom")
        self.ground_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground_geom")
        self.tower_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tower1_geom"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tower2_geom"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "tower3_geom"),
        ]

        self.tower_initial_positions = np.array([
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.5]
        ])

        # ============================================================
        # FREEJOINTS
        # ============================================================

        self.drone_joint_id = self.model.body_jntadr[self.drone_id]
        self.platform_joint_id = self.model.body_jntadr[self.platform_id]

        self.drone_dofadr = self.model.jnt_dofadr[self.drone_joint_id]

        # ============================================================
        # PLATFORM PLANE GEOMETRY (debug only)
        # ============================================================

        self.platform_plane_point_local = np.array([
            0.0, 0.0, platform_height + height_difference / 2.0
        ])
        platform_slope = height_difference / platform_width
        self.platform_slope = platform_slope
        self.platform_half_width = platform_width / 2.0
        self.platform_plane_normal_local = np.array([-platform_slope, 0.0, 1.0])
        self.platform_plane_normal_local /= np.linalg.norm(
            self.platform_plane_normal_local
        )

        # ============================================================
        # EPISODE STATE
        # ============================================================

        self.episode_steps = 0
        self.landing_stable_steps = 0

        # ============================================================
        # SPACES
        # ============================================================

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.viewer = None

    # ================================================================
    # GET OBSERVATION
    # ================================================================

    def _get_observation(self):
        drone_position = self.data.xpos[self.drone_id].copy()
        drone_orientation = self.data.xquat[self.drone_id].copy()

        drone_velocity = self.data.qvel[
            self.drone_dofadr:self.drone_dofadr + 6
        ].copy()
        drone_linear_velocity = drone_velocity[:3]
        drone_angular_velocity = drone_velocity[3:]

        platform_position = self.data.xpos[self.platform_id].copy()
        platform_orientation = self.data.xquat[self.platform_id].copy()

        platform_velocity = self.data.qvel[
            self.model.jnt_dofadr[self.platform_joint_id]:
            self.model.jnt_dofadr[self.platform_joint_id] + 6
        ].copy()
        platform_linear_velocity = platform_velocity[:3]

        relative_position = drone_position - platform_position
        relative_velocity = drone_linear_velocity - platform_linear_velocity

        observation = np.concatenate([
            drone_orientation,
            drone_linear_velocity,
            drone_angular_velocity,
            relative_position,
            relative_velocity,
            platform_orientation
        ])
        return observation.astype(np.float32)

    # ================================================================
    # TILT ANGLE (degrees from level)
    # ================================================================

    def _get_tilt_angle_deg(self):
        rotation_matrix = self.data.xmat[self.drone_id].reshape(3, 3)
        body_up_world = rotation_matrix[:, 2]
        world_up = np.array([0.0, 0.0, 1.0])
        alignment = float(np.clip(np.dot(body_up_world, world_up), -1.0, 1.0))
        return float(np.degrees(np.arccos(alignment)))

    # ================================================================
    # REWARD
    #
    # Change: distance term is now SQUARED instead of linear.
    # Squared distance gives a much steeper gradient near the target:
    #   1.0 m -> -1.00
    #   0.5 m -> -0.25
    #   0.2 m -> -0.04
    #   0.1 m -> -0.01
    # So closing the last bit of distance is heavily rewarded, and
    # overshooting (getting far again) is heavily punished.
    # ================================================================

    def _calculate_reward(self, terminated, termination_reason):

        drone_position = self.data.xpos[self.drone_id]
        drone_velocity = self.data.qvel[
            self.drone_dofadr:self.drone_dofadr + 3
        ]

        reward = 0.0

        if self.task_mode == "hover":
            to_target = self.hover_target_position - drone_position
            distance = float(np.linalg.norm(to_target))
            reward -= self.reward_position_distance * (distance ** 2)

            if distance > 1e-6:
                direction = to_target / distance
                v_toward = float(np.dot(drone_velocity, direction))
            else:
                v_toward = 0.0
            reward += self.reward_approach_gain * v_toward

        elif self.task_mode == "ram":
            tower_target = np.array([0.0, 0.0, 0.5])
            to_target = tower_target - drone_position
            distance = float(np.linalg.norm(to_target))
            reward -= self.reward_position_distance * (distance ** 2)

            if distance > 1e-6:
                direction = to_target / distance
                v_toward = float(np.dot(drone_velocity, direction))
            else:
                v_toward = 0.0
            reward += self.reward_approach_gain * v_toward

        linear_speed = float(np.linalg.norm(drone_velocity))
        reward -= self.reward_velocity_penalty * linear_speed

        tilt_deg = self._get_tilt_angle_deg()
        if self.tilt_safe_angle < tilt_deg < self.tilt_danger_angle:
            frac = (tilt_deg - self.tilt_safe_angle) / (
                self.tilt_danger_angle - self.tilt_safe_angle)
            reward -= self.tilt_penalty_max * frac

        if terminated:
            if termination_reason == "hover_success":
                reward += self.reward_hover_success
            elif termination_reason == "ground_collision":
                reward -= self.penalty_ground_collision
            elif termination_reason == "tilt_danger":
                reward -= self.penalty_tilt_danger
            elif termination_reason == "tilt_fatal":
                reward -= self.penalty_tilt_fatal
            elif termination_reason == "tower_collision":
                if self.task_mode == "hover":
                    reward -= self.penalty_tower_collision
                else:
                    reward += self.reward_tower_hit

        return reward

    # ================================================================
    # CHECK TERMINATION
    # ================================================================

    def _check_termination(self):

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            for tower_geom in self.tower_geom_ids:
                if ((g1 == self.drone_geom_id and g2 == tower_geom) or
                    (g2 == self.drone_geom_id and g1 == tower_geom)):
                    return True, "tower_collision"

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if ((g1 == self.drone_geom_id and g2 == self.ground_geom_id) or
                (g2 == self.drone_geom_id and g1 == self.ground_geom_id)):
                return True, "ground_collision"

        tilt_deg = self._get_tilt_angle_deg()
        if tilt_deg >= self.tilt_fatal_angle:
            return True, "tilt_fatal"
        if tilt_deg >= self.tilt_danger_angle:
            return True, "tilt_danger"

        if self.task_mode == "hover":
            drone_position = self.data.xpos[self.drone_id]
            distance = float(np.linalg.norm(
                drone_position - self.hover_target_position))

            if distance < self.hover_success_distance:
                self.hover_stable_steps += 1
            else:
                self.hover_stable_steps = 0

            required_steps = max(
                1, int(self.hover_success_time / self.model.opt.timestep))
            if self.hover_stable_steps >= required_steps:
                return True, "hover_success"

        elapsed_time = self.episode_steps * self.model.opt.timestep
        if elapsed_time >= self.max_episode_time:
            return False, "time_limit"

        return False, None

    # ================================================================
    # RESET
    # ================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.episode_steps = 0
        self.landing_stable_steps = 0
        self.hover_stable_steps = 0

        observation = self._get_observation()
        return observation, {}

    # ================================================================
    # STEP
    # ================================================================

    def step(self, action):

        thrust = (action[0] + 1.0) / 2.0 * self.max_thrust

        desired_body_rates = np.array([
            action[1] * self.max_body_rate,
            action[2] * self.max_body_rate,
            action[3] * self.max_body_rate
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

        observation = self._get_observation()

        terminated, termination_reason = self._check_termination()
        truncated = (termination_reason == "time_limit")

        reward = self._calculate_reward(terminated, termination_reason)

        drone_position = self.data.xpos[self.drone_id]
        distance_to_target = float(np.linalg.norm(
            drone_position - self.hover_target_position))
        tilt_deg = self._get_tilt_angle_deg()

        info = {
            "thrust": thrust,
            "distance_to_target": distance_to_target,
            "tilt_deg": tilt_deg,
            "termination_reason": termination_reason,
        }

        return observation, reward, terminated, truncated, info

    # ================================================================
    # RENDER / CLOSE
    # ================================================================

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
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

    thrust_action = -1.0
    wx_action = 0.0
    wy_action = 0.0
    wz_action = 0.0

    def key_callback(keycode):
        if keycode in (ord("P"), ord("p")):
            thrust_action_holder[0] = (
                1.0 if thrust_action_holder[0] == -1.0 else -1.0)
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
            thrust_action_holder[0] = -1.0
            wx_action_holder[0] = 0.0
            wy_action_holder[0] = 0.0
            wz_action_holder[0] = 0.0

    thrust_action_holder = [thrust_action]
    wx_action_holder = [wx_action]
    wy_action_holder = [wy_action]
    wz_action_holder = [wz_action]

    print("MANUAL DRONE CONTROL — UP/DOWN/LEFT/RIGHT, P=thrust, R=reset")

    with mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=key_callback) as viewer:

        env.viewer = viewer
        viewer.cam.lookat[:] = env.hover_target_position
        viewer.cam.distance = 2.0

        while viewer.is_running():
            action = np.array([
                thrust_action_holder[0],
                wx_action_holder[0],
                wy_action_holder[0],
                wz_action_holder[0]
            ], dtype=np.float32)

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("EPISODE ENDED:", info["termination_reason"],
                      "Reward:", reward,
                      "Tilt:", info["tilt_deg"])
                observation, info = env.reset()
                viewer.sync()
                continue

            viewer.sync()
            time.sleep(0.002)

    env.close()