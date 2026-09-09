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
        #
        # These define the physical geometry of the landing platform.
        # ============================================================

        platform_angle = 30       # deg
        platform_width = 0.2      # m
        platform_height = 0.1     # m

        height_difference = (
            platform_width
            * np.tan(np.radians(platform_angle))
        )

        highside_height = platform_height + height_difference

        # ============================================================
        # SUCCESSFUL LANDING PARAMETERS
        #
        # These determine when the drone is considered to have
        # successfully landed on the platform.
        # ============================================================

        self.landing_max_velocity = 0.2       # m/s  # PLACEHOLDER!!!!!!!
        self.landing_max_angle = 20.0         # deg  # PLACEHOLDER!!!!!!!
        self.landing_required_time = 0.2      # s    # PLACEHOLDER!!!!!!!

        # ============================================================
        # FAILURE / TERMINATION PARAMETERS
        #
        # These determine when an episode ends because the drone
        # has failed the task.
        # ============================================================

        self.ground_height = 0.035       # m  # PLACEHOLDER!!!!!!!
        self.max_horizontal_distance = 2.0  # m  # PLACEHOLDER!!!!!!!
        self.tower_max_horizontal_displacement = 0.3  # m  # PLACEHOLDER!!!!!!!
        self.tower_min_height = 0.05      # m  # PLACEHOLDER!!!!!!!
        self.max_episode_time = 10.0      # s  # PLACEHOLDER!!!!!!!

        # ============================================================
        # DRONE PHYSICAL PROPERTIES
        #
        # These describe the physical capabilities of the drone.
        #
        # These are temporary values for the simplified box drone.
        # They will eventually be replaced by values appropriate for
        # the actual quadrotor.
        # ============================================================

        self.max_thrust = 10.0       # N  # PLACEHOLDER!!!!!!!
        self.max_body_rate = 5.0     # rad/s  # PLACEHOLDER!!!!!!!

        # ============================================================
        # BODY-RATE CONTROLLER PARAMETERS
        #
        # Temporary low-level controller used to convert desired
        # body rates into torques.
        # ============================================================

        self.rate_controller_gain = 0.5       # PLACEHOLDER!!!!!!!
        self.max_torque = 1.0                 # N*m  # PLACEHOLDER!!!!!!!

        # ============================================================
        # REWARD PARAMETERS
        #
        # These determine how the agent is rewarded and penalized.
        # All values are temporary and will need tuning.
        # ============================================================

        # Successful landing
        self.reward_success = 100.0       # PLACEHOLDER!!!!!!!

        # General failure / crash
        self.penalty_failure = 100.0      # PLACEHOLDER!!!!!!!

        # Ground collision
        self.penalty_ground_collision = 500.0  # PLACEHOLDER!!!!!!!

        # Progress toward platform
        self.reward_proximity = 10.0      # PLACEHOLDER!!!!!!!

        # Relative velocity penalty
        self.penalty_velocity = 1.0       # PLACEHOLDER!!!!!!!

        # Orientation penalty
        self.penalty_orientation = 1.0    # PLACEHOLDER!!!!!!!

        # Tower disturbance penalty
        self.penalty_tower = 2.0          # PLACEHOLDER!!!!!!!
        self.tower_exponential_scale = 5.0  # PLACEHOLDER!!!!!!!

        # High-thrust penalty
        self.high_thrust_threshold = 0.8  # normalized thrust  # PLACEHOLDER!!!!!!!
        self.penalty_high_thrust = 0.1    # PLACEHOLDER!!!!!!!
        self.thrust_exponential_scale = 5.0  # PLACEHOLDER!!!!!!!

        # Time penalty
        self.penalty_time = 0.01          # PLACEHOLDER!!!!!!!

        # ============================================================
        # TILTED PLATFORM PLANE REWARD
        #
        # The top surface of the platform is treated as a tilted plane.
        #
        # The drone is rewarded for being close to this plane.
        #
        # Being above the plane is penalized increasingly as the drone
        # gets farther away.
        #
        # Being below the plane is penalized much more strongly,
        # because we want to prevent the drone from descending toward
        # the ground instead of landing on the platform.
        # ============================================================

        self.reward_plane_proximity = 5.0  # PLACEHOLDER!!!!!!!

        self.plane_reward_distance = 0.1  # m  # PLACEHOLDER!!!!!!!

        self.penalty_above_plane = 1.0  # PLACEHOLDER!!!!!!!

        self.penalty_below_plane = 5.0  # PLACEHOLDER!!!!!!!

        self.plane_exponential_scale = 5.0  # PLACEHOLDER!!!!!!!

        # Number of consecutive steps spent at high thrust
        self.high_thrust_steps = 0

        # Previous distance from drone to platform
        self.previous_distance = None

        # Previous absolute distance from drone to platform plane
        self.previous_plane_distance = None

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
                        3 7 4
                        3 4 0
                    "
                />

            </asset>

            <worldbody>

                <!-- Ground -->
                <geom
                    name="ground_geom"
                    type="plane"
                    size="5 5 0.1"
                />

                <!-- Box 1 -->
                <body name="tower1" pos="0 0 0.1">
                    <freejoint/>

                    <geom
                        type="box"
                        size="0.1 0.1 0.1"
                        mass="1"
                        rgba="0.65 0.55 0.1 1"
                    />
                </body>

                <!-- Box 2 -->
                <body name="tower2" pos="0 0 0.3">
                    <freejoint/>

                    <geom
                        type="box"
                        size="0.1 0.1 0.1"
                        mass="1"
                        rgba="0.65 0.55 0.1 1"
                    />
                </body>

                <!-- Box 3 -->
                <body name="tower3" pos="0 0 0.5">
                    <freejoint/>

                    <geom
                        type="box"
                        size="0.1 0.1 0.1"
                        mass="1"
                        rgba="0.65 0.55 0.1 1"
                    />
                </body>

                <!-- Sloped landing platform -->
                <body name="platform" pos="0 0 0.6">
                    <freejoint/>

                    <geom
                        name="platform_geom"
                        type="mesh"
                        mesh="landing_platform"
                        mass="1"
                        rgba="0.65 0.15 0.12 1"
                    />
                </body>

                <!-- Drone -->
                <body name="drone" pos="0 0 1">
                    <freejoint/>

                    <geom
                        name="drone_geom"
                        type="box"
                        size="0.03 0.03 0.03"
                        mass="1"
                        rgba="0.2 0.6 0.9 1"
                    />
                </body>

            </worldbody>

        </mujoco>
        """)

        self.data = mujoco.MjData(self.model)

        # ============================================================
        # BODY IDS
        # ============================================================

        self.drone_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "drone"
        )

        self.platform_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "platform"
        )

        self.tower_ids = [
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "tower1"
            ),
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "tower2"
            ),
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                "tower3"
            )
        ]

        # ============================================================
        # GEOM IDS
        # ============================================================

        self.drone_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "drone_geom"
        )

        self.platform_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "platform_geom"
        )

        self.ground_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "ground_geom"
        )

        # Original tower positions
        self.tower_initial_positions = np.array([
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.5]
        ])

        # Drone freejoint
        self.drone_joint_id = self.model.body_jntadr[self.drone_id]

        # Platform freejoint
        self.platform_joint_id = self.model.body_jntadr[self.platform_id]

        # ============================================================
        # PLATFORM PLANE GEOMETRY
        #
        # The top surface rises from the low side to the high side.
        #
        # Its center height in platform-local coordinates is the
        # average of the low and high side heights.
        #
        # The plane normal is derived directly from the platform slope.
        # ============================================================

        self.platform_plane_point_local = np.array([
            0.0,
            0.0,
            platform_height + height_difference / 2.0
        ])

        platform_slope = (
            height_difference
            / platform_width
        )

        self.platform_plane_normal_local = np.array([
            -platform_slope,
            0.0,
            1.0
        ])

        self.platform_plane_normal_local /= np.linalg.norm(
            self.platform_plane_normal_local
        )

        # ============================================================
        # EPISODE STATE
        # ============================================================

        self.episode_steps = 0

        # Previous drone height is used to distinguish:
        #
        # Drone starts on ground -> allowed
        # Drone was airborne -> hits ground -> failure
        #
        self.previous_drone_z = None

        # Number of consecutive simulation steps for which all
        # successful landing conditions are satisfied.
        self.landing_stable_steps = 0

        # ============================================================
        # OBSERVATION SPACE
        #
        # 20 values:
        #
        # [0:4]    Drone orientation quaternion
        # [4:7]    Drone linear velocity
        # [7:10]   Drone angular velocity
        # [10:13]  Drone position relative to platform
        # [13:16]  Drone velocity relative to platform
        # [16:20]  Platform orientation quaternion
        # ============================================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32
        )

        # ============================================================
        # ACTION SPACE
        #
        # Normalized CTBR interface:
        #
        # [T, wx, wy, wz]
        #
        # Every action is between -1 and +1.
        # The values are converted to physical units in step().
        # ============================================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # ============================================================
        # VIEWER
        # ============================================================

        self.viewer = None

    # ================================================================
    # GET OBSERVATION
    # ================================================================

    def _get_observation(self):

        # Drone position
        drone_position = self.data.xpos[self.drone_id].copy()

        # Drone orientation
        drone_orientation = self.data.xquat[self.drone_id].copy()

        # Drone velocity
        drone_velocity = self.data.qvel[
            self.model.jnt_dofadr[self.drone_joint_id]:
            self.model.jnt_dofadr[self.drone_joint_id] + 6
        ].copy()

        drone_linear_velocity = drone_velocity[:3]
        drone_angular_velocity = drone_velocity[3:]

        # Platform position
        platform_position = self.data.xpos[self.platform_id].copy()

        # Platform orientation
        platform_orientation = self.data.xquat[
            self.platform_id
        ].copy()

        # Platform velocity
        platform_velocity = self.data.qvel[
            self.model.jnt_dofadr[self.platform_joint_id]:
            self.model.jnt_dofadr[self.platform_joint_id] + 6
        ].copy()

        platform_linear_velocity = platform_velocity[:3]

        # Relative drone/platform state
        relative_position = (
            drone_position - platform_position
        )

        relative_velocity = (
            drone_linear_velocity - platform_linear_velocity
        )

        # Combine into 20-element observation
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
    # GET DISTANCE TO PLATFORM PLANE
    #
    # Returns the signed distance from the drone COM to the actual
    # tilted top plane of the platform.
    #
    # Positive = above the plane
    # Negative = below the plane
    # ================================================================

    def _get_platform_plane_distance(self):

        drone_position = self.data.xpos[
            self.drone_id
        ]

        platform_position = self.data.xpos[
            self.platform_id
        ]

        platform_rotation = self.data.xmat[
            self.platform_id
        ].reshape(3, 3)

        # Convert the point defining the platform plane into world
        # coordinates.
        plane_point_world = (
            platform_position
            + platform_rotation
            @ self.platform_plane_point_local
        )

        # Convert the plane normal into world coordinates.
        plane_normal_world = (
            platform_rotation
            @ self.platform_plane_normal_local
        )

        # Signed perpendicular distance from drone COM to plane.
        plane_distance = np.dot(
            drone_position - plane_point_world,
            plane_normal_world
        )

        return plane_distance

    # ================================================================
    # CHECK SUCCESSFUL LANDING
    #
    # A successful landing requires:
    #
    # 1. Drone is within the horizontal boundaries of the platform.
    # 2. Drone is touching the platform.
    # 3. Drone velocity is sufficiently low.
    # 4. Drone orientation is sufficiently aligned with the platform.
    # 5. All conditions remain true for the required duration.
    # ================================================================

    def _check_successful_landing(self):

        drone_position = self.data.xpos[self.drone_id]
        platform_position = self.data.xpos[self.platform_id]

        # ------------------------------------------------------------
        # 1. CHECK HORIZONTAL POSITION
        # ------------------------------------------------------------

        platform_rotation = self.data.xmat[
            self.platform_id
        ].reshape(3, 3)

        relative_position_world = (
            drone_position - platform_position
        )

        relative_position_platform = (
            platform_rotation.T @ relative_position_world
        )

        platform_half_width = 0.1

        inside_platform = (
            abs(relative_position_platform[0])
            <= platform_half_width
            and
            abs(relative_position_platform[1])
            <= platform_half_width
        )

        if not inside_platform:
            self.landing_stable_steps = 0
            return False

        # ------------------------------------------------------------
        # 2. CHECK PLATFORM CONTACT
        # ------------------------------------------------------------

        platform_contact = False

        for i in range(self.data.ncon):

            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            if (
                geom1 == self.drone_geom_id
                and geom2 == self.platform_geom_id
            ) or (
                geom2 == self.drone_geom_id
                and geom1 == self.platform_geom_id
            ):

                platform_contact = True
                break

        if not platform_contact:
            self.landing_stable_steps = 0
            return False

        # ------------------------------------------------------------
        # 3. CHECK RELATIVE VELOCITY
        # ------------------------------------------------------------

        drone_velocity = self.data.qvel[
            self.model.jnt_dofadr[self.drone_joint_id]:
            self.model.jnt_dofadr[self.drone_joint_id] + 6
        ].copy()

        platform_velocity = self.data.qvel[
            self.model.jnt_dofadr[self.platform_joint_id]:
            self.model.jnt_dofadr[self.platform_joint_id] + 6
        ].copy()

        relative_velocity = (
            drone_velocity[:3]
            - platform_velocity[:3]
        )

        relative_speed = np.linalg.norm(relative_velocity)

        if relative_speed > self.landing_max_velocity:
            self.landing_stable_steps = 0
            return False

        # ------------------------------------------------------------
        # 4. CHECK DRONE ORIENTATION
        # ------------------------------------------------------------

        drone_rotation = self.data.xmat[
            self.drone_id
        ].reshape(3, 3)

        drone_z_axis = drone_rotation[:, 2]

        platform_normal = platform_rotation[:, 2]

        alignment = np.clip(
            np.dot(drone_z_axis, platform_normal),
            -1.0,
            1.0
        )

        landing_angle = np.degrees(
            np.arccos(abs(alignment))
        )

        if landing_angle > self.landing_max_angle:
            self.landing_stable_steps = 0
            return False

        # ------------------------------------------------------------
        # DEBUG OUTPUT
        # ------------------------------------------------------------

        print(
            f"Landing stable: "
            f"{self.landing_stable_steps} / "
            f"{max(1, int(self.landing_required_time / self.model.opt.timestep))}"
        )

        # ------------------------------------------------------------
        # 5. REQUIRE STABILITY
        # ------------------------------------------------------------

        self.landing_stable_steps += 1

        required_steps = max(
            1,
            int(
                self.landing_required_time
                / self.model.opt.timestep
            )
        )

        if self.landing_stable_steps >= required_steps:
            return True

        return False

    # ================================================================
    # REWARD
    #
    # Reward components:
    #
    # 1. Progress toward platform
    # 2. Platform-plane proximity
    # 3. Penalty for being above the platform plane
    # 4. Strong exponential penalty for being below the plane
    # 5. Relative velocity penalty
    # 6. Orientation penalty
    # 7. Exponential tower disturbance penalty
    # 8. Exponential sustained high-thrust penalty
    # 9. Small time penalty
    # 10. Large successful-landing reward
    # 11. Large failure reward
    # ================================================================

    def _calculate_reward(
        self,
        terminated,
        truncated,
        termination_reason,
        thrust
    ):

        drone_position = self.data.xpos[self.drone_id]
        platform_position = self.data.xpos[self.platform_id]

        # ============================================================
        # 1. PROXIMITY / PROGRESS REWARD
        #
        # Reward only the reduction in distance to the platform.
        #
        # Positive = moved closer
        # Negative = moved farther away
        # ============================================================

        current_distance = np.linalg.norm(
            drone_position - platform_position
        )

        if self.previous_distance is None:

            proximity_reward = 0.0

        else:

            distance_progress = (
                self.previous_distance
                - current_distance
            )

            proximity_reward = (
                self.reward_proximity
                * distance_progress
            )

        self.previous_distance = current_distance

        # ============================================================
        # 2. PLATFORM-PLANE PROXIMITY
        #
        # Reward being close to the tilted platform plane.
        #
        # This uses perpendicular distance to the plane rather than
        # world Z, so it follows the platform's 30-degree tilt.
        # ============================================================

        signed_plane_distance = (
            self._get_platform_plane_distance()
        )

        absolute_plane_distance = abs(
            signed_plane_distance
        )

        plane_proximity_reward = (
            self.reward_plane_proximity
            * np.exp(
                -absolute_plane_distance
                / self.plane_reward_distance
            )
        )

        # ============================================================
        # 3. HEIGHT ABOVE PLATFORM PLANE
        #
        # Being above the plane is allowed, but increasingly costly
        # as the drone gets too far from the landing plane.
        #
        # This discourages the agent from simply remaining high above
        # the platform.
        # ============================================================

        if signed_plane_distance > 0.0:

            normalized_height = (
                signed_plane_distance
                / self.plane_reward_distance
            )

            above_plane_penalty = (
                self.penalty_above_plane
                * normalized_height ** 2
            )

        else:

            above_plane_penalty = 0.0

        # ============================================================
        # 4. BELOW PLATFORM PLANE
        #
        # Being below the landing plane is strongly discouraged.
        #
        # The penalty increases exponentially with distance below the
        # plane, making descent toward the floor increasingly costly.
        # ============================================================

        if signed_plane_distance < 0.0:

            normalized_below_distance = (
                abs(signed_plane_distance)
                / self.plane_reward_distance
            )

            below_plane_penalty = (
                self.penalty_below_plane
                * (
                    np.exp(
                        self.plane_exponential_scale
                        * normalized_below_distance
                    )
                    - 1.0
                )
            )

        else:

            below_plane_penalty = 0.0

        # ============================================================
        # 5. RELATIVE VELOCITY PENALTY
        #
        # Penalize movement relative to the platform.
        #
        # Squared velocity makes high speeds increasingly expensive.
        # ============================================================

        drone_velocity = self.data.qvel[
            self.model.jnt_dofadr[self.drone_joint_id]:
            self.model.jnt_dofadr[self.drone_joint_id] + 6
        ].copy()

        platform_velocity = self.data.qvel[
            self.model.jnt_dofadr[self.platform_joint_id]:
            self.model.jnt_dofadr[self.platform_joint_id] + 6
        ].copy()

        relative_velocity = (
            drone_velocity[:3]
            - platform_velocity[:3]
        )

        relative_speed = np.linalg.norm(relative_velocity)

        velocity_penalty = (
            self.penalty_velocity
            * relative_speed ** 2
        )

        # ============================================================
        # 6. ORIENTATION PENALTY
        #
        # Penalize the angle between the drone and the platform.
        #
        # This automatically adapts to the current platform angle.
        # ============================================================

        drone_rotation = self.data.xmat[
            self.drone_id
        ].reshape(3, 3)

        platform_rotation = self.data.xmat[
            self.platform_id
        ].reshape(3, 3)

        drone_z_axis = drone_rotation[:, 2]
        platform_normal = platform_rotation[:, 2]

        alignment = np.clip(
            np.dot(drone_z_axis, platform_normal),
            -1.0,
            1.0
        )

        orientation_angle = np.arccos(abs(alignment))

        orientation_penalty = (
            self.penalty_orientation
            * orientation_angle ** 2
        )

        # ============================================================
        # 7. EXPONENTIAL TOWER-DISTURBANCE PENALTY
        #
        # Find the largest horizontal displacement of any tower block.
        #
        # Small displacement -> small cost
        # Large displacement -> rapidly increasing cost
        # Collapse -> handled separately as failure
        # ============================================================

        maximum_tower_displacement = 0.0

        for i, tower_id in enumerate(self.tower_ids):

            tower_position = self.data.xpos[tower_id]

            initial_position = self.tower_initial_positions[i]

            horizontal_displacement = np.linalg.norm(
                tower_position[:2]
                - initial_position[:2]
            )

            maximum_tower_displacement = max(
                maximum_tower_displacement,
                horizontal_displacement
            )

        normalized_tower_displacement = (
            maximum_tower_displacement
            / self.tower_max_horizontal_displacement
        )

        tower_penalty = (
            self.penalty_tower
            * (
                np.exp(
                    self.tower_exponential_scale
                    * normalized_tower_displacement
                )
                - 1.0
            )
        )

        # ============================================================
        # 8. SUSTAINED HIGH-THRUST PENALTY
        #
        # Normal thrust is fine.
        #
        # Once thrust exceeds the high-thrust threshold:
        #
        # - the counter increases
        # - sustained high thrust becomes increasingly expensive
        #
        # A short burst is therefore acceptable, while holding high
        # thrust continuously becomes increasingly undesirable.
        # ============================================================

        normalized_thrust = np.clip(
            thrust / self.max_thrust,
            0.0,
            1.0
        )

        if normalized_thrust >= self.high_thrust_threshold:

            self.high_thrust_steps += 1

        else:

            self.high_thrust_steps = 0

        if self.high_thrust_steps > 0:

            normalized_duration = (
                self.high_thrust_steps
                * self.model.opt.timestep
            )

            thrust_penalty = (
                self.penalty_high_thrust
                * (
                    np.exp(
                        self.thrust_exponential_scale
                        * normalized_duration
                    )
                    - 1.0
                )
            )

        else:

            thrust_penalty = 0.0

        # ============================================================
        # 9. TIME PENALTY
        #
        # Small cost every simulation step.
        # ============================================================

        time_penalty = self.penalty_time

        # ============================================================
        # COMBINE STEP REWARD
        # ============================================================

        reward = (
            proximity_reward
            + plane_proximity_reward
            - above_plane_penalty
            - below_plane_penalty
            - velocity_penalty
            - orientation_penalty
            - tower_penalty
            - thrust_penalty
            - time_penalty
        )

        # ============================================================
        # 10. SUCCESS / FAILURE TERMS
        #
        # These dominate the smaller shaping rewards.
        # ============================================================

        if termination_reason == "successful_landing":

            reward += self.reward_success

        elif termination_reason == "ground_collision":

            reward -= self.penalty_ground_collision

        elif termination_reason in [
            "too_far",
            "tower_collapse"
        ]:

            reward -= self.penalty_failure

        return reward

    # ================================================================
    # CHECK FAILURE / TERMINATION
    # ================================================================

    def _check_termination(self):

        drone_position = self.data.xpos[self.drone_id]

        platform_position = self.data.xpos[self.platform_id]

        # ------------------------------------------------------------
        # 1. SUCCESSFUL LANDING
        # ------------------------------------------------------------

        if self._check_successful_landing():
            return True, "successful_landing"

        # ------------------------------------------------------------
        # 2. DRONE HIT GROUND
        #
        # Detect actual physical contact between the drone and ground.
        #
        # The previous height check is retained so that a drone which
        # starts on the ground is not immediately treated as a failure.
        # ------------------------------------------------------------

        ground_contact = False

        for i in range(self.data.ncon):

            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            if (
                geom1 == self.drone_geom_id
                and geom2 == self.ground_geom_id
            ) or (
                geom2 == self.drone_geom_id
                and geom1 == self.ground_geom_id
            ):

                ground_contact = True
                break

        if ground_contact:

            if self.previous_drone_z > self.ground_height:
                return True, "ground_collision"

        # ------------------------------------------------------------
        # 3. DRONE MOVED TOO FAR AWAY
        # ------------------------------------------------------------

        horizontal_distance = np.linalg.norm(
            drone_position[:2] - platform_position[:2]
        )

        if horizontal_distance > self.max_horizontal_distance:
            return True, "too_far"

        # ------------------------------------------------------------
        # 4. TOWER COLLAPSE
        # ------------------------------------------------------------

        for i, tower_id in enumerate(self.tower_ids):

            tower_position = self.data.xpos[tower_id]

            initial_position = self.tower_initial_positions[i]

            horizontal_displacement = np.linalg.norm(
                tower_position[:2] - initial_position[:2]
            )

            if (
                horizontal_displacement
                > self.tower_max_horizontal_displacement
            ):
                return True, "tower_collapse"

            if tower_position[2] < self.tower_min_height:
                return True, "tower_collapse"

        # ------------------------------------------------------------
        # 5. TIME LIMIT
        # ------------------------------------------------------------

        elapsed_time = (
            self.episode_steps
            * self.model.opt.timestep
        )

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

        self.previous_drone_z = self.data.xpos[
            self.drone_id
        ][2]

        self.landing_stable_steps = 0

        self.high_thrust_steps = 0

        drone_position = self.data.xpos[self.drone_id]
        platform_position = self.data.xpos[self.platform_id]

        self.previous_distance = np.linalg.norm(
            drone_position - platform_position
        )

        # Initialize distance to the tilted platform plane.
        self.previous_plane_distance = abs(
            self._get_platform_plane_distance()
        )

        observation = self._get_observation()

        info = {}

        return observation, info

    # ================================================================
    # STEP
    # ================================================================

    def step(self, action):

        # ------------------------------------------------------------
        # Remember drone's position BEFORE the simulation step.
        # ------------------------------------------------------------

        self.previous_drone_z = self.data.xpos[
            self.drone_id
        ][2]

        # ------------------------------------------------------------
        # Convert normalized RL action into physical CTBR commands.
        # ------------------------------------------------------------

        thrust = (
            (action[0] + 1.0)
            / 2.0
            * self.max_thrust
        )

        desired_body_rates = np.array([
            action[1] * self.max_body_rate,
            action[2] * self.max_body_rate,
            action[3] * self.max_body_rate
        ])

        # ------------------------------------------------------------
        # CURRENT BODY RATES
        # ------------------------------------------------------------

        drone_velocity = self.data.qvel[
            self.model.jnt_dofadr[self.drone_joint_id]:
            self.model.jnt_dofadr[self.drone_joint_id] + 6
        ].copy()

        current_body_rates = drone_velocity[3:]

        # ------------------------------------------------------------
        # BODY-RATE CONTROLLER
        # ------------------------------------------------------------

        rate_error = (
            desired_body_rates
            - current_body_rates
        )

        torque = (
            self.rate_controller_gain
            * rate_error
        )

        # ------------------------------------------------------------
        # LIMIT TORQUE
        # ------------------------------------------------------------

        torque = np.clip(
            torque,
            -self.max_torque,
            self.max_torque
        )

        # ------------------------------------------------------------
        # DRONE BODY Z-AXIS
        # ------------------------------------------------------------

        rotation_matrix = self.data.xmat[
            self.drone_id
        ].reshape(3, 3)

        body_z_axis = rotation_matrix[:, 2]

        # ------------------------------------------------------------
        # APPLY THRUST ALONG DRONE BODY Z-AXIS
        # ------------------------------------------------------------

        thrust_force = thrust * body_z_axis

        self.data.xfrc_applied[
            self.drone_id,
            0:3
        ] = thrust_force

        # ------------------------------------------------------------
        # APPLY BODY-FRAME TORQUE
        # ------------------------------------------------------------

        self.data.xfrc_applied[
            self.drone_id,
            3:6
        ] = torque

        # ------------------------------------------------------------
        # ADVANCE SIMULATION
        # ------------------------------------------------------------

        mujoco.mj_step(self.model, self.data)

        self.episode_steps += 1

        # ------------------------------------------------------------
        # CLEAR APPLIED FORCE/TORQUE
        # ------------------------------------------------------------

        self.data.xfrc_applied[self.drone_id] = 0.0

        observation = self._get_observation()

        # ------------------------------------------------------------
        # CHECK TERMINATION
        # ------------------------------------------------------------

        terminated, termination_reason = self._check_termination()

        truncated = False

        if termination_reason == "time_limit":
            truncated = True

        # ------------------------------------------------------------
        # CALCULATE REWARD
        # ------------------------------------------------------------

        reward = self._calculate_reward(
            terminated,
            truncated,
            termination_reason,
            thrust
        )

        # ------------------------------------------------------------
        # INFO
        # ------------------------------------------------------------

        info = {
            "thrust": thrust,
            "wx": desired_body_rates[0],
            "wy": desired_body_rates[1],
            "wz": desired_body_rates[2],
            "torque_x": torque[0],
            "torque_y": torque[1],
            "torque_z": torque[2],
            "termination_reason": termination_reason
        }

        return observation, reward, terminated, truncated, info

    # ================================================================
    # RENDER
    # ================================================================

    def render(self):

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data
            )

        self.viewer.sync()

    # ================================================================
    # CLOSE
    # ================================================================

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

    # ============================================================
    # MANUAL CONTROL STATE
    # ============================================================

    thrust_action = -1.0

    wx_action = 0.0
    wy_action = 0.0
    wz_action = 0.0

    # ============================================================
    # KEYBOARD CONTROL
    #
    # UP ARROW:
    #   Pitch forward
    #
    # DOWN ARROW:
    #   Pitch backward
    #
    # LEFT ARROW:
    #   Roll left
    #
    # RIGHT ARROW:
    #   Roll right
    #
    # Pressing the same arrow again stops that rotation.
    #
    # P:
    #   Toggle thrust on/off
    #
    # R:
    #   Reset drone
    #
    # ============================================================

    def key_callback(keycode):

        nonlocal_state = None

        # ------------------------------------------------------------
        # P = THRUST
        # ------------------------------------------------------------

        if keycode == ord("P") or keycode == ord("p"):

            if thrust_action_holder[0] == -1.0:
                thrust_action_holder[0] = 1.0
            else:
                thrust_action_holder[0] = -1.0

        # ------------------------------------------------------------
        # UP ARROW = PITCH FORWARD
        # ------------------------------------------------------------

        elif keycode == 265:

            if wy_action_holder[0] == 0.0:
                wy_action_holder[0] = 0.5
            else:
                wy_action_holder[0] = 0.0

        # ------------------------------------------------------------
        # DOWN ARROW = PITCH BACKWARD
        # ------------------------------------------------------------

        elif keycode == 264:

            if wy_action_holder[0] == 0.0:
                wy_action_holder[0] = -0.5
            else:
                wy_action_holder[0] = 0.0

        # ------------------------------------------------------------
        # LEFT ARROW = ROLL LEFT
        # ------------------------------------------------------------

        elif keycode == 263:

            if wx_action_holder[0] == 0.0:
                wx_action_holder[0] = -0.5
            else:
                wx_action_holder[0] = 0.0

        # ------------------------------------------------------------
        # RIGHT ARROW = ROLL RIGHT
        # ------------------------------------------------------------

        elif keycode == 262:

            if wx_action_holder[0] == 0.0:
                wx_action_holder[0] = 0.5
            else:
                wx_action_holder[0] = 0.0

        # ------------------------------------------------------------
        # R = RESET
        # ------------------------------------------------------------

        elif keycode == ord("R") or keycode == ord("r"):

            env.reset()

            thrust_action_holder[0] = -1.0
            wx_action_holder[0] = 0.0
            wy_action_holder[0] = 0.0
            wz_action_holder[0] = 0.0

    # ============================================================
    # MUTABLE KEYBOARD STATE
    # ============================================================

    thrust_action_holder = [thrust_action]
    wx_action_holder = [wx_action]
    wy_action_holder = [wy_action]
    wz_action_holder = [wz_action]

    print()
    print("==============================================")
    print("MANUAL DRONE CONTROL")
    print("==============================================")
    print()
    print("UP       = pitch forward")
    print("DOWN     = pitch backward")
    print("LEFT     = roll left")
    print("RIGHT    = roll right")
    print("P        = thrust on/off")
    print("R        = reset")
    print()
    print("Press the same arrow again to stop rotating.")
    print()
    print("Close the MuJoCo viewer to stop.")
    print()

    # ============================================================
    # LAUNCH VIEWER
    # ============================================================

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        key_callback=key_callback
    ) as viewer:

        env.viewer = viewer

        while viewer.is_running():

            action = np.array([
                thrust_action_holder[0],
                wx_action_holder[0],
                wy_action_holder[0],
                wz_action_holder[0]
            ], dtype=np.float32)

            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(
                    "EPISODE ENDED:",
                    info["termination_reason"],
                    "Reward:",
                    reward
                )

                observation, info = env.reset()
                viewer.sync()
                continue

            viewer.sync()

            time.sleep(0.002)

    env.close()