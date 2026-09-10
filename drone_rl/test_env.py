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

        # ============================================================
        # HORIZONTAL PLATFORM PROGRESS
        #
        # This is now the primary positioning objective.
        #
        # The agent is rewarded for reducing its horizontal distance
        # from the centre of the platform.
        # ============================================================

        self.reward_proximity = 10.0      # PLACEHOLDER!!!!!!!

        # ============================================================
        # FINITE PLATFORM SURFACE PROGRESS
        #
        # This rewards progress toward the actual finite tilted
        # platform surface rather than an infinite plane.
        #
        # The reward is only activated when the drone is reasonably
        # close to the platform horizontally.
        # ============================================================

        self.reward_surface_proximity = 5.0       # PLACEHOLDER!!!!!!!
        self.surface_activation_distance = 0.3     # m  # PLACEHOLDER!!!!!!!

        # ============================================================
        # RELATIVE VELOCITY PENALTY
        # ============================================================

        self.penalty_velocity = 1.0       # PLACEHOLDER!!!!!!!

        # ============================================================
        # ORIENTATION PENALTY
        # ============================================================

        self.penalty_orientation = 1.0    # PLACEHOLDER!!!!!!!

        # ============================================================
        # TOWER DISTURBANCE PENALTY
        # ============================================================

        self.penalty_tower = 2.0          # PLACEHOLDER!!!!!!!
        self.tower_exponential_scale = 5.0  # PLACEHOLDER!!!!!!!

        # ============================================================
        # HIGH-THRUST PENALTY
        # ============================================================

        self.high_thrust_threshold = 0.8  # normalized thrust  # PLACEHOLDER!!!!!!!
        self.penalty_high_thrust = 0.1    # PLACEHOLDER!!!!!!!
        self.thrust_exponential_scale = 5.0  # PLACEHOLDER!!!!!!!

        # ============================================================
        # TIME PENALTY
        # ============================================================

        self.penalty_time = 0.01          # PLACEHOLDER!!!!!!!

        # ============================================================
        # OLD TILTED PLATFORM PLANE REWARD
        #
        # These parameters are retained for reference but are no
        # longer used in the active reward calculation.
        #
        # The old reward treated the platform as an infinite plane.
        # This allowed the drone to receive a positive reward simply
        # by approaching the plane, even when it was horizontally
        # nowhere near the actual platform.
        # ============================================================

        # self.reward_plane_proximity = 5.0  # COMMENTED OUT
        # self.plane_reward_distance = 0.1  # m  # COMMENTED OUT
        # self.penalty_above_plane = 1.0  # COMMENTED OUT
        # self.penalty_below_plane = 5.0  # COMMENTED OUT
        # self.plane_exponential_scale = 5.0  # COMMENTED OUT

        # ============================================================
        # INTERNAL REWARD STATE
        # ============================================================

        # Number of consecutive steps spent at high thrust
        self.high_thrust_steps = 0

        # Previous horizontal distance from drone to platform centre
        self.previous_horizontal_distance = None

        # Previous distance from drone to finite platform surface
        self.previous_surface_distance = None

        # ============================================================
        # OLD DISTANCE STATE
        #
        # Retained for reference.
        # ============================================================

        # self.previous_distance = None  # COMMENTED OUT
        # self.previous_plane_distance = None  # COMMENTED OUT

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

        # ============================================================
        # ORIGINAL TOWER POSITIONS
        # ============================================================

        self.tower_initial_positions = np.array([
            [0.0, 0.0, 0.1],
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.5]
        ])

        # ============================================================
        # DRONE FREEJOINT
        # ============================================================

        self.drone_joint_id = self.model.body_jntadr[self.drone_id]

        # ============================================================
        # PLATFORM FREEJOINT
        # ============================================================

        self.platform_joint_id = self.model.body_jntadr[self.platform_id]

        # ============================================================
        # PLATFORM PLANE GEOMETRY
        #
        # The top surface rises from the low side to the high side.
        #
        # Its centre height in platform-local coordinates is the
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

        self.platform_slope = platform_slope

        self.platform_half_width = platform_width / 2.0

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
    #
    # This is retained as a geometric measurement.
    #
    # It is NO LONGER directly rewarded as an infinite-plane target.
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
    # GET DISTANCE TO FINITE PLATFORM SURFACE
    #
    # Unlike _get_platform_plane_distance(), this function considers
    # the actual finite 0.2 x 0.2 m landing surface.
    #
    # The drone position is first expressed in platform coordinates.
    # The closest point on the finite platform surface is then found.
    #
    # This prevents the reward from treating the entire infinite
    # tilted plane as a valid landing target.
    # ================================================================

    def _get_platform_surface_distance(self):

        drone_position = self.data.xpos[
            self.drone_id
        ]

        platform_position = self.data.xpos[
            self.platform_id
        ]

        platform_rotation = self.data.xmat[
            self.platform_id
        ].reshape(3, 3)

        # Drone position expressed in platform-local coordinates.
        relative_position_world = (
            drone_position - platform_position
        )

        relative_position_platform = (
            platform_rotation.T
            @ relative_position_world
        )

        # ------------------------------------------------------------
        # Find the closest x/y position on the finite platform.
        # ------------------------------------------------------------

        closest_x = np.clip(
            relative_position_platform[0],
            -self.platform_half_width,
            self.platform_half_width
        )

        closest_y = np.clip(
            relative_position_platform[1],
            -self.platform_half_width,
            self.platform_half_width
        )

        # ------------------------------------------------------------
        # Height of the tilted top surface at the closest x position.
        #
        # The low side is at x = -0.1.
        # The high side is at x = +0.1.
        # ------------------------------------------------------------

        closest_z = (
            0.1
            + self.platform_slope
            * (closest_x + self.platform_half_width)
        )

        closest_point_platform = np.array([
            closest_x,
            closest_y,
            closest_z
        ])

        # Convert closest point back into world coordinates.
        closest_point_world = (
            platform_position
            + platform_rotation
            @ closest_point_platform
        )

        # Euclidean distance to the finite platform surface.
        surface_distance = np.linalg.norm(
            drone_position - closest_point_world
        )

        return surface_distance

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
    # 1. Horizontal progress toward the platform
    # 2. Progress toward the finite tilted platform surface
    # 3. Relative velocity penalty, increasingly important near
    #    the platform
    # 4. Orientation penalty, increasingly important near the
    #    platform
    # 5. Tower disturbance penalty
    # 6. Sustained high-thrust penalty
    # 7. Small time penalty
    # 8. Large successful-landing reward
    # 9. Large failure penalty
    #
    # IMPORTANT:
    #
    # The old infinite-plane proximity reward has been removed from
    # the active calculation. It is retained below as commented-out
    # code for reference.
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

        platform_rotation = self.data.xmat[
            self.platform_id
        ].reshape(3, 3)

        # ============================================================
        # 1. HORIZONTAL PLATFORM PROGRESS
        #
        # This is now the primary navigation reward.
        #
        # Only horizontal x/y distance matters here.
        #
        # Positive = moved horizontally closer.
        # Negative = moved horizontally farther away.
        # ============================================================

        relative_position_world = (
            drone_position - platform_position
        )

        relative_position_platform = (
            platform_rotation.T
            @ relative_position_world
        )

        horizontal_distance = np.linalg.norm(
            relative_position_platform[:2]
        )

        if self.previous_horizontal_distance is None:

            horizontal_progress_reward = 0.0

        else:

            horizontal_progress = (
                self.previous_horizontal_distance
                - horizontal_distance
            )

            horizontal_progress_reward = (
                self.reward_proximity
                * horizontal_progress
            )

        self.previous_horizontal_distance = horizontal_distance

        # ============================================================
        # 2. FINITE PLATFORM SURFACE PROGRESS
        #
        # Reward progress toward the actual finite platform surface.
        #
        # This reward is disabled while the drone is far away from
        # the platform horizontally.
        #
        # This prevents the old "infinite plane" exploit.
        # ============================================================

        surface_distance = (
            self._get_platform_surface_distance()
        )

        if self.previous_surface_distance is None:

            surface_progress_reward = 0.0

        else:

            surface_progress = (
                self.previous_surface_distance
                - surface_distance
            )

            if horizontal_distance <= self.surface_activation_distance:

                surface_progress_reward = (
                    self.reward_surface_proximity
                    * surface_progress
                )

            else:

                surface_progress_reward = 0.0

        self.previous_surface_distance = surface_distance

        # ============================================================
        # OLD INFINITE-PLANE PROXIMITY REWARD
        #
        # COMMENTED OUT.
        #
        # This was the source of the unwanted behaviour because the
        # drone could approach the infinite plane without approaching
        # the finite landing platform.
        # ============================================================

        # signed_plane_distance = (
        #     self._get_platform_plane_distance()
        # )
        #
        # absolute_plane_distance = abs(
        #     signed_plane_distance
        # )
        #
        # plane_proximity_reward = (
        #     self.reward_plane_proximity
        #     * np.exp(
        #         -absolute_plane_distance
        #         / self.plane_reward_distance
        #     )
        # )

        # ============================================================
        # OLD ABOVE-PLANE PENALTY
        #
        # COMMENTED OUT.
        # ============================================================

        # if signed_plane_distance > 0.0:
        #
        #     normalized_height = (
        #         signed_plane_distance
        #         / self.plane_reward_distance
        #     )
        #
        #     above_plane_penalty = (
        #         self.penalty_above_plane
        #         * normalized_height ** 2
        #     )
        #
        # else:
        #
        #     above_plane_penalty = 0.0

        # ============================================================
        # OLD BELOW-PLANE PENALTY
        #
        # COMMENTED OUT.
        #
        # The exponential penalty could become extremely large and
        # discourage the agent from descending at all.
        # ============================================================

        # if signed_plane_distance < 0.0:
        #
        #     normalized_below_distance = (
        #         abs(signed_plane_distance)
        #         / self.plane_reward_distance
        #     )
        #
        #     below_plane_penalty = (
        #         self.penalty_below_plane
        #         * (
        #             np.exp(
        #                 self.plane_exponential_scale
        #                 * normalized_below_distance
        #             )
        #             - 1.0
        #         )
        #     )
        #
        # else:
        #
        #     below_plane_penalty = 0.0

        # ============================================================
        # 3. RELATIVE VELOCITY PENALTY
        #
        # Velocity is penalized more strongly when the drone is near
        # the platform.
        #
        # Far away:
        #     The drone is allowed to move aggressively.
        #
        # Near platform:
        #     The drone is encouraged to slow down for landing.
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

        near_platform_factor = np.exp(
            -horizontal_distance
            / self.surface_activation_distance
        )

        velocity_penalty = (
            self.penalty_velocity
            * near_platform_factor
            * relative_speed ** 2
        )

        # ============================================================
        # 4. ORIENTATION PENALTY
        #
        # Orientation is also made increasingly important as the
        # drone approaches the platform.
        #
        # This allows aggressive manoeuvring when far away while
        # encouraging platform alignment during the final approach.
        # ============================================================

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

        orientation_angle = np.arccos(abs(alignment))

        orientation_penalty = (
            self.penalty_orientation
            * near_platform_factor
            * orientation_angle ** 2
        )

        # ============================================================
        # 5. EXPONENTIAL TOWER-DISTURBANCE PENALTY
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
        # 6. SUSTAINED HIGH-THRUST PENALTY
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
        # 7. TIME PENALTY
        #
        # Small cost every simulation step.
        # ============================================================

        time_penalty = self.penalty_time

        # ============================================================
        # COMBINE STEP REWARD
        #
        # OLD:
        #
        # reward = (
        #     proximity_reward
        #     + plane_proximity_reward
        #     - above_plane_penalty
        #     - below_plane_penalty
        #     ...
        # )
        #
        # NEW:
        #
        # Horizontal progress is the main navigation objective.
        #
        # Surface progress becomes active only near the platform.
        #
        # Velocity and orientation become more important near the
        # platform.
        # ============================================================

        reward = (
            horizontal_progress_reward
            + surface_progress_reward
            - velocity_penalty
            - orientation_penalty
            - tower_penalty
            - thrust_penalty
            - time_penalty
        )

        # ============================================================
        # 8. SUCCESS / FAILURE TERMS
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

        platform_rotation = self.data.xmat[
            self.platform_id
        ].reshape(3, 3)

        relative_position_world = (
            drone_position - platform_position
        )

        relative_position_platform = (
            platform_rotation.T
            @ relative_position_world
        )

        # Initialize horizontal distance.
        self.previous_horizontal_distance = np.linalg.norm(
            relative_position_platform[:2]
        )

        # Initialize distance to the finite platform surface.
        self.previous_surface_distance = (
            self._get_platform_surface_distance()
        )

        # ============================================================
        # OLD RESET DISTANCE STATE
        #
        # Retained for reference.
        # ============================================================

        # self.previous_distance = np.linalg.norm(
        #     drone_position - platform_position
        # )

        # self.previous_plane_distance = abs(
        #     self._get_platform_plane_distance()
        # )

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

        # ============================================================
        # CALCULATE DEBUG VALUES FOR INFO
        # ============================================================

        drone_position = self.data.xpos[self.drone_id]
        platform_position = self.data.xpos[self.platform_id]

        platform_rotation = self.data.xmat[
            self.platform_id
        ].reshape(3, 3)

        relative_position_world = (
            drone_position - platform_position
        )

        relative_position_platform = (
            platform_rotation.T
            @ relative_position_world
        )

        horizontal_distance = np.linalg.norm(
            relative_position_platform[:2]
        )

        surface_distance = (
            self._get_platform_surface_distance()
        )

        plane_distance = (
            self._get_platform_plane_distance()
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

            # Reward/debug geometry
            "horizontal_distance": horizontal_distance,
            "surface_distance": surface_distance,
            "plane_distance": plane_distance,

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
