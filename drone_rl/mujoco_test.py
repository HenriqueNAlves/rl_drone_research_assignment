import mujoco
import mujoco.viewer
import numpy as np
import time



# platform parameters

platform_angle = 30       # deg
platform_width = 0.2      # m
platform_height = 0.1     # m



# calculate height difference needed for desired angle
height_difference = (
    platform_width
    * np.tan(np.radians(platform_angle))
)

highside_height = platform_height + height_difference


# mujoco model

model = mujoco.MjModel.from_xml_string(f"""
<mujoco>

    <asset>

        <!-- Triangular-prism / wedge -->
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
                1 6 2

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
            type="plane"
            size="5 5 0.1"
        />

        <!-- Box 1 -->
        <body pos="0 0 0.1">
            <freejoint/>

            <geom
                type="box"
                size="0.1 0.1 0.1"
                mass="1"
                rgba="0.65 0.55 0.1 1"
            />
        </body>

        <!-- Box 2 -->
        <body pos="0 0 0.3">
            <freejoint/>

            <geom
                type="box"
                size="0.1 0.1 0.1"
                mass="1"
                rgba="0.65 0.55 0.1 1"
            />
        </body>

        <!-- Box 3 -->
        <body pos="0 0 0.5">
            <freejoint/>

            <geom
                type="box"
                size="0.1 0.1 0.1"
                mass="1"
                rgba="0.65 0.55 0.1 1"
            />
        </body>

        <!-- Sloped landing platform -->
        <body pos="0 0 0.6">
            <freejoint/>

            <geom
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
                type="box"
                size="0.03 0.03 0.03"
                mass="1"
                rgba="0.2 0.6 0.9 1"
            />
        </body>

    </worldbody>

</mujoco>
""")

data = mujoco.MjData(model)

drone_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "drone"
)

drone_joint_id = model.body_jntadr[drone_id]

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        mujoco.mj_step(model, data)

        # Drone state
        position = data.xpos[drone_id].copy()
        orientation = data.xquat[drone_id].copy()

        velocity = data.qvel[model.jnt_dofadr[drone_joint_id]:
                            model.jnt_dofadr[drone_joint_id] + 6].copy()

        linear_velocity = velocity[:3]
        angular_velocity = velocity[3:]

        print(
            "Position:", position,
            "Linear velocity:", linear_velocity,
            "Orientation:", orientation,
            "Angular velocity:", angular_velocity
        )

        viewer.sync()

        time.sleep(0.002)