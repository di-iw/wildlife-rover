
import time

import pybullet as p
import pybullet_data


# ==========================================
# Simulation parameters
# ==========================================

TIME_STEP = 1.0 / 240.0
MAX_MOTOR_FORCE = 100.0

# Dimensions from the PyBullet Husky URDF
WHEEL_RADIUS = 0.17775
TRACK_WIDTH = 0.5708

# Keyboard control speed
LINEAR_SPEED = 0.4       # m/s
ANGULAR_SPEED = 0.8      # rad/s

WHEEL_JOINT_NAMES = (
    "front_left_wheel",
    "front_right_wheel",
    "rear_left_wheel",
    "rear_right_wheel",
)


# ==========================================
# Robot utilities
# ==========================================

def find_wheel_joints(robot_id):
    """Find Husky wheel joint IDs by name."""

    joints = {}

    for joint_id in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, joint_id)
        joint_name = joint_info[1].decode("utf-8")

        if joint_name in WHEEL_JOINT_NAMES:
            joints[joint_name] = joint_id

    missing = [
        name for name in WHEEL_JOINT_NAMES
        if name not in joints
    ]

    if missing:
        raise RuntimeError(
            f"Husky wheel joints not found: {missing}"
        )

    print("Detected Husky wheel joints:")

    for name in WHEEL_JOINT_NAMES:
        print(f"  {name}: joint ID {joints[name]}")

    return joints


def set_drive_command(
    robot_id,
    wheel_joints,
    linear_velocity,
    angular_velocity,
):
    """
    Convert linear/angular velocity
    into left/right wheel angular velocities.
    """

    # Differential drive kinematics
    left_linear = (
        linear_velocity
        - angular_velocity * TRACK_WIDTH / 2.0
    )

    right_linear = (
        linear_velocity
        + angular_velocity * TRACK_WIDTH / 2.0
    )

    left_angular = left_linear / WHEEL_RADIUS
    right_angular = right_linear / WHEEL_RADIUS

    # Wheel order:
    # front-left, front-right, rear-left, rear-right
    joint_indices = [
        wheel_joints[name]
        for name in WHEEL_JOINT_NAMES
    ]

    target_velocities = [
        left_angular,
        right_angular,
        left_angular,
        right_angular,
    ]

    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=joint_indices,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocities=target_velocities,
        forces=[MAX_MOTOR_FORCE] * 4,
    )


def key_is_down(key_events, key):
    """Check whether an ASCII key is currently pressed."""

    return any(
        key_events.get(ord(char), 0) & p.KEY_IS_DOWN
        for char in (key.lower(), key.upper())
    )


# ==========================================
# Main
# ==========================================

def main():

    client_id = p.connect(p.GUI)

    if client_id < 0:
        raise RuntimeError(
            "Could not connect to the PyBullet GUI."
        )

    try:
        # ----------------------------------
        # Initialize simulation
        # ----------------------------------

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(0.0, 0.0, -9.81)

        p.setTimeStep(TIME_STEP)
        p.setRealTimeSimulation(0)

        p.setPhysicsEngineParameter(
            numSolverIterations=100
        )

        # ----------------------------------
        # Load ground
        # ----------------------------------

        plane_id = p.loadURDF("plane.urdf")

        p.changeDynamics(
            plane_id,
            -1,
            lateralFriction=1.0,
        )

        # ----------------------------------
        # Load Husky
        # ----------------------------------

        robot_id = p.loadURDF(
            "husky/husky.urdf",
            basePosition=[0.0, 0.0, 0.1],
            useFixedBase=False,
        )

        # Identify wheel joints
        wheel_joints = find_wheel_joints(robot_id)

        # Set wheel-ground friction explicitly
        for joint_id in wheel_joints.values():
            p.changeDynamics(
                robot_id,
                joint_id,
                lateralFriction=1.0,
                rollingFriction=0.0,
            )

        # ----------------------------------
        # Camera view
        # ----------------------------------

        p.resetDebugVisualizerCamera(
            cameraDistance=3.5,
            cameraYaw=45.0,
            cameraPitch=-30.0,
            cameraTargetPosition=[0.0, 0.0, 0.3],
        )

        # ----------------------------------
        # Keyboard instructions
        # ----------------------------------

        print("\nSTEP1: Husky keyboard control")
        print("  W     : Forward")
        print("  S     : Backward")
        print("  A     : Turn left")
        print("  D     : Turn right")
        print("  SPACE : Stop")
        print("  Q     : Quit")
        print("\nClick the PyBullet window before using the keyboard.")

        frame_count = 0

        # ----------------------------------
        # Simulation loop
        # ----------------------------------

        while p.isConnected(client_id):

            key_events = p.getKeyboardEvents()

            if key_is_down(key_events, "q"):
                break

            linear_velocity = 0.0
            angular_velocity = 0.0

            if key_is_down(key_events, "w"):
                linear_velocity += LINEAR_SPEED

            if key_is_down(key_events, "s"):
                linear_velocity -= LINEAR_SPEED

            if key_is_down(key_events, "a"):
                angular_velocity += ANGULAR_SPEED

            if key_is_down(key_events, "d"):
                angular_velocity -= ANGULAR_SPEED

            if (
                key_events.get(ord(" "), 0)
                & p.KEY_IS_DOWN
            ):
                linear_velocity = 0.0
                angular_velocity = 0.0

            # Apply wheel velocity commands
            set_drive_command(
                robot_id,
                wheel_joints,
                linear_velocity,
                angular_velocity,
            )

            # Physics update
            p.stepSimulation()

            # Follow the robot with the GUI camera
            if frame_count % 24 == 0:

                position, _ = p.getBasePositionAndOrientation(
                    robot_id
                )

                p.resetDebugVisualizerCamera(
                    cameraDistance=3.5,
                    cameraYaw=45.0,
                    cameraPitch=-30.0,
                    cameraTargetPosition=[
                        position[0],
                        position[1],
                        0.3,
                    ],
                )

            frame_count += 1

            time.sleep(TIME_STEP)

    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)


if __name__ == "__main__":
    main()
