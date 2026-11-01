"""STEP2: Add a visual-only animal billboard to the PyBullet Husky scene.

Run from the project root:
    python src/step2_animal_object.py

This script reuses keyboard driving helpers from step1_flat_husky.py.
"""

from pathlib import Path
import time

import cv2
import numpy as np
import pybullet as p
import pybullet_data

from step1_flat_husky import (
    TIME_STEP,
    find_wheel_joints,
    set_drive_command,
    key_is_down,
)


# ============================================================
# Paths and simulation settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANIMAL_IMAGE_PATH = PROJECT_ROOT / "assets" / "animals" / "bear1.jpg"

# Keyboard driving speeds (same convention as STEP1)
LINEAR_SPEED = 0.4       # m/s
ANGULAR_SPEED = 0.8      # rad/s

# Animal billboard position in the world.
# Husky faces +X when loaded with its default orientation.
ANIMAL_POSITION = [3.0, 0.0, 0.95]  # center of the image plane [m]
ANIMAL_HEIGHT = 0.95               # width is derived from image aspect ratio [m]

# Virtual front camera (relative to Husky base frame)
CAMERA_LOCAL_POSITION = [0.42, 0.0, 0.72]
CAMERA_LOOK_AHEAD = 2.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOV = 60.0
CAMERA_NEAR = 0.05
CAMERA_FAR = 20.0
CAMERA_UPDATE_INTERVAL = 24  # 240 Hz physics / 24 = 10 Hz preview
CAMERA_WINDOW_NAME = "STEP2 - Husky Front Camera"


# ============================================================
# Animal visual object
# ============================================================

def create_animal_billboard(image_path: Path):
    """Create a textured vertical plane with no collision geometry."""

    if not image_path.is_file():
        raise FileNotFoundError(
            "Animal image was not found.\n"
            f"Expected file: {image_path}\n"
            "Place a JPG or PNG image at assets/animals/deer.jpg, "
            "or change ANIMAL_IMAGE_PATH in this script."
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read the image: {image_path}")

    image_height, image_width = image.shape[:2]
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"Invalid image dimensions: {image.shape}")

    aspect_ratio = image_width / image_height
    animal_width = ANIMAL_HEIGHT * aspect_ratio

    texture_id = p.loadTexture(str(image_path))
    if texture_id < 0:
        raise RuntimeError(f"PyBullet could not load texture: {image_path}")

    half_width = animal_width / 2.0
    half_height = ANIMAL_HEIGHT / 2.0
    #half_thickness = 0.001  # 板に見えるよう、X軸方向の厚みを極薄（1mm）にする
    half_thickness = 1

    # 変更点: 自作メッシュ(GEOM_MESH)の代わりに、
    # 自動で綺麗にテクスチャが貼られる極薄の直方体(GEOM_BOX)を使用する
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[half_thickness, half_width, half_height],
        rgbaColor=[1.0, 1.0, 1.0, 1.0],
    )
    if visual_shape_id < 0:
        raise RuntimeError("PyBullet could not create the animal visual shape.")

    # Important: baseCollisionShapeIndex=-1 means this body has no collision shape.
    # It is visible in the renderer, but it is not a physical obstacle and should
    # not be returned as a hit by rayTest/rayTestBatch.
    animal_body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=ANIMAL_POSITION,
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )
    if animal_body_id < 0:
        raise RuntimeError("PyBullet could not create the animal body.")

    p.changeVisualShape(
        objectUniqueId=animal_body_id,
        linkIndex=-1,
        textureUniqueId=texture_id,
        rgbaColor=[1.0, 1.0, 1.0, 1.0],
    )

    collision_data = p.getCollisionShapeData(animal_body_id, -1)
    if collision_data:
        raise RuntimeError(
            "The animal billboard unexpectedly has collision geometry."
        )

    print("Animal billboard created:")
    print(f"  image: {image_path}")
    print(f"  image size: {image_width} x {image_height}")
    print(f"  world position: {ANIMAL_POSITION}")
    print(f"  billboard size: {animal_width:.2f} x {ANIMAL_HEIGHT:.2f} m")
    print("  collision geometry: none (visual-only object)")

    return animal_body_id

def create_animal_billboard(image_path: Path):
    """Create a textured vertical plane facing -X with no collision geometry."""

    if not image_path.is_file():
        raise FileNotFoundError(
            "Animal image was not found.\n"
            f"Expected file: {image_path}\n"
            "Place a JPG or PNG image at assets/animals/deer.jpg, "
            "or change ANIMAL_IMAGE_PATH in this script."
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read the image: {image_path}")

    image_height, image_width = image.shape[:2]
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"Invalid image dimensions: {image.shape}")

    aspect_ratio = image_width / image_height
    animal_width = ANIMAL_HEIGHT * aspect_ratio

    texture_id = p.loadTexture(str(image_path))
    if texture_id < 0:
        raise RuntimeError(f"PyBullet could not load texture: {image_path}")

    half_width = animal_width / 2.0
    half_height = ANIMAL_HEIGHT / 2.0

    # 1. 正しいUV座標を持った1枚の板（Quad）のOBJデータをプログラム内で作成する
    # -X方向を向き、画像全体（U:0〜1, V:0〜1）が1面に完璧にマッピングされるように定義
    obj_content = f"""# Animal Billboard OBJ
        v 0.0 {+half_width} {-half_height}
        v 0.0 {-half_width} {-half_height}
        v 0.0 {-half_width} {+half_height}
        v 0.0 {+half_width} {+half_height}
        vt 0.0 0.0
        vt 1.0 0.0
        vt 1.0 1.0
        vt 0.0 1.0
        vn -1.0 0.0 0.0
        f 1/1/1 2/2/1 3/3/1
        f 1/1/1 3/3/1 4/4/1
        """
    # 一時的なOBJファイルをアセットフォルダ内に出力する
    obj_path = image_path.parent / "_temp_billboard.obj"
    obj_path.write_text(obj_content, encoding="utf-8")

    # 2. ファイル経由で GEOM_MESH としてビジュアル形状を作成する
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=str(obj_path),
        rgbaColor=[1.0, 1.0, 1.0, 1.0],
    )
    if visual_shape_id < 0:
        raise RuntimeError("PyBullet could not create the animal visual mesh from OBJ.")

    # 3. 物理的な当たり判定（コリジョン）なしのマルチボディを作成
    animal_body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=ANIMAL_POSITION,
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )
    if animal_body_id < 0:
        raise RuntimeError("PyBullet could not create the animal body.")

    # 4. テクスチャを適用（OBJ内のUVマッピングに従い、1面に画像全体が綺麗に貼られる）
    p.changeVisualShape(
        objectUniqueId=animal_body_id,
        linkIndex=-1,
        textureUniqueId=texture_id,
        rgbaColor=[1.0, 1.0, 1.0, 1.0],
    )

    collision_data = p.getCollisionShapeData(animal_body_id, -1)
    if collision_data:
        raise RuntimeError(
            "The animal billboard unexpectedly has collision geometry."
        )

    print("Animal billboard created:")
    print(f"  image: {image_path}")
    print(f"  image size: {image_width} x {image_height}")
    print(f"  world position: {ANIMAL_POSITION}")
    print(f"  billboard size: {animal_width:.2f} x {ANIMAL_HEIGHT:.2f} m")
    print("  collision geometry: none (visual-only object)")

    return animal_body_id

# ============================================================
# Virtual camera
# ============================================================

def get_front_camera_view_matrix(robot_id):
    """Build a front-facing camera view matrix from the current Husky pose."""

    base_position, base_orientation = p.getBasePositionAndOrientation(robot_id)

    local_eye = CAMERA_LOCAL_POSITION
    local_target = [
        local_eye[0] + CAMERA_LOOK_AHEAD,
        local_eye[1],
        local_eye[2],
    ]
    local_up_point = [
        local_eye[0],
        local_eye[1],
        local_eye[2] + 1.0,
    ]

    eye = p.multiplyTransforms(
        base_position, base_orientation,
        local_eye, [0.0, 0.0, 0.0, 1.0],
    )[0]
    target = p.multiplyTransforms(
        base_position, base_orientation,
        local_target, [0.0, 0.0, 0.0, 1.0],
    )[0]
    up_point = p.multiplyTransforms(
        base_position, base_orientation,
        local_up_point, [0.0, 0.0, 0.0, 1.0],
    )[0]
    up_vector = [up_point[i] - eye[i] for i in range(3)]

    return p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=up_vector,
    )


def render_front_camera(robot_id, projection_matrix):
    """Render the virtual camera image and return a BGR image for OpenCV."""

    view_matrix = get_front_camera_view_matrix(robot_id)

    result = p.getCameraImage(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    rgba = np.asarray(result[2], dtype=np.uint8).reshape(
        CAMERA_HEIGHT, CAMERA_WIDTH, 4
    )
    rgb = rgba[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ============================================================
# Main
# ============================================================

def main():
    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Could not connect to the PyBullet GUI.")

    camera_window_created = False

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(TIME_STEP)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(numSolverIterations=100)

        # Ground
        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(plane_id, -1, lateralFriction=1.0)

        # Husky
        robot_id = p.loadURDF(
            "husky/husky.urdf",
            basePosition=[0.0, 0.0, 0.1],
            useFixedBase=False,
        )
        wheel_joints = find_wheel_joints(robot_id)

        for joint_id in wheel_joints.values():
            p.changeDynamics(
                robot_id,
                joint_id,
                lateralFriction=1.0,
                rollingFriction=0.0,
            )

        # Add the animal image plane in front of the robot.
        animal_body_id = create_animal_billboard(ANIMAL_IMAGE_PATH)
        print(f"  animal body ID: {animal_body_id}")

        # PyBullet GUI camera (the observer view, not the robot camera)
        p.resetDebugVisualizerCamera(
            cameraDistance=4.5,
            cameraYaw=-45.0,
            cameraPitch=-25.0,
            cameraTargetPosition=[1.3, 0.0, 0.6],
        )

        # Robot-mounted virtual camera projection
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=CAMERA_FOV,
            aspect=CAMERA_WIDTH / CAMERA_HEIGHT,
            nearVal=CAMERA_NEAR,
            farVal=CAMERA_FAR,
        )

        cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAMERA_WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)
        camera_window_created = True

        print("\nSTEP2: Animal billboard and virtual camera preview")
        print("  W     : Forward")
        print("  S     : Backward")
        print("  A     : Turn left")
        print("  D     : Turn right")
        print("  SPACE : Stop")
        print("  Q     : Quit")
        print("\nClick the PyBullet window before using the keyboard.")

        frame_count = 0

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

            if key_events.get(ord(" "), 0) & p.KEY_IS_DOWN:
                linear_velocity = 0.0
                angular_velocity = 0.0

            set_drive_command(
                robot_id,
                wheel_joints,
                linear_velocity,
                angular_velocity,
            )

            p.stepSimulation()

            if frame_count % CAMERA_UPDATE_INTERVAL == 0:
                # Keep the observer camera centered on the robot.
                robot_position, _ = p.getBasePositionAndOrientation(robot_id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=4.5,
                    cameraYaw=-45.0,
                    cameraPitch=-25.0,
                    cameraTargetPosition=[
                        robot_position[0] + 1.3,
                        robot_position[1],
                        0.6,
                    ],
                )

                camera_bgr = render_front_camera(robot_id, projection_matrix)
                cv2.putText(
                    camera_bgr,
                    "STEP2: virtual front camera",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(CAMERA_WINDOW_NAME, camera_bgr)

            # Process OpenCV window events. Keep PyBullet focused for keyboard driving.
            cv2.waitKey(1)

            frame_count += 1
            time.sleep(TIME_STEP)

    finally:
        if camera_window_created:
            cv2.destroyAllWindows()
        if p.isConnected(client_id):
            p.disconnect(client_id)


if __name__ == "__main__":
    main()
