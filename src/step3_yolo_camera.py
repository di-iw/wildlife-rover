"""STEP3: Run YOLO26n on the Husky's PyBullet front-camera image.

Run from the project root:
    python src/step3_yolo_camera.py

The script reuses the STEP1 driving controls and STEP2 virtual animal billboard.
A single .pt checkpoint is loaded with Ultralytics. Class names are read from
checkpoint metadata unless CLASS_NAMES_OVERRIDE is explicitly configured.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import torch
from ultralytics import YOLO

from step1_flat_husky import (
    TIME_STEP,
    find_wheel_joints,
    key_is_down,
    set_drive_command,
)


# ============================================================
# Project paths and model configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = PROJECT_ROOT / "models"

ANIMAL_IMAGE_PATH = PROJECT_ROOT / "assets" / "animals" / "bear1.jpg"

ANIMAL_IMAGE_PATHS = [
    PROJECT_ROOT / "assets" / "animals" / "bear1.jpg",
    PROJECT_ROOT / "assets" / "animals" / "bear2.jpg", # 例
]
ANIMAL_POSITION = [3.0, 0.0, 0.95]
ANIMAL_POSITIONS = [
    [3.0, 0.0, 0.95],              # billboard center [m]
    [7.0, 2.5, 0.95],
] 
ANIMAL_HEIGHT = 0.95               # billboard width follows image aspect ratio
ANIMAL_FACINGS = [
    "-X",  # 1つ目は Y-Z平面で -X 方向を向く
    "-Y",  # 2つ目は X-Z平面で Y 方向を向く
]


# Recommended: leave as None when exactly one .pt file exists in models/.
# If there are multiple checkpoints, specify the intended file, for example:
# MODEL_PATH = Path("models/my_yolo26n_14cls_best.pt")
# Absolute paths are also accepted.
#MODEL_PATH: Path | None = Path("models\yolo26n_14cls_v7_best.pt")
MODEL_PATH: Path | None = None

# None means: use the class names stored in the selected checkpoint.
# This is the recommended setting, including when switching between 14-class
# and COCO 80-class checkpoints.
#
# Only set this dictionary when you intentionally need to override display
# labels. Class IDs MUST match the checkpoint's actual class ordering.
# The default None is safest: it automatically uses model.names from the .pt.
CLASS_NAMES_OVERRIDE: dict[int, str] | None = None
# Example override for the 14-class checkpoint described in this project:
# CLASS_NAMES_OVERRIDE = {
#     0: "person",
#     1: "bird",
#     2: "cat",
#     3: "dog",
#     4: "horse",
#     5: "sheep",
#     6: "cow",
#     7: "elephant",
#     8: "bear",
#     9: "zebra",
#     10: "giraffe",
#     11: "deer",
#     12: "fox",
#     13: "hare",
# }

# None means infer every class in the selected model.
# Example: TARGET_CLASS_IDS = {8, 11, 12, 13}
# Only use IDs that exist in the selected checkpoint.
TARGET_CLASS_IDS: set[int] | None = {8}
COCO_TARGET_CLASS_IDS = {21}
# CPU inference settings
YOLO_IMAGE_SIZE = 320
INFERENCE_DEVICE = "cpu"
INFERENCE_FPS = 2.0
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100
TORCH_NUM_THREADS = 4


# ============================================================
# Simulation and robot settings (retained from STEP2)
# ============================================================

LINEAR_SPEED = 0.4       # m/s
ANGULAR_SPEED = 0.8      # rad/s

# Husky-front camera, expressed in the Husky base frame.
CAMERA_LOCAL_POSITION = [0.42, 0.0, 0.72]
CAMERA_LOOK_AHEAD = 2.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOV = 60.0
CAMERA_NEAR = 0.05
CAMERA_FAR = 20.0
CAMERA_UPDATE_INTERVAL = 24  # 240 Hz physics / 24 = 10 Hz camera preview
CAMERA_WINDOW_NAME = "STEP3 - Husky Camera + YOLO26n"


# ============================================================
# Model loading and class metadata
# ============================================================

def resolve_model_path() -> Path:
    """Resolve a .pt path without silently choosing among multiple models."""

    if MODEL_PATH is not None:
        path = Path(MODEL_PATH).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(f"YOLO model file was not found: {path}")
        if path.suffix.lower() != ".pt":
            raise ValueError(
                f"This STEP3 implementation expects a .pt checkpoint: {path}"
            )
        return path

    if MODEL_DIRECTORY.is_dir():
        candidates = sorted(
            path.resolve()
            for path in MODEL_DIRECTORY.glob("*.pt")
            if path.is_file()
        )

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            candidate_list = "\n".join(f"  - {path}" for path in candidates)
            raise RuntimeError(
                "Multiple .pt checkpoints were found. Do not select one implicitly.\n"
                f"{candidate_list}\n"
                "Set MODEL_PATH explicitly near the top of this script."
            )

    # 有効なローカルファイルが見つからない場合のフォールバック
    print("No valid local YOLO checkpoint found. Falling back to 'yolo26n.pt' (auto-download).")
    global TARGET_CLASS_IDS, COCO_TARGET_CLASS_IDS
    TARGET_CLASS_IDS = COCO_TARGET_CLASS_IDS
    return "yolo26n.pt"


def normalize_model_names(raw_names: Any) -> dict[int, str]:
    """Normalize Ultralytics model.names (dict or sequence) to {class_id: name}."""

    if isinstance(raw_names, dict):
        names = {int(class_id): str(name) for class_id, name in raw_names.items()}
    elif isinstance(raw_names, (list, tuple)):
        names = {class_id: str(name) for class_id, name in enumerate(raw_names)}
    else:
        raise TypeError(
            "Unsupported model.names format. Expected dict, list, or tuple; "
            f"got {type(raw_names).__name__}."
        )

    if not names:
        raise ValueError("The selected checkpoint contains no class-name metadata.")

    expected_ids = set(range(len(names)))
    if set(names) != expected_ids:
        raise ValueError(
            "Model class IDs are not contiguous from 0. "
            f"Found IDs: {sorted(names)}"
        )

    return names


def load_detector() -> tuple[YOLO, dict[int, str], Path]:
    """Load the selected .pt model on CPU and obtain its actual class names."""

    model_path = resolve_model_path()

    # Limit PyTorch CPU thread usage on the laptop.
    torch.set_num_threads(TORCH_NUM_THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch may reject this if inter-op work was already initialized.
        pass

    print(f"Loading YOLO checkpoint: {model_path}")
    model = YOLO(str(model_path))
    model_names = normalize_model_names(model.names)

    class_names = dict(model_names)
    if CLASS_NAMES_OVERRIDE is not None:
        unknown_ids = set(CLASS_NAMES_OVERRIDE) - set(model_names)
        if unknown_ids:
            raise ValueError(
                "CLASS_NAMES_OVERRIDE contains IDs absent from the checkpoint: "
                f"{sorted(unknown_ids)}"
            )
        class_names.update(
            {int(class_id): str(name) for class_id, name in CLASS_NAMES_OVERRIDE.items()}
        )

    if TARGET_CLASS_IDS is not None:
        unknown_ids = set(TARGET_CLASS_IDS) - set(model_names)
        if unknown_ids:
            raise ValueError(
                "TARGET_CLASS_IDS contains IDs absent from the checkpoint: "
                f"{sorted(unknown_ids)}"
            )

    print(f"Model class count: {len(model_names)}")
    print("Class names from checkpoint metadata:")
    for class_id in sorted(class_names):
        print(f"  {class_id}: {class_names[class_id]}")

    if TARGET_CLASS_IDS is None:
        print("Inference classes: all classes")
    else:
        print(f"Inference class IDs: {sorted(TARGET_CLASS_IDS)}")

    return model, class_names, model_path


# ============================================================
# Animal visual object
# ============================================================

def create_animal_billboard(image_path: Path) -> int:
    """Create a textured, vertical, visual-only OBJ billboard.

    A temporary OBJ is used because its explicit UV coordinates produced
    reliable texture mapping in the current STEP2 environment. The OBJ file
    is removed after PyBullet has created the visual shape; the loaded Bullet
    visual shape and texture remain available in the simulation.
    """

    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(
            "Animal image was not found.\n"
            f"Expected file: {image_path}\n"
            "Place the image at assets/animals/deer.jpg, "
            "or change ANIMAL_IMAGE_PATH in this script."
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read the image: {image_path}")

    image_height, image_width = image.shape[:2]
    if image_height <= 0 or image_width <= 0:
        raise ValueError(f"Invalid image dimensions: {image.shape}")

    animal_width = ANIMAL_HEIGHT * (image_width / image_height)
    half_width = animal_width / 2.0
    half_height = ANIMAL_HEIGHT / 2.0

    texture_id = p.loadTexture(str(image_path))
    if texture_id < 0:
        raise RuntimeError(f"PyBullet could not load texture: {image_path}")

    # The board lies in the Y-Z plane and its visible normal points toward -X.
    # OBJ UVs map the complete image onto this single face.
    obj_content = f"""# Temporary animal billboard
v 0.0 {+half_width:.8f} {-half_height:.8f}
v 0.0 {-half_width:.8f} {-half_height:.8f}
v 0.0 {-half_width:.8f} {+half_height:.8f}
v 0.0 {+half_width:.8f} {+half_height:.8f}
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vn -1.0 0.0 0.0
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
"""

    # Improvement over the STEP2 version that left _temp_billboard.obj in
    # assets/: use a unique temporary directory and avoid repository clutter.
    with tempfile.TemporaryDirectory(prefix="wildlife_rover_billboard_") as temp_dir:
        obj_path = Path(temp_dir) / "animal_billboard.obj"
        obj_path.write_text(obj_content, encoding="utf-8")

        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(obj_path),
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
        )

    if visual_shape_id < 0:
        raise RuntimeError(
            "PyBullet could not create the animal visual mesh from the temporary OBJ."
        )

    animal_body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=ANIMAL_POSITION,
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )
    if animal_body_id < 0:
        raise RuntimeError("PyBullet could not create the animal billboard body.")

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

def create_animal_billboards(
    image_paths: list[Path], 
    positions: list[list[float]], 
    facings: list[str] = None
) -> list[int]:
    """Create textured, vertical, visual-only OBJ billboards.
    
    facings can specify the orientation for each billboard (e.g., "-X" or "-Y").
    Defaults to "-X" for all if not provided.
    """
    body_ids = []

    # facingsが指定されていない場合はすべて "-X" にする
    if facings is None:
        facings = ["-X"] * len(image_paths)

    # image_path, position, facing を同時にループで処理
    for image_path, position, facing in zip(image_paths, positions, facings):
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(
                "Animal image was not found.\n"
                f"Expected file: {image_path}\n"
                "Place the image at assets/animals/deer.jpg, "
                "or change ANIMAL_IMAGE_PATH in this script."
            )

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"OpenCV could not read the image: {image_path}")

        image_height, image_width = image.shape[:2]
        if image_height <= 0 or image_width <= 0:
            raise ValueError(f"Invalid image dimensions: {image.shape}")

        animal_width = ANIMAL_HEIGHT * (image_width / image_height)
        half_width = animal_width / 2.0
        half_height = ANIMAL_HEIGHT / 2.0

        texture_id = p.loadTexture(str(image_path))
        if texture_id < 0:
            raise RuntimeError(f"PyBullet could not load texture: {image_path}")

        # 向き（facing）に応じて、変化する頂点と法線のみを変数に格納する
        if facing == "-X":
            # Y-Z平面（X=0固定、幅がY軸方向）
            v_coords = [
                f"0.0 {+half_width:.8f} {-half_height:.8f}",
                f"0.0 {-half_width:.8f} {-half_height:.8f}",
                f"0.0 {-half_width:.8f} {+half_height:.8f}",
                f"0.0 {+half_width:.8f} {+half_height:.8f}",
            ]
            vn = "-1.0 0.0 0.0"
        elif facing == "-Y":
            # X-Z平面（Y=0固定、幅がX軸方向）
            # 頂点の順番を反転させて、表裏の向きと左右の向きを正しく合わせる
            v_coords = [
                f"{-half_width:.8f} 0.0 {-half_height:.8f}",  # 左下
                f"{+half_width:.8f} 0.0 {-half_height:.8f}",  # 右下
                f"{+half_width:.8f} 0.0 {+half_height:.8f}",  # 右上
                f"{-half_width:.8f} 0.0 {+half_height:.8f}",  # 左上
            ]
            vn = "0.0 -1.0 0.0"
        else:
            raise ValueError(f"Unsupported facing direction: {facing}")

        # obj_content の生成は1箇所にまとめる
        obj_content = f"""# Temporary animal billboard ({facing})
v {v_coords[0]}
v {v_coords[1]}
v {v_coords[2]}
v {v_coords[3]}
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
vn {vn}
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
"""

        with tempfile.TemporaryDirectory(prefix="wildlife_rover_billboard_") as temp_dir:
            obj_path = Path(temp_dir) / "animal_billboard.obj"
            obj_path.write_text(obj_content, encoding="utf-8")

            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=str(obj_path),
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
            )

        if visual_shape_id < 0:
            raise RuntimeError(
                "PyBullet could not create the animal visual mesh from the temporary OBJ."
            )

        animal_body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
        )
        if animal_body_id < 0:
            raise RuntimeError("PyBullet could not create the animal billboard body.")

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

        print(f"Animal billboard created (facing {facing}):")
        print(f"  image: {image_path}")
        print(f"  world position: {position}")
        print(f"  billboard size: {animal_width:.2f} x {ANIMAL_HEIGHT:.2f} m")

        body_ids.append(animal_body_id)

    return body_ids

# ============================================================
# Husky front camera
# ============================================================

def get_front_camera_view_matrix(robot_id: int) -> list[float]:
    """Build a camera view matrix from the current Husky pose."""

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

    identity_orientation = [0.0, 0.0, 0.0, 1.0]

    eye = p.multiplyTransforms(
        base_position, base_orientation,
        local_eye, identity_orientation,
    )[0]
    target = p.multiplyTransforms(
        base_position, base_orientation,
        local_target, identity_orientation,
    )[0]
    up_point = p.multiplyTransforms(
        base_position, base_orientation,
        local_up_point, identity_orientation,
    )[0]
    up_vector = [up_point[i] - eye[i] for i in range(3)]

    return p.computeViewMatrix(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraUpVector=up_vector,
    )


def render_front_camera(robot_id: int, projection_matrix: list[float]) -> np.ndarray:
    """Capture the virtual front camera and return a BGR uint8 image."""

    result = p.getCameraImage(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        viewMatrix=get_front_camera_view_matrix(robot_id),
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    rgba = np.asarray(result[2], dtype=np.uint8).reshape(
        CAMERA_HEIGHT, CAMERA_WIDTH, 4
    )
    return cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)


# ============================================================
# Inference and visualization
# ============================================================

def run_inference(
    model: YOLO,
    frame_bgr: np.ndarray,
) -> tuple[list[dict[str, Any]], float]:
    """Run one CPU inference and return boxes in original camera-image pixels."""

    start_time = time.perf_counter()

    results = model.predict(
        source=frame_bgr,
        imgsz=YOLO_IMAGE_SIZE,
        rect=False,  # force a 320 x 320 letterboxed model input
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETECTIONS,
        classes=(sorted(TARGET_CLASS_IDS) if TARGET_CLASS_IDS is not None else None),
        device=INFERENCE_DEVICE,
        verbose=False,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    detections: list[dict[str, Any]] = []
    if not results:
        return detections, elapsed_ms

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections, elapsed_ms

    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)

    for box, confidence, class_id in zip(xyxy, confidences, class_ids):
        detections.append(
            {
                "xyxy": box.astype(float).tolist(),
                "confidence": float(confidence),
                "class_id": int(class_id),
            }
        )

    return detections, elapsed_ms


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    class_names: dict[int, str],
) -> np.ndarray:
    """Draw bounding boxes, class labels, and confidence on a BGR frame."""

    output = frame_bgr.copy()
    image_height, image_width = output.shape[:2]

    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"]
        class_id = detection["class_id"]
        confidence = detection["confidence"]
        class_name = class_names.get(class_id, f"class_{class_id}")

        x1 = max(0, min(image_width - 1, int(round(x1))))
        y1 = max(0, min(image_height - 1, int(round(y1))))
        x2 = max(0, min(image_width - 1, int(round(x2))))
        y2 = max(0, min(image_height - 1, int(round(y2))))

        if x2 <= x1 or y2 <= y1:
            continue

        box_color = (40, 220, 40)  # BGR: green
        text = f"{class_name} {confidence:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        label_y = max(text_height + baseline + 2, y1)
        cv2.rectangle(
            output,
            (x1, label_y - text_height - baseline - 4),
            (min(image_width - 1, x1 + text_width + 6), label_y),
            box_color,
            thickness=-1,
        )
        cv2.putText(
            output,
            text,
            (x1 + 3, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def draw_status(
    frame_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    inference_ms: float | None,
    inference_age_sec: float | None,
) -> np.ndarray:
    """Add compact inference status information to the camera preview."""

    output = frame_bgr.copy()
    cv2.rectangle(output, (0, 0), (CAMERA_WIDTH, 58), (25, 25, 25), -1)

    line1 = (
        f"YOLO26n | CPU | imgsz={YOLO_IMAGE_SIZE} | "
        f"target={INFERENCE_FPS:.1f} FPS"
    )
    if inference_ms is None:
        line2 = "Inference: waiting for first frame..."
    else:
        age = inference_age_sec if inference_age_sec is not None else 0.0
        line2 = (
            f"Detections: {len(detections)} | "
            f"Latency: {inference_ms:.0f} ms | Last inference: {age:.1f} s ago"
        )

    cv2.putText(
        output, line1, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        output, line2, (10, 47), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return output


# ============================================================
# Main
# ============================================================

def main() -> None:
    model, class_names, model_path = load_detector()
    print(f"Selected checkpoint: {model_path}")
    print(
        f"Inference settings: CPU, {YOLO_IMAGE_SIZE}x{YOLO_IMAGE_SIZE}, "
        f"target {INFERENCE_FPS:.1f} FPS, conf={CONF_THRESHOLD}"
    )

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

        # Visual-only animal billboard (no collision geometry)
        #animal_body_id = create_animal_billboard(ANIMAL_IMAGE_PATH)
        animal_body_ids = create_animal_billboards(
                            ANIMAL_IMAGE_PATHS,
                            ANIMAL_POSITIONS,
                            ANIMAL_FACINGS
                        )
        print(f"Animal body ID: {animal_body_ids}")

        # PyBullet observer camera
        p.resetDebugVisualizerCamera(
            cameraDistance=4.5,
            cameraYaw=45.0,
            cameraPitch=-25.0,
            cameraTargetPosition=[1.3, 0.0, 0.6],
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=CAMERA_FOV,
            aspect=CAMERA_WIDTH / CAMERA_HEIGHT,
            nearVal=CAMERA_NEAR,
            farVal=CAMERA_FAR,
        )

        cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAMERA_WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)
        camera_window_created = True

        print("\nSTEP3: Husky front camera + YOLO26n inference")
        print("  W     : Forward")
        print("  S     : Backward")
        print("  A     : Turn left")
        print("  D     : Turn right")
        print("  SPACE : Stop")
        print("  Q     : Quit")
        print("Click the PyBullet window before using the keyboard.\n")

        frame_count = 0
        last_detections: list[dict[str, Any]] = []
        last_inference_ms: float | None = None
        last_inference_finished: float | None = None
        next_inference_time = 0.0
        inference_interval = 1.0 / INFERENCE_FPS

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
                robot_position, _ = p.getBasePositionAndOrientation(robot_id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=4.5,
                    cameraYaw=45.0,
                    cameraPitch=-25.0,
                    cameraTargetPosition=[
                        robot_position[0] + 1.3,
                        robot_position[1],
                        0.6,
                    ],
                )

                camera_bgr = render_front_camera(robot_id, projection_matrix)
                now = time.perf_counter()

                # Render preview at about 10 Hz, but run YOLO at 1-2 Hz.
                if now >= next_inference_time:
                    last_detections, last_inference_ms = run_inference(
                        model,
                        camera_bgr,
                    )
                    last_inference_finished = time.perf_counter()
                    next_inference_time = last_inference_finished + inference_interval

                    if last_detections:
                        summary_items = []
                        for detection in last_detections:
                            class_id = detection["class_id"]
                            class_name = class_names.get(
                                class_id, f"class_{class_id}"
                            )
                            summary_items.append(
                                f"{class_name} ({detection['confidence']:.2f})"
                            )
                        print("Detected: " + ", ".join(summary_items))

                camera_display = draw_detections(
                    camera_bgr,
                    last_detections,
                    class_names,
                )

                inference_age = (
                    None
                    if last_inference_finished is None
                    else time.perf_counter() - last_inference_finished
                )
                camera_display = draw_status(
                    camera_display,
                    last_detections,
                    last_inference_ms,
                    inference_age,
                )
                cv2.imshow(CAMERA_WINDOW_NAME, camera_display)

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
