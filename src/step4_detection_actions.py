"""STEP4: Add detection-triggered stop, event recording, notification, and observation.

Run from the project root:
    python src/step4_detection_actions.py

This script reuses the STEP3 camera, YOLO, and animal-billboard implementation.
PATROL is keyboard-driven in STEP4 so the detection event can be tested before
autonomous waypoint patrol is implemented. AVOID is reserved for STEP6.
"""

from __future__ import annotations

import math
import csv
import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import pybullet as p
import pybullet_data

from step1_flat_husky import (
    TIME_STEP,
    find_wheel_joints,
    key_is_down,
    set_drive_command,
)
from step3_yolo_camera import (
    PROJECT_ROOT,
    ANIMAL_IMAGE_PATH,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FOV,
    CAMERA_NEAR,
    CAMERA_FAR,
    CAMERA_UPDATE_INTERVAL,
    LINEAR_SPEED,
    ANGULAR_SPEED,
    INFERENCE_FPS,
    YOLO_IMAGE_SIZE,
    INFERENCE_DEVICE,
    CONF_THRESHOLD,
    load_detector,
    create_animal_billboard,
    render_front_camera,
    run_inference,
    draw_detections,
)


# ============================================================
# STEP4 behavior settings
# ============================================================

class RoverState(str, Enum):
    PATROL = "PATROL"
    AVOID = "AVOID"       # Reserved for STEP6 obstacle avoidance.
    OBSERVE = "OBSERVE"


INITIAL_STATE = RoverState.PATROL

# A detection event is accepted only after this many consecutive
# inference results contain at least one eligible detection.
DETECTION_CONFIRM_INFERENCES = 2

# After an event, the detector is re-armed only after this many consecutive
# inference results contain no eligible detections while in PATROL.
REARM_NO_DETECTION_INFERENCES = 3

# How long the rover remains stopped after recording an animal event.
OBSERVE_DURATION_SEC = 3.0

# Default for this project's 14-class model: animal classes 1-13.
# Class 0 is "person" and therefore does not trigger a wildlife event.
# When switching to another checkpoint (e.g. COCO80), update this set using
# that checkpoint's actual class IDs. Set None to let every model class trigger.
EVENT_CLASS_IDS: set[int] | None = set(range(1, 14))

# Detections used for event activation and logging must satisfy this threshold.
# Defaults to the inference confidence threshold configured in STEP3.
EVENT_CONF_THRESHOLD = CONF_THRESHOLD

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "detections"
EVENT_LOG_PATH = OUTPUT_DIRECTORY / "events.csv"

CAMERA_WINDOW_NAME = "STEP4 - Wildlife Rover"
OBSERVER_CAMERA_DISTANCE = 4.5
OBSERVER_CAMERA_YAW = 45.0
OBSERVER_CAMERA_PITCH = -25.0


# ============================================================
# Event filtering, saving, and display
# ============================================================

def select_event_detections(
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return detections eligible to trigger/log a wildlife event."""

    selected = []

    for detection in detections:
        class_id = int(detection["class_id"])
        confidence = float(detection["confidence"])

        if EVENT_CLASS_IDS is not None and class_id not in EVENT_CLASS_IDS:
            continue
        if confidence < EVENT_CONF_THRESHOLD:
            continue

        selected.append(detection)

    return selected


# def save_detection_event(
#     frame_bgr,
#     detections: list[dict[str, Any]],
#     class_names: dict[int, str],
#     simulation_time_sec: float,
# ) -> dict[str, str]:
#     """Save one annotated image and append one event record to CSV."""

#     OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

#     timestamp = datetime.now().astimezone()
#     event_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")
#     image_filename = f"event_{event_id}.jpg"
#     image_path = OUTPUT_DIRECTORY / image_filename

#     annotated = draw_detections(frame_bgr, detections, class_names)

#     primary = max(detections, key=lambda item: float(item["confidence"]))
#     primary_class_id = int(primary["class_id"])
#     primary_class_name = class_names.get(
#         primary_class_id, f"class_{primary_class_id}"
#     )
#     primary_confidence = float(primary["confidence"])

#     # Mark the saved image as an event snapshot.
#     cv2.rectangle(
#         annotated,
#         (0, 0),
#         (CAMERA_WIDTH, 34),
#         (20, 20, 20),
#         thickness=-1,
#     )
#     cv2.putText(
#         annotated,
#         f"ANIMAL EVENT: {primary_class_name} {primary_confidence:.2f}",
#         (10, 23),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.58,
#         (0, 255, 255),
#         2,
#         cv2.LINE_AA,
#     )

#     if not cv2.imwrite(str(image_path), annotated):
#         raise OSError(f"Could not write event image: {image_path}")

#     event_detections = []
#     for detection in detections:
#         class_id = int(detection["class_id"])
#         event_detections.append(
#             {
#                 "class_id": class_id,
#                 "class_name": class_names.get(class_id, f"class_{class_id}"),
#                 "confidence": float(detection["confidence"]),
#                 "xyxy": [float(value) for value in detection["xyxy"]],
#             }
#         )

#     write_header = not EVENT_LOG_PATH.is_file() or EVENT_LOG_PATH.stat().st_size == 0

#     row = {
#         "event_id": event_id,
#         "timestamp_local": timestamp.isoformat(timespec="milliseconds"),
#         "simulation_time_sec": f"{simulation_time_sec:.3f}",
#         "primary_class_id": str(primary_class_id),
#         "primary_class_name": primary_class_name,
#         "primary_confidence": f"{primary_confidence:.5f}",
#         "detection_count": str(len(event_detections)),
#         "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
#         "detections_json": json.dumps(event_detections, ensure_ascii=False),
#     }

#     fieldnames = [
#         "event_id",
#         "timestamp_local",
#         "simulation_time_sec",
#         "primary_class_id",
#         "primary_class_name",
#         "primary_confidence",
#         "detection_count",
#         "image_path",
#         "detections_json",
#     ]

#     with EVENT_LOG_PATH.open(
#         "a",
#         newline="",
#         # Write a BOM only when creating the file. Re-opening an existing
#         # file with utf-8-sig in append mode would otherwise insert another BOM.
#         encoding="utf-8-sig" if write_header else "utf-8",
#     ) as csv_file:
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         if write_header:
#             writer.writeheader()
#         writer.writerow(row)

#     return {
#         "event_id": event_id,
#         "primary_class_name": primary_class_name,
#         "primary_confidence": f"{primary_confidence:.2f}",
#         "image_path": str(image_path),
#     }

# def save_detection_event(
#     frame_bgr,
#     detections: list[dict[str, Any]],
#     class_names: dict[int, str],
#     simulation_time_sec: float,
#     robot_id: int,  # 追加: ロボットの位置を取得するため
# ) -> dict[str, str]:
#     """Save one annotated image and append one event record to CSV."""

#     OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

#     timestamp = datetime.now().astimezone()
#     event_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")
#     image_filename = f"event_{event_id}.jpg"
#     image_path = OUTPUT_DIRECTORY / image_filename

#     annotated = draw_detections(frame_bgr, detections, class_names)

#     # 検出時のロボットの座標を取得
#     robot_pos, _ = p.getBasePositionAndOrientation(robot_id)
#     robot_x, robot_y, robot_z = robot_pos

#     primary = max(detections, key=lambda item: float(item["confidence"]))
#     primary_class_id = int(primary["class_id"])
#     primary_class_name = class_names.get(
#         primary_class_id, f"class_{primary_class_id}"
#     )
#     primary_confidence = float(primary["confidence"])

#     # (中略: 画像へのテキスト描画やファイルの書き込み処理はそのまま)
#     if not cv2.imwrite(str(image_path), annotated):
#         raise OSError(f"Could not write event image: {image_path}")

#     event_detections = []
#     for detection in detections:
#         class_id = int(detection["class_id"])
#         event_detections.append(
#             {
#                 "class_id": class_id,
#                 "class_name": class_names.get(class_id, f"class_{class_id}"),
#                 "confidence": float(detection["confidence"]),
#                 "xyxy": [float(value) for value in detection["xyxy"]],
#             }
#         )

#     write_header = not EVENT_LOG_PATH.is_file() or EVENT_LOG_PATH.stat().st_size == 0

#     # rowにロボットの座標を追加
#     row = {
#         "event_id": event_id,
#         "timestamp_local": timestamp.isoformat(timespec="milliseconds"),
#         "simulation_time_sec": f"{simulation_time_sec:.3f}",
#         "robot_x": f"{robot_x:.3f}",
#         "robot_y": f"{robot_y:.3f}",
#         "robot_z": f"{robot_z:.3f}",
#         "primary_class_id": str(primary_class_id),
#         "primary_class_name": primary_class_name,
#         "primary_confidence": f"{primary_confidence:.5f}",
#         "detection_count": str(len(event_detections)),
#         "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
#         "detections_json": json.dumps(event_detections, ensure_ascii=False),
#     }

#     # fieldnamesにもロボットの座標項目を追加
#     fieldnames = [
#         "event_id",
#         "timestamp_local",
#         "simulation_time_sec",
#         "robot_x",
#         "robot_y",
#         "robot_z",
#         "primary_class_id",
#         "primary_class_name",
#         "primary_confidence",
#         "detection_count",
#         "image_path",
#         "detections_json",
#     ]

#     with EVENT_LOG_PATH.open(
#         "a",
#         newline="",
#         encoding="utf-8-sig" if write_header else "utf-8",
#     ) as csv_file:
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         if write_header:
#             writer.writeheader()
#         writer.writerow(row)

#     return {
#         "event_id": event_id,
#         "primary_class_name": primary_class_name,
#         "primary_confidence": f"{primary_confidence:.2f}",
#         "image_path": str(image_path),
#     }


def save_detection_event(
    frame_bgr,
    detections: list[dict[str, Any]],
    class_names: dict[int, str],
    simulation_time_sec: float,
    robot_id: int,  # ロボットの位置と姿勢を取得するため[cite: 1]
) -> dict[str, str]:
    """Save one annotated image and append one event record to CSV."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone()
    event_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")
    image_filename = f"event_{event_id}.jpg"
    image_path = OUTPUT_DIRECTORY / image_filename

    annotated = draw_detections(frame_bgr, detections, class_names)

    # 検出時のロボットの座標と姿勢（クォータニオン）を取得[cite: 1]
    robot_pos, robot_orn = p.getBasePositionAndOrientation(robot_id)
    robot_x, robot_y, robot_z = robot_pos
    
    # クォータニオンからオイラー角を算出し、水平方向の向き（Yaw角）を取得[cite: 1]
    _, _, robot_yaw = p.getEulerFromQuaternion(robot_orn)

    primary = max(detections, key=lambda item: float(item["confidence"]))
    primary_class_id = int(primary["class_id"])
    primary_class_name = class_names.get(
        primary_class_id, f"class_{primary_class_id}"
    )
    primary_confidence = float(primary["confidence"])

    if not cv2.imwrite(str(image_path), annotated):
        raise OSError(f"Could not write event image: {image_path}")

    event_detections = []
    for detection in detections:
        class_id = int(detection["class_id"])
        event_detections.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, f"class_{class_id}"),
                "confidence": float(detection["confidence"]),
                "xyxy": [float(value) for value in detection["xyxy"]],
            }
        )

    write_header = not EVENT_LOG_PATH.is_file() or EVENT_LOG_PATH.stat().st_size == 0

    # rowにロボットの向き（robot_yaw_deg）を追加[cite: 1]
    row = {
        "event_id": event_id,
        "timestamp_local": timestamp.isoformat(timespec="milliseconds"),
        "simulation_time_sec": f"{simulation_time_sec:.3f}",
        "robot_x": f"{robot_x:.3f}",
        "robot_y": f"{robot_y:.3f}",
        "robot_z": f"{robot_z:.3f}",
        "robot_yaw_deg": f"{math.degrees(robot_yaw):.2f}",
        "primary_class_id": str(primary_class_id),
        "primary_class_name": primary_class_name,
        "primary_confidence": f"{primary_confidence:.5f}",
        "detection_count": str(len(event_detections)),
        "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
        "detections_json": json.dumps(event_detections, ensure_ascii=False),
    }

    # fieldnamesにもロボットの向きの項目を追加[cite: 1]
    fieldnames = [
        "event_id",
        "timestamp_local",
        "simulation_time_sec",
        "robot_x",
        "robot_y",
        "robot_z",
        "robot_yaw_deg",
        "primary_class_id",
        "primary_class_name",
        "primary_confidence",
        "detection_count",
        "image_path",
        "detections_json",
    ]

    with EVENT_LOG_PATH.open(
        "a",
        newline="",
        encoding="utf-8-sig" if write_header else "utf-8",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return {
        "event_id": event_id,
        "primary_class_name": primary_class_name,
        "primary_confidence": f"{primary_confidence:.2f}",
        "image_path": str(image_path),
    }

def draw_step4_status(
    frame_bgr,
    detections: list[dict[str, Any]],
    class_names: dict[int, str],
    state: RoverState,
    inference_ms: float | None,
    inference_age_sec: float | None,
    observe_remaining_sec: float,
    last_event_message: str,
    event_latched: bool,
):
    """Draw detections and the current state/event status on the camera image."""

    output = draw_detections(frame_bgr, detections, class_names)

    state_colors = {
        RoverState.PATROL: (40, 180, 40),       # BGR green
        RoverState.AVOID: (0, 165, 255),        # BGR orange
        RoverState.OBSERVE: (0, 0, 230),        # BGR red
    }
    state_color = state_colors[state]

    cv2.rectangle(
        output,
        (0, 0),
        (CAMERA_WIDTH, 91),
        (24, 24, 24),
        thickness=-1,
    )

    if state == RoverState.OBSERVE:
        state_text = f"STATE: OBSERVE ({observe_remaining_sec:.1f}s)"
    elif state == RoverState.PATROL:
        state_text = "STATE: PATROL (keyboard-driven test mode)"
    else:
        state_text = "STATE: AVOID"

    cv2.putText(
        output,
        state_text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        state_color,
        2,
        cv2.LINE_AA,
    )

    if inference_ms is None:
        inference_text = "YOLO: waiting for first inference"
    else:
        age = inference_age_sec if inference_age_sec is not None else 0.0
        inference_text = (
            f"Detections: {len(detections)} | "
            f"Latency: {inference_ms:.0f} ms | "
            f"Result age: {age:.1f}s"
        )

    cv2.putText(
        output,
        inference_text,
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    latch_text = "EVENT LATCHED" if event_latched else "EVENT READY"
    event_text = f"{latch_text} | Last: {last_event_message}"
    cv2.putText(
        output,
        event_text[:105],
        (10, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 220, 255) if event_latched else (190, 190, 190),
        1,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# Main
# ============================================================

def main() -> None:
    model, class_names, model_path = load_detector()

    if EVENT_CLASS_IDS is not None:
        unknown_event_ids = EVENT_CLASS_IDS - set(class_names)
        if unknown_event_ids:
            raise ValueError(
                "EVENT_CLASS_IDS contains IDs absent from the loaded model: "
                f"{sorted(unknown_event_ids)}"
            )

    print(f"Selected checkpoint: {model_path}")
    print(
        f"Inference: {INFERENCE_DEVICE}, {YOLO_IMAGE_SIZE}x{YOLO_IMAGE_SIZE}, "
        f"target {INFERENCE_FPS:.1f} FPS, "
        f"conf={CONF_THRESHOLD}"
    )
    print(f"Event confidence threshold: {EVENT_CONF_THRESHOLD}")
    print(
        "Event classes: all model classes"
        if EVENT_CLASS_IDS is None
        else f"Event class IDs: {sorted(EVENT_CLASS_IDS)}"
    )

    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Could not connect to the PyBullet GUI.")

    camera_window_created = False

    try:
        # ----------------------------------------------------
        # Initialize simulation (same base setup as STEP3)
        # ----------------------------------------------------
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(TIME_STEP)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(numSolverIterations=100)

        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(plane_id, -1, lateralFriction=1.0)

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

        animal_body_id = create_animal_billboard(ANIMAL_IMAGE_PATH)
        print(f"Animal body ID: {animal_body_id}")

        p.resetDebugVisualizerCamera(
            cameraDistance=OBSERVER_CAMERA_DISTANCE,
            cameraYaw=OBSERVER_CAMERA_YAW,
            cameraPitch=OBSERVER_CAMERA_PITCH,
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

        print("\nSTEP4: detection-triggered rover actions")
        print("  W     : Forward while in PATROL")
        print("  S     : Backward while in PATROL")
        print("  A     : Turn left while in PATROL")
        print("  D     : Turn right while in PATROL")
        print("  SPACE : Stop")
        print("  Q     : Quit")
        print("\nPATROL is keyboard-driven in STEP4.")
        print("A detection event stops the rover, records one event,")
        print(f"observes for {OBSERVE_DURATION_SEC:.1f} seconds, then returns to PATROL.")
        print("Click the PyBullet window before using the keyboard.\n")

        # ----------------------------------------------------
        # State and event-management variables
        # ----------------------------------------------------
        state = INITIAL_STATE
        observe_until = 0.0

        # Event latch prevents repeated saves while the same animal
        # remains visible after OBSERVE has finished.
        event_latched = False
        positive_detection_streak = 0
        missing_detection_streak = 0

        last_event_message = "none"
        last_detections: list[dict[str, Any]] = []
        last_inference_ms: float | None = None
        last_inference_finished: float | None = None

        next_inference_time = 0.0
        inference_interval = 1.0 / max(INFERENCE_FPS, 0.1)

        frame_count = 0
        simulation_steps = 0

        while p.isConnected(client_id):
            now = time.perf_counter()
            key_events = p.getKeyboardEvents()

            if key_is_down(key_events, "q"):
                break

            # Finish observation and return to PATROL.
            if state == RoverState.OBSERVE and now >= observe_until:
                state = RoverState.PATROL
                positive_detection_streak = 0
                missing_detection_streak = 0
                print("[STATE] OBSERVE -> PATROL")

            # ------------------------------------------------
            # Drive command depends on the current state.
            # AVOID is intentionally inactive until STEP6.
            # ------------------------------------------------
            linear_velocity = 0.0
            angular_velocity = 0.0

            if state == RoverState.PATROL:
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

            # OBSERVE and AVOID remain stopped in this STEP4 implementation.
            set_drive_command(
                robot_id,
                wheel_joints,
                linear_velocity,
                angular_velocity,
            )

            p.stepSimulation()
            simulation_steps += 1

            # ------------------------------------------------
            # Camera update and scheduled YOLO inference
            # ------------------------------------------------
            if frame_count % CAMERA_UPDATE_INTERVAL == 0:
                robot_position, _ = p.getBasePositionAndOrientation(robot_id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=OBSERVER_CAMERA_DISTANCE,
                    cameraYaw=OBSERVER_CAMERA_YAW,
                    cameraPitch=OBSERVER_CAMERA_PITCH,
                    cameraTargetPosition=[
                        robot_position[0] + 1.3,
                        robot_position[1],
                        0.6,
                    ],
                )

                camera_bgr = render_front_camera(robot_id, projection_matrix)
                now = time.perf_counter()

                # Only update event counters on a new inference result,
                # not on every camera-preview refresh.
                if now >= next_inference_time:
                    last_detections, last_inference_ms = run_inference(
                        model,
                        camera_bgr,
                    )
                    last_inference_finished = time.perf_counter()
                    next_inference_time = (
                        last_inference_finished + inference_interval
                    )

                    event_detections = select_event_detections(last_detections)

                    if state == RoverState.PATROL:
                        if event_detections:
                            positive_detection_streak += 1
                            missing_detection_streak = 0
                        else:
                            positive_detection_streak = 0

                            if event_latched:
                                missing_detection_streak += 1
                                if (
                                    missing_detection_streak
                                    >= REARM_NO_DETECTION_INFERENCES
                                ):
                                    event_latched = False
                                    missing_detection_streak = 0
                                    print(
                                        "[EVENT] Detection disappeared; "
                                        "event trigger re-armed."
                                    )

                        # Trigger exactly once after consecutive positive inference results.
                        if (
                            not event_latched
                            and positive_detection_streak
                            >= DETECTION_CONFIRM_INFERENCES
                        ):
                            event_latched = True
                            positive_detection_streak = 0
                            missing_detection_streak = 0

                            state = RoverState.OBSERVE
                            observe_until = (
                                time.perf_counter() + OBSERVE_DURATION_SEC
                            )

                            # Stop immediately rather than waiting for the next loop.
                            set_drive_command(
                                robot_id,
                                wheel_joints,
                                0.0,
                                0.0,
                            )

                            # Save exactly one annotated image and one CSV event row.
                            try:
                                simulation_time_sec = (
                                    simulation_steps * TIME_STEP
                                )
                                event_info = save_detection_event(
                                    camera_bgr,
                                    event_detections,
                                    class_names,
                                    simulation_time_sec,
                                    robot_id,
                                )
                                last_event_message = (
                                    f"{event_info['primary_class_name']} "
                                    f"{event_info['primary_confidence']} "
                                    f"({event_info['event_id']})"
                                )
                                print("\n[WILDLIFE DETECTED]")
                                print(
                                    f"  class: {event_info['primary_class_name']}"
                                )
                                print(
                                    f"  confidence: "
                                    f"{event_info['primary_confidence']}"
                                )
                                print(f"  image: {event_info['image_path']}")
                                print(f"  log: {EVENT_LOG_PATH}")
                            except Exception as exc:
                                last_event_message = f"SAVE ERROR: {exc}"
                                print(
                                    "[ERROR] Detection event occurred, "
                                    f"but saving failed: {exc}"
                                )

                            print(
                                f"[STATE] PATROL -> OBSERVE "
                                f"({OBSERVE_DURATION_SEC:.1f} seconds)"
                            )

                inference_age = (
                    None
                    if last_inference_finished is None
                    else time.perf_counter() - last_inference_finished
                )
                observe_remaining = (
                    max(0.0, observe_until - time.perf_counter())
                    if state == RoverState.OBSERVE
                    else 0.0
                )

                camera_display = draw_step4_status(
                    camera_bgr,
                    last_detections,
                    class_names,
                    state,
                    last_inference_ms,
                    inference_age,
                    observe_remaining,
                    last_event_message,
                    event_latched,
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
