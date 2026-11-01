"""STEP5: Add a simple, reproducible forest to the STEP4 wildlife rover.

Run from the project root:
    python src/step5_forest_environment.py

The forest consists of static cylindrical trunks with collision geometry and
spherical canopies without collision geometry. The animal billboard remains
visual-only. STEP4's YOLO event handling and keyboard-driven patrol test are
preserved; automatic LiDAR obstacle avoidance is intentionally not included
until STEP6.
"""

from __future__ import annotations

import random
import time
from typing import Any

import cv2
import pybullet as p
import pybullet_data

import step3_yolo_camera as step3
import step4_detection_actions as step4
from step1_flat_husky import (
    TIME_STEP,
    find_wheel_joints,
    key_is_down,
    set_drive_command,
)


# ============================================================
# STEP5 forest settings
# ============================================================

FOREST_RANDOM_SEED = 2026

# Trees are arranged on both sides of a clear central driving lane.
# A small seeded jitter avoids a perfectly regular grid while retaining
# repeatable placement and a wide passage from Husky's spawn to the animal.
# FOREST_COLUMNS_X = (-1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0)
# FOREST_ROWS_Y = (-3.35, -1.70, 1.70, 3.35)
FOREST_COLUMNS_X = (-1.0, 1.0, 3.5, 9.0)
FOREST_ROWS_Y = (-3.50, -2.20, 2.0, 3.50)
TREE_X_JITTER = 0.28
TREE_Y_JITTER = 0.10

TRUNK_RADIUS = 0.18
TRUNK_HEIGHT = 2.0
TRUNK_COLOR = (0.34, 0.19, 0.09, 1.0)

CROWN_RADIUS = 0.82
CROWN_CENTER_Z = 2.45
CROWN_COLORS = (
    (0.13, 0.38, 0.12, 1.0),
    (0.17, 0.44, 0.14, 1.0),
    (0.22, 0.40, 0.12, 1.0),
    (0.12, 0.32, 0.10, 1.0),
)

# Keep the camera window separate from STEP4's window title.
CAMERA_WINDOW_NAME = "STEP5 - Forest Wildlife Rover"

# Wider external view so the user can inspect the forest around Husky.
OBSERVER_CAMERA_DISTANCE = 6.5
OBSERVER_CAMERA_YAW = 45.0
OBSERVER_CAMERA_PITCH = -25.0


# ============================================================
# Forest generation
# ============================================================

def generate_tree_positions(
    seed: int = FOREST_RANDOM_SEED,
) -> list[tuple[float, float]]:
    """Generate repeatable tree positions while keeping a central corridor."""

    rng = random.Random(seed)
    positions: list[tuple[float, float]] = []

    for row_index, row_y in enumerate(FOREST_ROWS_Y):
        # Stagger every second row to make the layout look less grid-like.
        row_stagger = 0.65 if row_index % 2 else 0.0

        for column_x in FOREST_COLUMNS_X:
            x = column_x + row_stagger + rng.uniform(
                -TREE_X_JITTER, TREE_X_JITTER
            )
            y = row_y + rng.uniform(-TREE_Y_JITTER, TREE_Y_JITTER)
            positions.append((round(x, 3), round(y, 3)))

    return positions


def create_forest() -> tuple[list[int], list[int], list[tuple[float, float]]]:
    """Create static tree trunks (collidable) and spherical crowns (visual only).

    Returns
    -------
    trunk_body_ids:
        Body IDs of the collidable tree trunks. STEP6 can use these IDs to
        identify forest obstacles in rayTestBatch results.
    crown_body_ids:
        Body IDs of the non-collidable canopy visuals.
    tree_positions:
        World XY coordinates, in the same order as trunk_body_ids.
    """

    tree_positions = generate_tree_positions()
    rng = random.Random(FOREST_RANDOM_SEED + 1)

    # Reuse one trunk collision shape for all trees. PyBullet supports shape
    # instancing when creating multiple bodies from the same shape ID.
    trunk_collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=TRUNK_RADIUS,
        height=TRUNK_HEIGHT,
    )
    if trunk_collision_shape_id < 0:
        raise RuntimeError("Could not create the tree-trunk collision shape.")

    trunk_body_ids: list[int] = []
    crown_body_ids: list[int] = []

    for x, y in tree_positions:
        # Visual cylinder uses 'length'; collision cylinder uses 'height'.
        trunk_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=TRUNK_RADIUS,
            length=TRUNK_HEIGHT,
            rgbaColor=TRUNK_COLOR,
        )
        if trunk_visual_shape_id < 0:
            raise RuntimeError(
                f"Could not create trunk visual shape at ({x:.2f}, {y:.2f})."
            )

        trunk_body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=trunk_collision_shape_id,
            baseVisualShapeIndex=trunk_visual_shape_id,
            basePosition=[x, y, TRUNK_HEIGHT / 2.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
        )
        if trunk_body_id < 0:
            raise RuntimeError(
                f"Could not create tree trunk body at ({x:.2f}, {y:.2f})."
            )

        p.changeDynamics(
            trunk_body_id,
            -1,
            lateralFriction=1.0,
            restitution=0.0,
        )

        # Each crown is visual-only: no collision shape, so leaves/canopy do
        # not block the robot or become false positives for future LiDAR rays.
        crown_color = rng.choice(CROWN_COLORS)
        crown_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=CROWN_RADIUS,
            rgbaColor=crown_color,
        )
        if crown_visual_shape_id < 0:
            raise RuntimeError(
                f"Could not create canopy visual shape at ({x:.2f}, {y:.2f})."
            )

        crown_body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=crown_visual_shape_id,
            basePosition=[x, y, CROWN_CENTER_Z],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
        )
        if crown_body_id < 0:
            raise RuntimeError(
                f"Could not create canopy body at ({x:.2f}, {y:.2f})."
            )

        # Runtime checks: the trunk must collide; the crown must not.
        if not p.getCollisionShapeData(trunk_body_id, -1):
            raise RuntimeError(
                f"Tree trunk {trunk_body_id} has no collision geometry."
            )
        if p.getCollisionShapeData(crown_body_id, -1):
            raise RuntimeError(
                f"Tree canopy {crown_body_id} unexpectedly has collision geometry."
            )

        trunk_body_ids.append(trunk_body_id)
        crown_body_ids.append(crown_body_id)

    print("\nForest created:")
    print(f"  random seed: {FOREST_RANDOM_SEED}")
    print(f"  tree count: {len(tree_positions)}")
    print(f"  trunk radius / height: {TRUNK_RADIUS:.2f} / {TRUNK_HEIGHT:.2f} m")
    print(f"  canopy radius: {CROWN_RADIUS:.2f} m")
    print("  trunk collision: enabled")
    print("  canopy collision: disabled")
    print("  central corridor: kept clear around y = 0")
    print(f"  trunk body IDs (for STEP6): {trunk_body_ids}")

    return trunk_body_ids, crown_body_ids, tree_positions


# ============================================================
# Main
# ============================================================

def main() -> None:
    model, class_names, model_path = step3.load_detector()

    if step4.EVENT_CLASS_IDS is not None:
        unknown_event_ids = step4.EVENT_CLASS_IDS - set(class_names)
        if unknown_event_ids:
            raise ValueError(
                "EVENT_CLASS_IDS contains IDs absent from the loaded model: "
                f"{sorted(unknown_event_ids)}"
            )

    print(f"Selected checkpoint: {model_path}")
    print(
        f"Inference: {step3.INFERENCE_DEVICE}, "
        f"{step3.YOLO_IMAGE_SIZE}x{step3.YOLO_IMAGE_SIZE}, "
        f"target {step3.INFERENCE_FPS:.1f} FPS, "
        f"conf={step3.CONF_THRESHOLD}"
    )
    print(f"Event confidence threshold: {step4.EVENT_CONF_THRESHOLD}")

    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Could not connect to the PyBullet GUI.")

    camera_window_created = False

    try:
        # ----------------------------------------------------
        # Initialize flat simulation world
        # ----------------------------------------------------
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(TIME_STEP)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(numSolverIterations=100)

        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(plane_id, -1, lateralFriction=1.0)

        # ----------------------------------------------------
        # Load Husky
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # Add forest and the visual-only animal billboard
        # ----------------------------------------------------
        trunk_body_ids, crown_body_ids, tree_positions = create_forest()

        animal_body_id = step3.create_animal_billboard(
            step3.ANIMAL_IMAGE_PATH
        )
        print(f"Animal body ID: {animal_body_id} (visual-only; no collision)")

        p.resetDebugVisualizerCamera(
            cameraDistance=OBSERVER_CAMERA_DISTANCE,
            cameraYaw=OBSERVER_CAMERA_YAW,
            cameraPitch=OBSERVER_CAMERA_PITCH,
            cameraTargetPosition=[1.3, 0.0, 0.8],
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=step3.CAMERA_FOV,
            aspect=step3.CAMERA_WIDTH / step3.CAMERA_HEIGHT,
            nearVal=step3.CAMERA_NEAR,
            farVal=step3.CAMERA_FAR,
        )

        cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            CAMERA_WINDOW_NAME,
            step3.CAMERA_WIDTH,
            step3.CAMERA_HEIGHT,
        )
        camera_window_created = True

        print("\nSTEP5: Forest environment + STEP4 wildlife event behavior")
        print("  W     : Forward while in PATROL")
        print("  S     : Backward while in PATROL")
        print("  A     : Turn left while in PATROL")
        print("  D     : Turn right while in PATROL")
        print("  SPACE : Stop")
        print("  Q     : Quit")
        print("\nSTEP5 does not include LiDAR avoidance yet.")
        print("Tree trunks are physical obstacles; canopies and animal billboard are visual-only.")
        print("A wide central passage is kept clear for the initial forward test.")
        print("Click the PyBullet window before using the keyboard.\n")

        # ----------------------------------------------------
        # STEP4 state and event-management variables
        # ----------------------------------------------------
        state = step4.INITIAL_STATE
        observe_until = 0.0

        event_latched = False
        positive_detection_streak = 0
        missing_detection_streak = 0

        last_event_message = "none"
        last_detections: list[dict[str, Any]] = []
        last_inference_ms: float | None = None
        last_inference_finished: float | None = None

        next_inference_time = 0.0
        inference_interval = 1.0 / max(step3.INFERENCE_FPS, 0.1)

        frame_count = 0
        simulation_steps = 0

        while p.isConnected(client_id):
            now = time.perf_counter()
            key_events = p.getKeyboardEvents()

            if key_is_down(key_events, "q"):
                break

            # Finish observation and return to PATROL.
            if state == step4.RoverState.OBSERVE and now >= observe_until:
                state = step4.RoverState.PATROL
                positive_detection_streak = 0
                missing_detection_streak = 0
                print("[STATE] OBSERVE -> PATROL")

            # Keyboard-driven PATROL is retained until STEP6.
            linear_velocity = 0.0
            angular_velocity = 0.0

            if state == step4.RoverState.PATROL:
                if key_is_down(key_events, "w"):
                    linear_velocity += step3.LINEAR_SPEED
                if key_is_down(key_events, "s"):
                    linear_velocity -= step3.LINEAR_SPEED
                if key_is_down(key_events, "a"):
                    angular_velocity += step3.ANGULAR_SPEED
                if key_is_down(key_events, "d"):
                    angular_velocity -= step3.ANGULAR_SPEED

                if key_events.get(ord(" "), 0) & p.KEY_IS_DOWN:
                    linear_velocity = 0.0
                    angular_velocity = 0.0

            # OBSERVE and AVOID are stopped in this STEP5 version.
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
            if frame_count % step3.CAMERA_UPDATE_INTERVAL == 0:
                robot_position, _ = p.getBasePositionAndOrientation(robot_id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=OBSERVER_CAMERA_DISTANCE,
                    cameraYaw=OBSERVER_CAMERA_YAW,
                    cameraPitch=OBSERVER_CAMERA_PITCH,
                    cameraTargetPosition=[
                        robot_position[0] + 1.3,
                        robot_position[1],
                        0.8,
                    ],
                )

                camera_bgr = step3.render_front_camera(
                    robot_id,
                    projection_matrix,
                )
                now = time.perf_counter()

                # Event counters are updated only on new inference results.
                if now >= next_inference_time:
                    last_detections, last_inference_ms = step3.run_inference(
                        model,
                        camera_bgr,
                    )
                    last_inference_finished = time.perf_counter()
                    next_inference_time = (
                        last_inference_finished + inference_interval
                    )

                    event_detections = step4.select_event_detections(
                        last_detections
                    )

                    if state == step4.RoverState.PATROL:
                        if event_detections:
                            positive_detection_streak += 1
                            missing_detection_streak = 0
                        else:
                            positive_detection_streak = 0

                            if event_latched:
                                missing_detection_streak += 1
                                if (
                                    missing_detection_streak
                                    >= step4.REARM_NO_DETECTION_INFERENCES
                                ):
                                    event_latched = False
                                    missing_detection_streak = 0
                                    print(
                                        "[EVENT] Detection disappeared; "
                                        "event trigger re-armed."
                                    )

                        if (
                            not event_latched
                            and positive_detection_streak
                            >= step4.DETECTION_CONFIRM_INFERENCES
                        ):
                            event_latched = True
                            positive_detection_streak = 0
                            missing_detection_streak = 0

                            state = step4.RoverState.OBSERVE
                            observe_until = (
                                time.perf_counter()
                                + step4.OBSERVE_DURATION_SEC
                            )

                            set_drive_command(
                                robot_id,
                                wheel_joints,
                                0.0,
                                0.0,
                            )

                            try:
                                simulation_time_sec = (
                                    simulation_steps * TIME_STEP
                                )
                                event_info = step4.save_detection_event(
                                    camera_bgr,
                                    event_detections,
                                    class_names,
                                    simulation_time_sec,
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
                                    "  confidence: "
                                    f"{event_info['primary_confidence']}"
                                )
                                print(f"  image: {event_info['image_path']}")
                                print(f"  log: {step4.EVENT_LOG_PATH}")
                            except Exception as exc:
                                last_event_message = f"SAVE ERROR: {exc}"
                                print(
                                    "[ERROR] Detection event occurred, "
                                    f"but saving failed: {exc}"
                                )

                            print(
                                "[STATE] PATROL -> OBSERVE "
                                f"({step4.OBSERVE_DURATION_SEC:.1f} seconds)"
                            )

                inference_age = (
                    None
                    if last_inference_finished is None
                    else time.perf_counter() - last_inference_finished
                )
                observe_remaining = (
                    max(0.0, observe_until - time.perf_counter())
                    if state == step4.RoverState.OBSERVE
                    else 0.0
                )

                camera_display = step4.draw_step4_status(
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
