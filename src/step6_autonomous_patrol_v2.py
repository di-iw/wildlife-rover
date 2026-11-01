"""STEP6 v2: Footprint-aware autonomous Husky patrol in a forest.

Run from the project root:
    python src/step6_autonomous_patrol_v2.py

Dependencies are the already completed STEP1, STEP3, STEP4 and STEP5 modules.
The controller keeps the explicit detour phases:
    TURN_OUT -> SHIFT -> TURN_ALIGN -> PASS -> PATROL

Navigation design:
- 360-degree 2D virtual LiDAR observes registered tree trunks.
- Detected trunks are retained in a small rolling, sparse obstacle cache.
  This is not an occupancy-grid map or SLAM (STEP7 remains separate).
- Both left and right detours are generated before motion starts.
- Every candidate is checked against all cached trunks using the Husky
  rectangular footprint, including the outer edges of its four tires.
- AVOID replanning occurs only if a newly observed obstacle intersects the
  remaining planned path, not merely because a different tree appears ahead.
- An unexpected physical contact with a tree causes an immediate stop.
- OBSERVE has priority over AVOID and PATROL; the active AVOID phase is saved
  and resumed after observation.

The robot footprint dimensions below are derived from the PyBullet
pybullet_data/husky/husky.urdf collision geometry (base box + wheel cylinders).
"""

from __future__ import annotations

import math
import numpy as np
import time
from typing import Any

import cv2
import pybullet as p
import pybullet_data

import step3_yolo_camera as step3
import step4_detection_actions as step4
import step5_forest_environment as step5
from step1_flat_husky import TIME_STEP, find_wheel_joints, key_is_down, set_drive_command


# ============================================================
# Patrol route and deterministic obstacle test
# ============================================================

PATROL_WAYPOINTS: tuple[tuple[float, float], ...] = (
    (7.0, 0.0),
    (7.0, 0.85),
    (0.0, 0.85),
    (0.0, 0.0),
)
WAYPOINT_REACHED_DISTANCE = 0.35

# Keep enabled for a reproducible first avoidance test. Set False when testing
# only the STEP5 forest layout.
ADD_ROUTE_TEST_TREE = True
ROUTE_TEST_TREE_POSITION = (5.0, 0.0)


# ============================================================
# Virtual 2D LiDAR
# ============================================================

# The scan plane is above the wheel/chassis collision geometry. A centered
# origin allows a full 360-degree scan without the rover body occluding rays.
LIDAR_LOCAL_ORIGIN = (0.0, 0.0, 0.48)
LIDAR_ANGLES_DEG = tuple(range(-180, 180, 3))
LIDAR_MAX_RANGE = 5.0
LIDAR_UPDATE_INTERVAL_SEC = 0.10
SHOW_LIDAR_RAYS = True
LIDAR_RAY_LIFETIME = 0.16

# Keep only a local sparse cache of trunks actually observed by LiDAR.
OBSTACLE_MEMORY_RADIUS = 6.0
OBSTACLE_MEMORY_SMOOTHING = 0.25


# ============================================================
# Husky footprint and collision margins
# ============================================================

# PyBullet Husky URDF base collision box: 1.0074 x 0.5709 m.
ROBOT_HALF_LENGTH = 1.0074 / 2.0
# Wheel centers are at y=+/-0.2854 m; each tire is 0.1143 m wide.
# The tires therefore extend to +/- (0.2854 + 0.1143/2) = 0.34255 m.
ROBOT_HALF_WIDTH = 0.2854 + 0.1143 / 2.0
ROBOT_SWEEP_RADIUS = math.hypot(ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH)

# Additional clearance beyond the geometric robot/trunk collision boundary.
# Increase this when the rover passes too close; decrease only after verifying.
COLLISION_SAFETY_MARGIN = 0.12


# ============================================================
# Patrol controller
# ============================================================

PATROL_LINEAR_SPEED = 0.24
PATROL_ANGULAR_GAIN = 1.45
PATROL_MAX_ANGULAR_SPEED = 0.62
PATROL_ALIGN_ONLY_ANGLE_DEG = 18.0
PATROL_HEADING_CORRECTION_LIMIT = 0.22

# Inspect only the near part of the route; the LiDAR range is 5 m.
AVOID_TRIGGER_LOOKAHEAD = 3.2


# ============================================================
# Local detour planner and AVOID controller
# ============================================================

# Lateral centerline offset from the detected trunk center. The minimum
# geometric value is ROBOT_HALF_WIDTH + trunk_radius + safety_margin.
AVOID_SIDE_EXTRA_CLEARANCE = 0.10
AVOID_SIDE_OFFSET = (
    ROBOT_HALF_WIDTH + step5.TRUNK_RADIUS
    + COLLISION_SAFETY_MARGIN + AVOID_SIDE_EXTRA_CLEARANCE
)

# Keep the robot center this far before/after the trunk while rotating/passing.
AVOID_LONGITUDINAL_EXTRA_CLEARANCE = 0.15
AVOID_BEFORE_TRUNK_OFFSET = (
    ROBOT_HALF_LENGTH + step5.TRUNK_RADIUS
    + COLLISION_SAFETY_MARGIN + AVOID_LONGITUDINAL_EXTRA_CLEARANCE
)
AVOID_AFTER_TRUNK_OFFSET = AVOID_BEFORE_TRUNK_OFFSET

AVOID_SHIFT_SPEED = 0.18
AVOID_PASS_SPEED = 0.20
AVOID_TURN_GAIN = 1.8
AVOID_MAX_TURN_RATE = 0.48
AVOID_HEADING_GAIN = 1.2
AVOID_MAX_HEADING_CORRECTION = 0.12
AVOID_TURN_TOLERANCE_DEG = 3.5
AVOID_ALIGN_TOLERANCE_DEG = 4.0
AVOID_SHIFT_TOLERANCE = 0.15
AVOID_PASS_TOLERANCE = 0.18

# These limits include the longer SHIFT distance after early obstacle detection.
# AVOID_PHASE_TIMEOUTS = {
#     "TURN_OUT": 12.0,
#     "SHIFT": 26.0,
#     "TURN_ALIGN": 12.0,
#     "PASS": 20.0,
# }
AVOID_PHASE_TIMEOUTS = {
    "TURN_OUT": 30.0,
    "SHIFT": 26.0,
    "TURN_ALIGN": 30.0,
    "PASS": 20.0,
}
AVOID_STALL_CHECK_INTERVAL = 3.0
AVOID_STALL_MIN_POSITION_PROGRESS = 0.04
AVOID_STALL_MIN_ANGLE_PROGRESS = math.radians(2.5)
AVOID_MAX_REPLANS = 2
AVOID_COOLDOWN_SEC = 0.5


# ============================================================
# Display
# ============================================================

CAMERA_WINDOW_NAME = "STEP6 v2 - Autonomous Wildlife Patrol Rover"
OBSERVER_CAMERA_DISTANCE = 7.0
OBSERVER_CAMERA_YAW = -45.0
OBSERVER_CAMERA_PITCH = -45.0


# ============================================================
# Math and footprint helpers
# ============================================================


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def get_robot_xy_yaw(robot_id: int) -> tuple[float, float, float]:
    position, orientation = p.getBasePositionAndOrientation(robot_id)
    yaw = p.getEulerFromQuaternion(orientation)[2]
    return float(position[0]), float(position[1]), float(yaw)


def point_to_footprint_clearance(
    robot_xy: tuple[float, float],
    robot_yaw: float,
    obstacle_xy: tuple[float, float],
    obstacle_radius: float,
) -> float:
    """Distance from a circular trunk surface to the oriented robot rectangle.

    A non-positive value means geometric overlap. The safety margin is applied
    separately by the caller, so it remains an easy-to-tune parameter.
    """
    dx = obstacle_xy[0] - robot_xy[0]
    dy = obstacle_xy[1] - robot_xy[1]
    c = math.cos(robot_yaw)
    s = math.sin(robot_yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy

    outside_x = max(abs(local_x) - ROBOT_HALF_LENGTH, 0.0)
    outside_y = max(abs(local_y) - ROBOT_HALF_WIDTH, 0.0)
    return math.hypot(outside_x, outside_y) - obstacle_radius


def segment_clearance(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    heading: float,
    obstacles: dict[int, dict[str, Any]],
    sample_spacing: float = 0.10,
) -> tuple[bool, float, int | None]:
    """Check the swept rectangular footprint along a straight segment."""
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    length = math.hypot(dx, dy)
    sample_count = max(1, math.ceil(length / sample_spacing))

    minimum_clearance = float("inf")
    first_collision_fraction = float("inf")
    first_collision_clearance = float("inf")
    first_collision_id: int | None = None

    for sample_index in range(sample_count + 1):
        fraction = sample_index / sample_count
        robot_xy = (
            start_xy[0] + dx * fraction,
            start_xy[1] + dy * fraction,
        )
        for obstacle_id, obstacle in obstacles.items():
            clearance = point_to_footprint_clearance(
                robot_xy,
                heading,
                obstacle["xy"],
                float(obstacle["radius"]),
            )
            minimum_clearance = min(minimum_clearance, clearance)
            if clearance < COLLISION_SAFETY_MARGIN:
                earlier = fraction < first_collision_fraction - 1e-9
                same_sample_but_closer = (
                    abs(fraction - first_collision_fraction) <= 1e-9
                    and clearance < first_collision_clearance
                )
                if earlier or same_sample_but_closer:
                    first_collision_fraction = fraction
                    first_collision_clearance = clearance
                    first_collision_id = obstacle_id

    if not obstacles:
        minimum_clearance = float("inf")

    return (
        first_collision_id is None,
        minimum_clearance,
        first_collision_id,
    )


def rotation_clearance(
    robot_xy: tuple[float, float],
    obstacles: dict[int, dict[str, Any]],
) -> tuple[bool, float, int | None]:
    """Conservatively check a full in-place rotation using the footprint radius."""
    minimum_clearance = float("inf")
    blocker_id: int | None = None
    blocker_distance = float("inf")

    for obstacle_id, obstacle in obstacles.items():
        distance = math.hypot(
            obstacle["xy"][0] - robot_xy[0],
            obstacle["xy"][1] - robot_xy[1],
        )
        clearance = (
            distance - ROBOT_SWEEP_RADIUS - float(obstacle["radius"])
        )
        if clearance < minimum_clearance:
            minimum_clearance = clearance
        if clearance < COLLISION_SAFETY_MARGIN:
            if blocker_id is None or distance < blocker_distance:
                blocker_id = obstacle_id
                blocker_distance = distance

    return (
        blocker_id is None,
        minimum_clearance,
        blocker_id,
    )


def command_rotation_to_heading(
    robot_id: int,
    wheel_joints: dict[str, int],
    target_yaw: float,
    tolerance_deg: float,
) -> bool:
    """Rotate toward a fixed heading; return True when aligned."""
    _, _, yaw = get_robot_xy_yaw(robot_id)
    error = wrap_angle(target_yaw - yaw)

    if abs(error) <= math.radians(tolerance_deg):
        set_drive_command(robot_id, wheel_joints, 0.0, 0.0)
        return True

    angular = clamp(
        AVOID_TURN_GAIN * error,
        -AVOID_MAX_TURN_RATE,
        AVOID_MAX_TURN_RATE,
    )
    set_drive_command(robot_id, wheel_joints, 0.0, angular)
    return False


def drive_along_fixed_heading(
    robot_id: int,
    wheel_joints: dict[str, int],
    target_yaw: float,
    linear_speed: float,
) -> None:
    """Drive on a fixed heading; large heading errors rotate in place."""
    _, _, yaw = get_robot_xy_yaw(robot_id)
    error = wrap_angle(target_yaw - yaw)

    if abs(error) > math.radians(AVOID_TURN_TOLERANCE_DEG * 2.0):
        angular = clamp(
            AVOID_TURN_GAIN * error,
            -AVOID_MAX_TURN_RATE,
            AVOID_MAX_TURN_RATE,
        )
        set_drive_command(robot_id, wheel_joints, 0.0, angular)
        return

    angular = clamp(
        AVOID_HEADING_GAIN * error,
        -AVOID_MAX_HEADING_CORRECTION,
        AVOID_MAX_HEADING_CORRECTION,
    )
    set_drive_command(robot_id, wheel_joints, linear_speed, angular)


def patrol_drive_command(
    robot_id: int,
    target_xy: tuple[float, float],
) -> tuple[float, float, float]:
    """Return (distance, linear, angular) for waypoint-seeking patrol motion."""
    x, y, yaw = get_robot_xy_yaw(robot_id)
    dx = target_xy[0] - x
    dy = target_xy[1] - y
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        return distance, 0.0, 0.0

    target_yaw = math.atan2(dy, dx)
    error = wrap_angle(target_yaw - yaw)
    angular = clamp(
        PATROL_ANGULAR_GAIN * error,
        -PATROL_MAX_ANGULAR_SPEED,
        PATROL_MAX_ANGULAR_SPEED,
    )

    if abs(error) > math.radians(PATROL_ALIGN_ONLY_ANGLE_DEG):
        linear = 0.0
    else:
        linear = PATROL_LINEAR_SPEED * max(0.35, math.cos(error))
        angular = clamp(
            angular,
            -PATROL_HEADING_CORRECTION_LIMIT,
            PATROL_HEADING_CORRECTION_LIMIT,
        )

    return distance, linear, angular


# ============================================================
# Additional route-test tree
# ============================================================


def create_route_test_tree(position_xy: tuple[float, float]) -> tuple[int, int]:
    """Create a collidable trunk and visual-only crown at a route waypoint."""
    x, y = position_xy

    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=step5.TRUNK_RADIUS,
        height=step5.TRUNK_HEIGHT,
    )
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=step5.TRUNK_RADIUS,
        length=step5.TRUNK_HEIGHT,
        rgbaColor=step5.TRUNK_COLOR,
    )
    if collision_shape_id < 0 or visual_shape_id < 0:
        raise RuntimeError("Could not create the route-test tree trunk.")

    trunk_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[x, y, step5.TRUNK_HEIGHT / 2.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )
    p.changeDynamics(trunk_id, -1, lateralFriction=1.0, restitution=0.0)

    crown_visual_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=step5.CROWN_RADIUS,
        rgbaColor=step5.CROWN_COLORS[0],
    )
    if crown_visual_id < 0:
        raise RuntimeError("Could not create the route-test tree crown.")

    crown_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=crown_visual_id,
        basePosition=[x, y, step5.CROWN_CENTER_Z],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )

    if not p.getCollisionShapeData(trunk_id, -1):
        raise RuntimeError("Route-test trunk has no collision geometry.")
    if p.getCollisionShapeData(crown_id, -1):
        raise RuntimeError("Route-test crown unexpectedly has collision geometry.")

    print(f"  route-test tree: body={trunk_id}, xy=({x:.2f}, {y:.2f})")
    return trunk_id, crown_id


# ============================================================
# Virtual LiDAR and local obstacle cache
# ============================================================


def scan_virtual_lidar(robot_id: int, trunk_body_ids: set[int]) -> dict[str, Any]:
    """Cast a 360-degree scan and estimate trunk centers from hit normals.

    For a cylindrical trunk, the ray hit point minus its outward horizontal
    surface normal times the known trunk radius estimates the trunk center.
    This uses the ray result, rather than preloading every tree coordinate.
    """
    base_position, base_orientation = p.getBasePositionAndOrientation(robot_id)
    ray_origin = p.multiplyTransforms(
        base_position,
        base_orientation,
        LIDAR_LOCAL_ORIGIN,
        [0.0, 0.0, 0.0, 1.0],
    )[0]
    yaw = p.getEulerFromQuaternion(base_orientation)[2]

    ray_from = [list(ray_origin) for _ in LIDAR_ANGLES_DEG]
    ray_to: list[list[float]] = []
    for angle_deg in LIDAR_ANGLES_DEG:
        angle = yaw + math.radians(angle_deg)
        ray_to.append([
            ray_origin[0] + LIDAR_MAX_RANGE * math.cos(angle),
            ray_origin[1] + LIDAR_MAX_RANGE * math.sin(angle),
            ray_origin[2],
        ])

    raw_hits = p.rayTestBatch(ray_from, ray_to)
    rays: list[dict[str, Any]] = []
    center_candidates: dict[int, list[tuple[float, float]]] = {}
    front_distances: list[float] = []
    left_distances: list[float] = []
    right_distances: list[float] = []

    for angle_deg, endpoint, result in zip(LIDAR_ANGLES_DEG, ray_to, raw_hits):
        body_id = int(result[0])
        hit_fraction = float(result[2])
        hit_position = tuple(float(v) for v in result[3])
        hit_normal = tuple(float(v) for v in result[4])
        is_trunk = body_id in trunk_body_ids

        distance = (
            clamp(hit_fraction * LIDAR_MAX_RANGE, 0.0, LIDAR_MAX_RANGE)
            if is_trunk else LIDAR_MAX_RANGE
        )
        if is_trunk:
            normal_xy_length = math.hypot(hit_normal[0], hit_normal[1])
            if normal_xy_length > 0.5:
                nx = hit_normal[0] / normal_xy_length
                ny = hit_normal[1] / normal_xy_length
                center_candidates.setdefault(body_id, []).append((
                    hit_position[0] - step5.TRUNK_RADIUS * nx,
                    hit_position[1] - step5.TRUNK_RADIUS * ny,
                ))

        rays.append({
            "angle_deg": angle_deg,
            "distance": distance,
            "body_id": body_id if is_trunk else None,
            "hit_position": hit_position if is_trunk else tuple(endpoint),
        })

        if is_trunk:
            if abs(angle_deg) <= 30:
                front_distances.append(distance)
            elif 30 < angle_deg < 150:
                left_distances.append(distance)
            elif -150 < angle_deg < -30:
                right_distances.append(distance)

    obstacle_centers = {
        body_id: (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
        for body_id, points in center_candidates.items()
        if points
    }

    return {
        "origin": tuple(float(v) for v in ray_origin),
        "rays": rays,
        "obstacle_centers": obstacle_centers,
        "front_min_distance": min(front_distances, default=LIDAR_MAX_RANGE),
        "left_clearance": min(left_distances, default=LIDAR_MAX_RANGE),
        "right_clearance": min(right_distances, default=LIDAR_MAX_RANGE),
        "trunk_hit_count": len(obstacle_centers),
    }


def update_obstacle_memory(
    scan: dict[str, Any],
    obstacle_memory: dict[int, dict[str, Any]],
    robot_xy: tuple[float, float],
    now: float,
) -> None:
    """Update a rolling cache of trunks seen by LiDAR; no occupancy grid."""
    for obstacle_id, center_xy in scan["obstacle_centers"].items():
        if obstacle_id in obstacle_memory:
            old_xy = obstacle_memory[obstacle_id]["xy"]
            alpha = OBSTACLE_MEMORY_SMOOTHING
            smoothed_xy = (
                (1.0 - alpha) * old_xy[0] + alpha * center_xy[0],
                (1.0 - alpha) * old_xy[1] + alpha * center_xy[1],
            )
        else:
            smoothed_xy = center_xy

        obstacle_memory[obstacle_id] = {
            "xy": smoothed_xy,
            "radius": step5.TRUNK_RADIUS,
            "last_seen": now,
        }

    expired_ids = [
        obstacle_id
        for obstacle_id, obstacle in obstacle_memory.items()
        if math.hypot(
            obstacle["xy"][0] - robot_xy[0],
            obstacle["xy"][1] - robot_xy[1],
        ) > OBSTACLE_MEMORY_RADIUS
    ]
    for obstacle_id in expired_ids:
        del obstacle_memory[obstacle_id]


def draw_lidar_rays(scan: dict[str, Any]) -> None:
    if not SHOW_LIDAR_RAYS:
        return
    origin = scan["origin"]
    for ray in scan["rays"]:
        hit = ray["body_id"] is not None
        p.addUserDebugLine(
            lineFromXYZ=origin,
            lineToXYZ=ray["hit_position"],
            lineColorRGB=[1.0, 0.12, 0.08] if hit else [0.15, 0.85, 0.25],
            lineWidth=1.0,
            lifeTime=LIDAR_RAY_LIFETIME,
        )


def draw_robot_footprint(robot_id: int) -> None:
    """Draw the conservative rectangle that the planner uses for collision checks."""
    x, y, yaw = get_robot_xy_yaw(robot_id)
    c, s = math.cos(yaw), math.sin(yaw)
    local_corners = (
        (ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH),
        (ROBOT_HALF_LENGTH, -ROBOT_HALF_WIDTH),
        (-ROBOT_HALF_LENGTH, -ROBOT_HALF_WIDTH),
        (-ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH),
    )
    corners = [
        (x + c * lx - s * ly, y + s * lx + c * ly, 0.035)
        for lx, ly in local_corners
    ]
    for index in range(4):
        p.addUserDebugLine(
            corners[index],
            corners[(index + 1) % 4],
            lineColorRGB=[0.15, 0.75, 1.0],
            lineWidth=2.0,
            lifeTime=LIDAR_RAY_LIFETIME,
        )


def draw_avoid_plan(avoid_context: dict[str, Any] | None) -> None:
    if not avoid_context:
        return
    shift = avoid_context["shift_point"]
    passed = avoid_context["pass_point"]
    start = avoid_context["entry_xy"]
    p.addUserDebugLine(
        [start[0], start[1], 0.06],
        [shift[0], shift[1], 0.06],
        lineColorRGB=[0.0, 0.85, 1.0],
        lineWidth=3.0,
        lifeTime=LIDAR_RAY_LIFETIME,
    )
    p.addUserDebugLine(
        [shift[0], shift[1], 0.06],
        [passed[0], passed[1], 0.06],
        lineColorRGB=[0.0, 0.85, 1.0],
        lineWidth=3.0,
        lifeTime=LIDAR_RAY_LIFETIME,
    )


# ============================================================
# Route blocker detection and candidate detour planning
# ============================================================


def find_first_route_blocker(
    robot_id: int,
    target_xy: tuple[float, float],
    obstacles: dict[int, dict[str, Any]],
) -> int | None:
    """Find the first observed trunk that blocks the near waypoint route."""
    if not obstacles:
        return None

    x, y, yaw = get_robot_xy_yaw(robot_id)
    dx = target_xy[0] - x
    dy = target_xy[1] - y
    target_distance = math.hypot(dx, dy)
    if target_distance <= WAYPOINT_REACHED_DISTANCE:
        return None

    route_yaw = math.atan2(dy, dx)
    heading_error = abs(wrap_angle(route_yaw - yaw))

    # A skid-steer turn sweeps the whole rectangular footprint around its center.
    if heading_error > math.radians(PATROL_ALIGN_ONLY_ANGLE_DEG):
        rotation_clear, _, rotation_blocker = rotation_clearance((x, y), obstacles)
        if not rotation_clear:
            return rotation_blocker

    lookahead = min(target_distance, AVOID_TRIGGER_LOOKAHEAD)
    endpoint = (
        x + lookahead * math.cos(route_yaw),
        y + lookahead * math.sin(route_yaw),
    )
    path_clear, _, path_blocker = segment_clearance(
        (x, y), endpoint, route_yaw, obstacles
    )
    return None if path_clear else path_blocker


def evaluate_detour_candidate(
    robot_xy: tuple[float, float],
    current_yaw: float,
    route_yaw: float,
    obstacle_id: int,
    obstacle: dict[str, Any],
    direction: int,
    obstacles: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Build and collision-check one complete side-detour candidate."""
    forward = (math.cos(route_yaw), math.sin(route_yaw))
    left = (-math.sin(route_yaw), math.cos(route_yaw))
    obstacle_x, obstacle_y = obstacle["xy"]

    rel_x = obstacle_x - robot_xy[0]
    rel_y = obstacle_y - robot_xy[1]
    obstacle_lateral = rel_x * left[0] + rel_y * left[1]
    target_lateral = obstacle_lateral + direction * AVOID_SIDE_OFFSET

    shift_point = (
        obstacle_x
        - forward[0] * AVOID_BEFORE_TRUNK_OFFSET
        + left[0] * target_lateral,
        obstacle_y
        - forward[1] * AVOID_BEFORE_TRUNK_OFFSET
        + left[1] * target_lateral,
    )
    pass_point = (
        obstacle_x
        + forward[0] * AVOID_AFTER_TRUNK_OFFSET
        + left[0] * target_lateral,
        obstacle_y
        + forward[1] * AVOID_AFTER_TRUNK_OFFSET
        + left[1] * target_lateral,
    )
    shift_yaw = math.atan2(
        shift_point[1] - robot_xy[1],
        shift_point[0] - robot_xy[0],
    )

    start_turn_clear, start_turn_margin, start_turn_blocker = rotation_clearance(
        robot_xy, obstacles
    )
    shift_clear, shift_margin, shift_blocker = segment_clearance(
        robot_xy, shift_point, shift_yaw, obstacles
    )
    align_clear, align_margin, align_blocker = rotation_clearance(
        shift_point, obstacles
    )
    pass_clear, pass_margin, pass_blocker = segment_clearance(
        shift_point, pass_point, route_yaw, obstacles
    )

    margins = (start_turn_margin, shift_margin, align_margin, pass_margin)
    min_margin = min(margins)
    feasible = all((start_turn_clear, shift_clear, align_clear, pass_clear))

    blockers = (
        start_turn_blocker,
        shift_blocker,
        align_blocker,
        pass_blocker,
    )
    blocking_id = next((item for item in blockers if item is not None), None)

    # Among collision-free candidates, prefer clearance and avoid unnecessary
    # steering that points away from the patrol goal.
    turn_cost = abs(wrap_angle(shift_yaw - current_yaw))
    score = min_margin - 0.035 * turn_cost - 0.015 * abs(target_lateral)

    return {
        "feasible": feasible,
        "score": score,
        "min_clearance": min_margin,
        "obstacle_id": obstacle_id,
        "direction": direction,
        "shift_point": shift_point,
        "pass_point": pass_point,
        "shift_yaw": shift_yaw,
        "entry_yaw": route_yaw,
        "blocking_id": blocking_id,
        "phase": "TURN_OUT",
        "target_lateral": target_lateral,
    }


def create_avoid_plan(
    robot_id: int,
    target_xy: tuple[float, float],
    obstacle_id: int,
    obstacles: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Evaluate left/right full detours and return the best safe candidate."""
    if obstacle_id not in obstacles:
        return None, f"Obstacle {obstacle_id} is not present in the local LiDAR cache."

    x, y, current_yaw = get_robot_xy_yaw(robot_id)
    goal_yaw = math.atan2(target_xy[1] - y, target_xy[0] - x)
    robot_xy = (x, y)

    candidates = [
        evaluate_detour_candidate(
            robot_xy,
            current_yaw,
            goal_yaw,
            obstacle_id,
            obstacles[obstacle_id],
            direction,
            obstacles,
        )
        for direction in (1, -1)
    ]
    feasible_candidates = [candidate for candidate in candidates if candidate["feasible"]]

    if not feasible_candidates:
        details = ", ".join(
            f"{'LEFT' if item['direction'] > 0 else 'RIGHT'} blocked by "
            f"{item['blocking_id']} (min clearance={item['min_clearance']:.2f}m)"
            for item in candidates
        )
        return None, f"No collision-free side detour. {details}"

    selected = max(feasible_candidates, key=lambda item: item["score"])
    selected["entry_xy"] = robot_xy
    selected["target_xy"] = target_xy
    return selected, ""


def find_remaining_plan_blocker(
    robot_id: int,
    avoid_context: dict[str, Any],
    obstacles: dict[int, dict[str, Any]],
) -> int | None:
    """Check only the uncompleted portion of the active AVOID plan."""
    x, y, _ = get_robot_xy_yaw(robot_id)
    phase = avoid_context["phase"]
    current_xy = (x, y)

    if phase in ("TURN_OUT", "TURN_ALIGN"):
        is_clear, _, blocker = rotation_clearance(current_xy, obstacles)
    elif phase == "SHIFT":
        is_clear, _, blocker = segment_clearance(
            current_xy,
            avoid_context["shift_point"],
            float(avoid_context["shift_yaw"]),
            obstacles,
        )
    elif phase == "PASS":
        is_clear, _, blocker = segment_clearance(
            current_xy,
            avoid_context["pass_point"],
            float(avoid_context["entry_yaw"]),
            obstacles,
        )
    else:
        return None

    return None if is_clear else blocker


def avoid_progress_metric(
    robot_id: int,
    avoid_context: dict[str, Any],
) -> tuple[float, bool]:
    """Return distance/angle error and whether it is an angular metric."""
    x, y, yaw = get_robot_xy_yaw(robot_id)
    phase = avoid_context["phase"]
    if phase == "TURN_OUT":
        return abs(wrap_angle(float(avoid_context["shift_yaw"]) - yaw)), True
    if phase == "TURN_ALIGN":
        return abs(wrap_angle(float(avoid_context["entry_yaw"]) - yaw)), True
    target = avoid_context["shift_point"] if phase == "SHIFT" else avoid_context["pass_point"]
    return math.hypot(target[0] - x, target[1] - y), False


# ============================================================
# Camera display
# ============================================================


def draw_status(
    frame_bgr,
    detections: list[dict[str, Any]],
    class_names: dict[int, str],
    state: step4.RoverState,
    resume_state: step4.RoverState,
    inference_ms: float | None,
    inference_age_sec: float | None,
    observe_remaining_sec: float,
    last_event_message: str,
    event_latched: bool,
    waypoint_index: int,
    scan: dict[str, Any],
    avoid_context: dict[str, Any] | None,
    avoid_failed: bool,
    motion_time: float,
    obstacle_memory_count: int,
):
    output = step3.draw_detections(frame_bgr, detections, class_names)

    if state == step4.RoverState.OBSERVE:
        state_text = (
            f"STATE: OBSERVE ({observe_remaining_sec:.1f}s) / "
            f"RESUME {resume_state.value}"
        )
        state_color = (0, 0, 230)
    elif state == step4.RoverState.AVOID:
        phase = avoid_context["phase"] if avoid_context else "UNKNOWN"
        state_text = f"STATE: AVOID / {phase}"
        state_color = (0, 165, 255)
    else:
        state_text = f"STATE: PATROL / WP {waypoint_index + 1}/{len(PATROL_WAYPOINTS)}"
        state_color = (40, 190, 40)

    if avoid_failed:
        state_text = "STATE: AVOID FAILED - STOPPED"
        state_color = (0, 0, 230)

    cv2.rectangle(output, (0, 0), (step3.CAMERA_WIDTH, 112), (24, 24, 24), -1)
    cv2.putText(output, state_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, state_color, 2, cv2.LINE_AA)

    lidar_text = (
        f"LiDAR front {scan['front_min_distance']:.2f}m | "
        f"L/R {scan['left_clearance']:.2f}/{scan['right_clearance']:.2f}m | "
        f"hits {scan['trunk_hit_count']} | local {obstacle_memory_count}"
    )
    cv2.putText(output, lidar_text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (235, 235, 235), 1, cv2.LINE_AA)

    if inference_ms is None:
        infer_text = "YOLO: waiting"
    else:
        age = inference_age_sec if inference_age_sec is not None else 0.0
        infer_text = f"YOLO latency {inference_ms:.0f}ms | result age {age:.1f}s"
    cv2.putText(output, infer_text, (10, 67), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (235, 235, 235), 1, cv2.LINE_AA)

    if avoid_context:
        direction_text = "LEFT" if avoid_context["direction"] > 0 else "RIGHT"
        avoid_text = (
            f"AVOID {avoid_context['phase']} {direction_text} | "
            f"clearance {avoid_context['min_clearance']:.2f}m | "
            f"replans {avoid_context['replan_count']} | t={motion_time:.1f}s"
        )
    else:
        avoid_text = "AVOID: inactive"
    cv2.putText(output, avoid_text, (10, 89), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (200, 220, 255), 1, cv2.LINE_AA)

    latch_text = "EVENT LATCHED" if event_latched else "EVENT READY"
    cv2.putText(output, f"{latch_text} | {last_event_message}"[:110],
                (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (0, 220, 255) if event_latched else (190, 190, 190),
                1, cv2.LINE_AA)
    return output


# ========================================================
# 3画面同期・動画出力設定
# ========================================================
wireframe_mode = True
output_video_path = step3.PROJECT_ROOT / "outputs" / "patrol_3views_combined.mp4"
output_video_path.parent.mkdir(parents=True, exist_ok=True)

# 全体動画解像度 (1280x720)
TOTAL_VID_WIDTH = 1280
TOTAL_VID_HEIGHT = 720

# 各画面の解像度
MAIN_VIEW_W = 640
MAIN_VIEW_H = 720
SUB_VIEW_W = 640
SUB_VIEW_H = 360

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    str(output_video_path), fourcc, 20.0, (TOTAL_VID_WIDTH, TOTAL_VID_HEIGHT)
)
print(f"3-View Video export enabled: {output_video_path}")


def render_wireframe_mesh_view(
    robot_position: list[float],
    scan: dict[str, Any],
    obstacle_memory: dict[int, dict[str, Any]],
    wireframe_enabled: bool,
) -> np.ndarray:
    """3画面目: Wキーでメッシュ/マップ表示に切り替わるワイヤーフレームビジュアライザ画面"""
    # キャンバス初期化 (ダークグレー背景)
    canvas = np.full((SUB_VIEW_H, SUB_VIEW_W, 3), 30, dtype=np.uint8)
    
    center_x, center_y = SUB_VIEW_W // 2, SUB_VIEW_H // 2
    scale = 25.0  # 1メートルあたりのピクセル数

    if wireframe_enabled:
        # --- メッシュ (グリッド) 描画 ---
        grid_size = 20
        grid_color = (60, 60, 60)
        for i in range(-15, 16):
            # 縦線
            px = int(center_x + i * scale)
            cv2.line(canvas, (px, 0), (px, SUB_VIEW_H), grid_color, 1)
            # 横線
            py = int(center_y + i * scale)
            cv2.line(canvas, (0, py), (SUB_VIEW_W, py), grid_color, 1)

        rx, ry = robot_position[0], robot_position[1]

        # 障害物 (木) のメッシュ位置を描画
        for obs in obstacle_memory.values():
            ox, oy = obs["xy"]
            screen_x = int(center_x + (ox - rx) * scale)
            screen_y = int(center_y - (oy - ry) * scale)
            cv2.circle(canvas, (screen_x, screen_y), int(step5.TRUNK_RADIUS * scale), (0, 255, 255), 1)

        # LiDARスキャン点群のメッシュ線を描画
        for ray in scan.get("rays", []):
            hx, hy, _ = ray["hit_position"]
            sx = int(center_x + (hx - rx) * scale)
            sy = int(center_y - (hy - ry) * scale)
            cv2.line(canvas, (center_x, center_y), (sx, sy), (0, 180, 0), 1)

        # ローバーの位置 (中心)
        cv2.rectangle(canvas, (center_x - 8, center_y - 8), (center_x + 8, center_y + 8), (0, 165, 255), 2)
        cv2.putText(canvas, "MODE: MESH / WIREFRAME (W-KEY: ON)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    else:
        # WキーがOFFの時の表示
        cv2.putText(canvas, "MODE: NORMAL (Press 'W' to enable Mesh View)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return canvas


# ============================================================
# Main
# ============================================================


def main() -> None:
    model, class_names, model_path = step3.load_detector()
    if step4.EVENT_CLASS_IDS is not None:
        unknown_ids = step4.EVENT_CLASS_IDS - set(class_names)
        if unknown_ids:
            raise ValueError(
                "STEP4.EVENT_CLASS_IDS contains IDs missing from the loaded model: "
                f"{sorted(unknown_ids)}"
            )

    print(f"Selected checkpoint: {model_path}")
    print(f"YOLO: CPU, {step3.YOLO_IMAGE_SIZE}x{step3.YOLO_IMAGE_SIZE}, "
          f"target {step3.INFERENCE_FPS:.1f} FPS")
    print("\nSTEP6 v2: Footprint-aware local detour patrol")
    print(
        f"  footprint envelope: {2 * ROBOT_HALF_LENGTH:.3f}m x "
        f"{2 * ROBOT_HALF_WIDTH:.3f}m (including tire width)"
    )
    print(
        f"  LiDAR: {len(LIDAR_ANGLES_DEG)} rays / 360deg / "
        f"{LIDAR_MAX_RANGE:.1f}m / {1.0 / LIDAR_UPDATE_INTERVAL_SEC:.1f}Hz"
    )
    print(
        f"  detour side offset={AVOID_SIDE_OFFSET:.2f}m, "
        f"before/after={AVOID_BEFORE_TRUNK_OFFSET:.2f}m, "
        f"safety margin={COLLISION_SAFETY_MARGIN:.2f}m"
    )
    print("  priority: OBSERVE > AVOID > PATROL")
    print("  Q: quit (focus the PyBullet window first)\n")

    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Could not connect to the PyBullet GUI.")

    camera_window_created = False
    try:
        # World
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81)
        p.setTimeStep(TIME_STEP)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(numSolverIterations=100)
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

        # STEP5 forest and optional route-test tree
        trunk_ids, crown_ids, tree_positions = step5.create_forest()
        if ADD_ROUTE_TEST_TREE:
            test_trunk_id, test_crown_id = create_route_test_tree(
                ROUTE_TEST_TREE_POSITION
            )
            trunk_ids.append(test_trunk_id)
            crown_ids.append(test_crown_id)

        trunk_body_ids = set(trunk_ids)
        print(f"LiDAR targets: {len(trunk_body_ids)} tree trunks")

        # STEP3 animal and camera
        #animal_body_id = step3.create_animal_billboard(step3.ANIMAL_IMAGE_PATH)
        animal_body_ids = step3.create_animal_billboards(
                                    step3.ANIMAL_IMAGE_PATHS,
                                    step3.ANIMAL_POSITIONS,
                                    step3.ANIMAL_FACINGS
                                )
        print(f"Animal billboard body={animal_body_ids}; visual only, no collision")

        p.resetDebugVisualizerCamera(
            cameraDistance=OBSERVER_CAMERA_DISTANCE,
            cameraYaw=OBSERVER_CAMERA_YAW,
            cameraPitch=OBSERVER_CAMERA_PITCH,
            cameraTargetPosition=[3.0, 0.0, 0.8],
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=step3.CAMERA_FOV,
            aspect=step3.CAMERA_WIDTH / step3.CAMERA_HEIGHT,
            nearVal=step3.CAMERA_NEAR,
            farVal=step3.CAMERA_FAR,
        )
        cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAMERA_WINDOW_NAME, step3.CAMERA_WIDTH, step3.CAMERA_HEIGHT)
        camera_window_created = True

        # State and event management
        state = step4.RoverState.PATROL
        resume_state = step4.RoverState.PATROL
        observe_until = 0.0
        event_latched = False
        positive_detection_streak = 0
        missing_detection_streak = 0
        last_event_message = "none"

        avoid_context: dict[str, Any] | None = None
        avoid_failed = False
        avoid_cooldown_until = 0.0
        waypoint_index = 0
        obstacle_memory: dict[int, dict[str, Any]] = {}

        # Simulated motion time pauses while OBSERVE is active.
        motion_time = 0.0
        simulation_steps = 0

        # Camera / inference / LiDAR scheduling
        last_detections: list[dict[str, Any]] = []
        last_inference_ms: float | None = None
        last_inference_finished: float | None = None
        inference_interval = 1.0 / max(step3.INFERENCE_FPS, 0.1)
        next_inference_time = 0.0

        latest_scan = scan_virtual_lidar(robot_id, trunk_body_ids)
        update_obstacle_memory(
            latest_scan,
            obstacle_memory,
            (0.0, 0.0),
            time.perf_counter(),
        )
        next_lidar_time = time.perf_counter() + LIDAR_UPDATE_INTERVAL_SEC
        frame_count = 0

        def fail_avoidance(reason: str) -> None:
            nonlocal state, avoid_failed
            avoid_failed = True
            state = step4.RoverState.AVOID
            set_drive_command(robot_id, wheel_joints, 0.0, 0.0)
            x, y, yaw = get_robot_xy_yaw(robot_id)
            print(f"[AVOID][STOP] {reason}")
            print(
                f"[AVOID][STOP] pose=({x:.2f}, {y:.2f}, "
                f"{math.degrees(yaw):.1f}deg), "
                f"front={latest_scan['front_min_distance']:.2f}m, "
                f"local obstacles={len(obstacle_memory)}"
            )
            print("[AVOID][STOP] Rover remains stopped; press Q to exit.")

        def start_avoidance(
            obstacle_id: int,
            target_xy: tuple[float, float],
            replan: bool = False,
        ) -> None:
            nonlocal state, avoid_context, avoid_failed

            replan_count = (
                int(avoid_context["replan_count"]) + 1
                if replan and avoid_context is not None else 0
            )
            if replan_count > AVOID_MAX_REPLANS:
                fail_avoidance("Maximum local replans exceeded.")
                return

            plan, reason = create_avoid_plan(
                robot_id,
                target_xy,
                obstacle_id,
                obstacle_memory,
            )
            if plan is None:
                fail_avoidance(reason)
                return

            plan["phase_started_motion_time"] = motion_time
            plan["progress_check_time"] = motion_time
            plan["progress_metric"], plan["progress_is_angle"] = avoid_progress_metric(
                robot_id, plan
            )
            plan["replan_count"] = replan_count
            avoid_context = plan
            avoid_failed = False
            state = step4.RoverState.AVOID

            transition = "REPLAN" if replan else "PATROL -> AVOID"
            direction_text = "LEFT" if plan["direction"] > 0 else "RIGHT"
            print(
                f"[STATE] {transition} | tree={obstacle_id} | side={direction_text} | "
                f"min_clearance={plan['min_clearance']:.2f}m"
            )
            print(
                f"  TURN_OUT={math.degrees(plan['shift_yaw']):.1f}deg, "
                f"SHIFT={plan['shift_point']}, PASS={plan['pass_point']}"
            )

        def set_avoid_phase(phase: str) -> None:
            nonlocal avoid_context
            if avoid_context is None:
                return
            avoid_context["phase"] = phase
            avoid_context["phase_started_motion_time"] = motion_time
            avoid_context["progress_check_time"] = motion_time
            metric, is_angle = avoid_progress_metric(robot_id, avoid_context)
            avoid_context["progress_metric"] = metric
            avoid_context["progress_is_angle"] = is_angle
            set_drive_command(robot_id, wheel_joints, 0.0, 0.0)
            print(f"[AVOID] -> {phase}")

        def begin_observation(event: dict[str, Any]) -> None:
            """Enter OBSERVE and remember the exact state to resume afterward."""
            nonlocal state, resume_state, observe_until, last_event_message

            resume_state = state
            set_drive_command(robot_id, wheel_joints, 0.0, 0.0)
            state = step4.RoverState.OBSERVE
            observe_until = time.perf_counter() + step4.OBSERVE_DURATION_SEC

            try:
                info = step4.save_detection_event(
                    event["frame_bgr"],
                    event["detections"],
                    class_names,
                    event["simulation_time_sec"],
                    robot_id,
                )
                last_event_message = (
                    f"{info['primary_class_name']} {info['primary_confidence']} "
                    f"({info['event_id']})"
                )
                print("\n[WILDLIFE DETECTED]")
                print(f"  class: {info['primary_class_name']}")
                print(f"  confidence: {info['primary_confidence']}")
                print(f"  image: {info['image_path']}")
                print(f"  log: {step4.EVENT_LOG_PATH}")
            except Exception as exc:
                last_event_message = f"SAVE ERROR: {exc}"
                print(f"[ERROR] Could not save detection event: {exc}")

            print(
                f"[STATE] {resume_state.value} -> OBSERVE "
                f"({step4.OBSERVE_DURATION_SEC:.1f}s); will resume {resume_state.value}"
            )

        # Main loop
        while p.isConnected(client_id):
            now = time.perf_counter()
            key_events = p.getKeyboardEvents()
            if key_is_down(key_events, "q"):
                break

            if state != step4.RoverState.OBSERVE:
                motion_time += TIME_STEP

            # LiDAR and local obstacle cache remain active during OBSERVE.
            lidar_updated = False
            if now >= next_lidar_time:
                latest_scan = scan_virtual_lidar(robot_id, trunk_body_ids)
                robot_x, robot_y, _ = get_robot_xy_yaw(robot_id)
                update_obstacle_memory(
                    latest_scan,
                    obstacle_memory,
                    (robot_x, robot_y),
                    now,
                )
                draw_lidar_rays(latest_scan)
                draw_robot_footprint(robot_id)
                draw_avoid_plan(avoid_context)
                next_lidar_time = now + LIDAR_UPDATE_INTERVAL_SEC
                lidar_updated = True

            # Finish observation and restore PATROL or the preserved AVOID phase.
            if state == step4.RoverState.OBSERVE and now >= observe_until:
                state = resume_state
                positive_detection_streak = 0
                missing_detection_streak = 0
                print(f"[STATE] OBSERVE -> {state.value}")

            # Avoidance is initiated only for an actual footprint collision on
            # the near waypoint route (or an unsafe in-place turn).
            if state == step4.RoverState.PATROL and not avoid_failed:
                target = PATROL_WAYPOINTS[waypoint_index]
                blocker_id = find_first_route_blocker(
                    robot_id,
                    target,
                    obstacle_memory,
                )
                if (
                    blocker_id is not None
                    and now >= avoid_cooldown_until
                ):
                    start_avoidance(blocker_id, target)

            # Replan only when a newly observed trunk intersects the remainder
            # of the active plan. Other trees entering the camera/LiDAR view do
            # not automatically restart the detour.
            elif (
                state == step4.RoverState.AVOID
                and not avoid_failed
                and avoid_context is not None
                and lidar_updated
            ):
                blocker_id = find_remaining_plan_blocker(
                    robot_id,
                    avoid_context,
                    obstacle_memory,
                )
                if blocker_id is not None:
                    print(
                        f"[AVOID] New/updated trunk {blocker_id} intersects the "
                        "remaining detour; replan from current pose."
                    )
                    start_avoidance(
                        blocker_id,
                        PATROL_WAYPOINTS[waypoint_index],
                        replan=True,
                    )

            # Apply exactly one state controller per simulation step.
            if state == step4.RoverState.PATROL:
                target = PATROL_WAYPOINTS[waypoint_index]
                distance, linear, angular = patrol_drive_command(robot_id, target)
                if distance <= WAYPOINT_REACHED_DISTANCE:
                    print(f"[PATROL] Reached waypoint {waypoint_index + 1}: {target}")
                    waypoint_index = (waypoint_index + 1) % len(PATROL_WAYPOINTS)
                    set_drive_command(robot_id, wheel_joints, 0.0, 0.0)
                else:
                    set_drive_command(robot_id, wheel_joints, linear, angular)

            elif state == step4.RoverState.AVOID:
                if avoid_failed or avoid_context is None:
                    set_drive_command(robot_id, wheel_joints, 0.0, 0.0)
                else:
                    phase = str(avoid_context["phase"])
                    phase_elapsed = (
                        motion_time - float(avoid_context["phase_started_motion_time"])
                    )
                    phase_timeout = AVOID_PHASE_TIMEOUTS[phase]

                    if phase_elapsed > phase_timeout:
                        fail_avoidance(
                            f"Phase {phase} timed out after {phase_elapsed:.1f}s."
                        )
                    else:
                        metric, is_angle = avoid_progress_metric(
                            robot_id, avoid_context
                        )
                        check_elapsed = (
                            motion_time - float(avoid_context["progress_check_time"])
                        )
                        if check_elapsed >= AVOID_STALL_CHECK_INTERVAL:
                            previous_metric = float(avoid_context["progress_metric"])
                            minimum_progress = (
                                AVOID_STALL_MIN_ANGLE_PROGRESS
                                if is_angle else AVOID_STALL_MIN_POSITION_PROGRESS
                            )
                            progress = previous_metric - metric
                            if progress < minimum_progress:
                                fail_avoidance(
                                    f"Phase {phase} made insufficient progress "
                                    f"({progress:.3f} in {check_elapsed:.1f}s)."
                                )
                            else:
                                avoid_context["progress_metric"] = metric
                                avoid_context["progress_check_time"] = motion_time

                        if not avoid_failed:
                            if phase == "TURN_OUT":
                                aligned = command_rotation_to_heading(
                                    robot_id,
                                    wheel_joints,
                                    float(avoid_context["shift_yaw"]),
                                    AVOID_TURN_TOLERANCE_DEG,
                                )
                                if aligned:
                                    set_avoid_phase("SHIFT")

                            elif phase == "SHIFT":
                                x, y, _ = get_robot_xy_yaw(robot_id)
                                shift_point = avoid_context["shift_point"]
                                distance = math.hypot(
                                    shift_point[0] - x,
                                    shift_point[1] - y,
                                )
                                if distance <= AVOID_SHIFT_TOLERANCE:
                                    set_avoid_phase("TURN_ALIGN")
                                else:
                                    drive_along_fixed_heading(
                                        robot_id,
                                        wheel_joints,
                                        float(avoid_context["shift_yaw"]),
                                        AVOID_SHIFT_SPEED,
                                    )

                            elif phase == "TURN_ALIGN":
                                aligned = command_rotation_to_heading(
                                    robot_id,
                                    wheel_joints,
                                    float(avoid_context["entry_yaw"]),
                                    AVOID_ALIGN_TOLERANCE_DEG,
                                )
                                if aligned:
                                    set_avoid_phase("PASS")

                            elif phase == "PASS":
                                x, y, _ = get_robot_xy_yaw(robot_id)
                                pass_point = avoid_context["pass_point"]
                                distance = math.hypot(
                                    pass_point[0] - x,
                                    pass_point[1] - y,
                                )
                                if distance <= AVOID_PASS_TOLERANCE:
                                    completed_obstacle = avoid_context["obstacle_id"]
                                    avoid_context = None
                                    state = step4.RoverState.PATROL
                                    avoid_cooldown_until = now + AVOID_COOLDOWN_SEC
                                    set_drive_command(robot_id, wheel_joints, 0.0, 0.0)
                                    print(
                                        f"[STATE] AVOID -> PATROL | "
                                        f"passed tree {completed_obstacle}"
                                    )
                                else:
                                    drive_along_fixed_heading(
                                        robot_id,
                                        wheel_joints,
                                        float(avoid_context["entry_yaw"]),
                                        AVOID_PASS_SPEED,
                                    )

            elif state == step4.RoverState.OBSERVE:
                set_drive_command(robot_id, wheel_joints, 0.0, 0.0)

            # Physics update
            p.stepSimulation()
            simulation_steps += 1

            # Last-resort safety interlock: do not keep driving into a tree if
            # an unexpected collision occurs despite the geometric prediction.
            if state != step4.RoverState.OBSERVE and not avoid_failed:
                contacted_tree = None
                for trunk_id in trunk_body_ids:
                    if p.getContactPoints(bodyA=robot_id, bodyB=trunk_id):
                        contacted_tree = trunk_id
                        break
                if contacted_tree is not None:
                    fail_avoidance(
                        f"Unexpected physical contact with tree {contacted_tree}; "
                        "drive command stopped immediately."
                    )

            # Camera update and YOLO inference
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
                camera_bgr = step3.render_front_camera(robot_id, projection_matrix)

                if time.perf_counter() >= next_inference_time:
                    last_detections, last_inference_ms = step3.run_inference(
                        model, camera_bgr
                    )
                    last_inference_finished = time.perf_counter()
                    next_inference_time = last_inference_finished + inference_interval
                    event_detections = step4.select_event_detections(last_detections)

                    if state in (step4.RoverState.PATROL, step4.RoverState.AVOID):
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
                                        "[EVENT] Target disappeared; "
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
                            event = {
                                "frame_bgr": camera_bgr.copy(),
                                "detections": [dict(item) for item in event_detections],
                                "simulation_time_sec": simulation_steps * TIME_STEP,
                            }
                            begin_observation(event)

                inference_age = (
                    None if last_inference_finished is None
                    else time.perf_counter() - last_inference_finished
                )
                observe_remaining = (
                    max(0.0, observe_until - time.perf_counter())
                    if state == step4.RoverState.OBSERVE else 0.0
                )
                camera_display = draw_status(
                    camera_bgr,
                    last_detections,
                    class_names,
                    state,
                    resume_state,
                    last_inference_ms,
                    inference_age,
                    observe_remaining,
                    last_event_message,
                    event_latched,
                    waypoint_index,
                    latest_scan,
                    avoid_context,
                    avoid_failed,
                    motion_time,
                    len(obstacle_memory),
                )
                cv2.imshow(CAMERA_WINDOW_NAME, camera_display)

                camera_display = draw_status(
                    camera_bgr,
                    last_detections,
                    class_names,
                    state,
                    resume_state,
                    last_inference_ms,
                    inference_age,
                    observe_remaining,
                    last_event_message,
                    event_latched,
                    waypoint_index,
                    latest_scan,
                    avoid_context,
                    avoid_failed,
                    motion_time,
                    len(obstacle_memory),
                )
                cv2.imshow(CAMERA_WINDOW_NAME, camera_display)

                # ========================================================
                # 3画面レイアウトの動画フレーム作成
                # ========================================================
                # 1. 左側: メインビュー (斜め上からの俯瞰, 640x720)
                obs_target = [robot_position[0] + 1.3, robot_position[1], 0.8]
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=obs_target,
                    distance=OBSERVER_CAMERA_DISTANCE,
                    yaw=OBSERVER_CAMERA_YAW,
                    pitch=OBSERVER_CAMERA_PITCH,
                    roll=0.0,
                    upAxisIndex=2,
                )
                proj_matrix_obs = p.computeProjectionMatrixFOV(
                    fov=60.0, aspect=MAIN_VIEW_W / MAIN_VIEW_H, nearVal=0.1, farVal=100.0
                )
                _, _, rgb_obs, _, _ = p.getCameraImage(
                    width=MAIN_VIEW_W,
                    height=MAIN_VIEW_H,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix_obs,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                )
                main_view_bgr = cv2.cvtColor(rgb_obs, cv2.COLOR_RGBA2BGR)

                # 2. 右上: ローバーカメラ画面 (YOLO描画付き, 640x360)
                sub_rover_cam = cv2.resize(camera_display, (SUB_VIEW_W, SUB_VIEW_H))

                # 3. 右下: メッシュ/ワイヤーフレーム画面 (640x360)
                sub_mesh_view = render_wireframe_mesh_view(
                    robot_position, latest_scan, obstacle_memory, wireframe_mode
                )

                # 4. レイアウトの結合 (右側2画面を縦結合し、左側のメイン画面と横結合)
                right_side_combined = cv2.vconcat([sub_rover_cam, sub_mesh_view])
                final_3view_frame = cv2.hconcat([main_view_bgr, right_side_combined])

                # 動画ファイルへ書き込み & GUI表示
                video_writer.write(final_3view_frame)
                cv2.imshow(CAMERA_WINDOW_NAME, camera_display)

            cv2.waitKey(1)
            frame_count += 1
            time.sleep(TIME_STEP)

    finally:
        # 動画ライターを解放
        if 'video_writer' in locals() and video_writer.isOpened():
            video_writer.release()
            print(f"Video saved successfully to: {output_video_path}")
        if camera_window_created:
            cv2.destroyAllWindows()
        if p.isConnected(client_id):
            p.disconnect(client_id)


if __name__ == "__main__":
    main()
