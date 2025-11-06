"""
Helper functions for Robobo robot navigation and reward calculation.
"""

import math
from math import dist
from robobosim.RoboboSim import RoboboSim


def parse_action(action: int):
    """
    Convert action number to visual symbol.
    
    Args:
        action: Action number (0=forward, 1=left, 2=right)
        
    Returns:
        Unicode arrow symbol representing the action
    """
    if action == 0:
        return "↑"
    elif action == 1:
        return "←"
    elif action == 2:
        return "→"
    else:
        return "?"


def get_robot_pos(sim: RoboboSim):
    """
    Get current robot position and orientation.
    
    Args:
        sim: RoboboSim instance
        
    Returns:
        Dictionary with x, z positions and y rotation
    """
    data = sim.getRobotLocation(0)

    pos_x = data["position"]["x"]
    pos_z = data["position"]["z"]
    rot_y = data["rotation"]["y"]

    return {"x": pos_x, "z": pos_z, "y": rot_y}


def get_cylinder_pos(sim: RoboboSim, target_name: str = "CYLINDERMIDBALL"):
    """
    Get position of the target cylinder.
    
    Args:
        sim: RoboboSim instance
        target_name: Name of the target object in simulator (default: "CYLINDERMIDBALL")
        
    Returns:
        Dictionary with x, z positions of the cylinder
    """
    print(f"Getting location of object {target_name}...")
    try:
        data = sim.getObjectLocation(target_name)
        pos_x = data["position"]["x"]
        pos_z = data["position"]["z"]
    except TypeError:
        print("We have objects", sim.getObjects())
        quit()
    

    return {"x": pos_x, "z": pos_z}


def get_distance_to_target(robot_pos: dict, target_pos: dict):
    """
    Calculate Euclidean distance to target.
    
    Args:
        robot_pos: Robot position dictionary with 'x' and 'z' keys
        target_pos: Target position dictionary with 'x' and 'z' keys
        
    Returns:
        Euclidean distance to target
    """
    rx, rz = robot_pos["x"], robot_pos["z"]
    tx, tz = target_pos["x"], target_pos["z"]
    return dist((rx, rz), (tx, tz))


def get_angle_to_target(robot_pos: dict, target_pos: dict):
    """
    Calculate angle between robot orientation and direction to target.
    
    Args:
        robot_pos: Robot position dictionary with 'x', 'z', and 'y' keys
        target_pos: Target position dictionary with 'x' and 'z' keys
        
    Returns:
        Angle difference in degrees, normalized to [-180, 180]
    """
    rx, rz = robot_pos["x"], robot_pos["z"]
    tx, tz = target_pos["x"], target_pos["z"]

    dx = tx - rx
    dz = tz - rz

    target_angle = math.degrees(math.atan2(dx, dz))
    robot_angle = robot_pos["y"]
    angle_diff = target_angle - robot_angle

    # Normalize to range [-180, 180]
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360

    return angle_diff


def get_reward(distance: float, angle: float, alpha: float = 0.5):
    """
    Calculate multi-component reward.
    
    Combines distance-based reward (closer is better) with angle-based reward
    (facing target is better).
    
    Args:
        distance: Distance to target
        angle: Angle to target in degrees
        alpha: Weight of distance component (0-1), default 0.5
        
    Returns:
        Combined reward value
    """
    r1 = 1000 / distance
    r1 = min(5, r1)
    r2 = -(abs(angle) / 90)

    return (alpha) * r1 + (1 - alpha) * r2


def get_hybrid_reward(distance: float, angle: float, sector: int, ir_front: int, 
                       prev_distance: float = None, alpha: float = 0.5):
    """
    Calculate hybrid reward that combines obstacle avoidance with distance-based navigation.
    
    Strategy:
    - When obstacle is detected (ir_front >= 2): Penalize and encourage turning
    - When target not visible (sector == 5) but no obstacle: Use distance-based reward
    - When target visible and no obstacle: Strong distance + angle reward
    
    Args:
        distance: Current distance to target
        angle: Angle to target in degrees
        sector: Visual sector where target appears (5 = not visible)
        ir_front: Front IR sensor reading (0=far, 3=very close)
        prev_distance: Previous distance to target (for progress tracking)
        alpha: Weight for distance vs angle (default 0.5)
        
    Returns:
        Combined reward value
    """
    # Base distance reward component
    distance_reward = 1000 / max(distance, 1)
    distance_reward = min(5, distance_reward)
    
    # Angle reward component (penalize if not facing target)
    angle_reward = -(abs(angle) / 90)
    
    # Progress reward (if moving closer)
    progress_reward = 0
    if prev_distance is not None:
        distance_diff = prev_distance - distance
        progress_reward = distance_diff * 0.01  # Small bonus for getting closer
    
    # CASE 1: Obstacle detected (ir_front >= 2)
    if ir_front >= 2:
        # Strong penalty for being near obstacle
        obstacle_penalty = -2.0
        # Encourage turning when near obstacle
        turn_bonus = 0.5 if sector != 2 and sector != 3 else 0  # Bonus for not going straight
        return obstacle_penalty + turn_bonus + progress_reward
    
    # CASE 2: Target not visible (sector == 5) but no obstacle
    elif sector == 5:
        # Use distance-based navigation (like P1)
        # Reward getting closer even if target not visible
        return alpha * distance_reward + (1 - alpha) * angle_reward + progress_reward
    
    # CASE 3: Target visible and no obstacle - optimal situation
    else:
        # Strong reward for both distance and alignment
        visibility_bonus = 1.0  # Bonus for seeing target
        # Extra bonus if target is centered (sectors 2 or 3)
        centering_bonus = 0.5 if sector == 2 or sector == 3 else 0
        
        return (alpha * distance_reward + 
                (1 - alpha) * angle_reward + 
                visibility_bonus + 
                centering_bonus + 
                progress_reward)
