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


def get_cylinder_pos(sim: RoboboSim):
    """
    Get position of the target cylinder.
    
    Args:
        sim: RoboboSim instance
        
    Returns:
        Dictionary with x, z positions of the cylinder
    """
    data = sim.getObjectLocation("CYLINDERMIDBALL")

    pos_x = data["position"]["x"]
    pos_z = data["position"]["z"]

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
