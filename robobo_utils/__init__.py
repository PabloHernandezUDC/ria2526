"""
Robobo Utilities Module

Shared components for Robobo robot reinforcement learning projects.
Includes custom environments, callbacks, helper functions, and visualization tools.
"""

from .environment import RoboboEnv
from .callbacks import CustomCallback
from .helpers import (
    parse_action,
    get_robot_pos,
    get_cylinder_pos,
    get_distance_to_target,
    get_angle_to_target,
    get_reward,
    get_hybrid_reward
)
from .visualization import plot_evaluation_results

__all__ = [
    'RoboboEnv',
    'CustomCallback',
    'parse_action',
    'get_robot_pos',
    'get_cylinder_pos',
    'get_distance_to_target',
    'get_angle_to_target',
    'get_reward',
    'get_hybrid_reward',
    'plot_evaluation_results'
]
