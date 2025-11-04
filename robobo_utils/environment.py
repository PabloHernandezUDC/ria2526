import time
import numpy as np
import gymnasium as gym
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from robobopy.utils.IR import IR

from .helpers import (
    get_robot_pos,
    get_cylinder_pos,
    get_distance_to_target,
    get_angle_to_target,
    get_reward,
    parse_action
)


class RoboboEnv(gym.Env):
    """
    Custom Gymnasium environment for Robobo robot.
    
    The robot must navigate towards a red cylinder visible using only
    visual information (sector of the field of view where the target appears)
    and IR sensor information (discretized into distance sectors).
    
    Observation Space:
        Dict with keys:
            - "sector": Discrete(6) representing visual sectors (0-5)
            - "ir_front": Discrete(4) representing front IR distance (0=far, 3=very close)
        
    Action Space:
        Discrete(3): 0=forward, 1=turn left, 2=turn right
    
    Args:
        verbose: If True, prints step information (default: True)
        target_name: Name of the target object in simulator (default: "CYLINDERMIDBALL")
        alpha: Weight for angle vs distance in reward calculation (default: 0.4)
    """

    def __init__(self, verbose=True, target_name="CYLINDERMIDBALL", alpha=0.4):
        # Observation space: visual sector (0-5) + IR sensor sectors (0-3)
        self.observation_space = gym.spaces.Dict(
            {
                "sector": gym.spaces.Discrete(6),
                "ir_front": gym.spaces.Discrete(4)
            }
        )

        # Action space: 3 discrete actions
        self.action_space = gym.spaces.Discrete(3)

        # Map actions to wheel speeds
        speed = 10
        self._action_to_direction = {
            0: np.array([speed*3, speed*3]),   # Forward
            1: np.array([0, speed]),         # Turn Left
            2: np.array([speed, 0]),         # Turn Right
        }

        # Connect to RoboboSim
        ip = "localhost"
        self.robobo = Robobo(ip)
        self.sim = RoboboSim(ip)
        self.robobo.connect()
        self.sim.connect()

        self.target_name = target_name
        self.target_pos = get_cylinder_pos(self.sim, self.target_name)
        self.target_color = BlobColor.RED
        self.steps_without_target = 0
        self.verbose = verbose
        self.alpha = alpha

    def step(self, action):
        """
        Execute action and return result.
        
        Args:
            action: Integer action (0=forward, 1=left, 2=right)
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether episode ended successfully
            truncated: Whether episode was cut off
            info: Additional information dictionary
        """
        l_speed, r_speed = self._action_to_direction[action]

        duration = 0.5
        self.robobo.moveWheelsByTime(
            r_speed, l_speed, duration=duration, wait=True)
        time.sleep(.01)

        observation = self._get_obs()

        # Counter for steps without seeing target
        if observation["sector"] == 5:
            self.steps_without_target += 1
        else:
            self.steps_without_target = 0

        # Calculate distance and angle to target
        distance = get_distance_to_target(
            get_robot_pos(self.sim),
            self.target_pos
        )

        angle = get_angle_to_target(
            get_robot_pos(self.sim),
            self.target_pos
        )

        # Calculate reward
        reward = get_reward(distance, angle, alpha=self.alpha)

        if self.verbose:
            print(f"Action: {parse_action(action)} | Reward: {(reward):.3f} | Distance: {(distance):.3f} | Obs: {observation}")

        # Check if target reached
        terminated = False
        if distance <= 100:
            if self.verbose:
                print(f"Target reached!")
            terminated = True
            reward += 90

        # Store additional information
        info = self._get_info()
        info["step_reward"] = reward
        info["robot_pos"] = get_robot_pos(self.sim)

        # Check truncation (lost target for too long)
        truncated = False
        if self.steps_without_target >= 35:
            if self.verbose:
                print(f"Too many steps without seeing target!")
            truncated = True
            self.steps_without_target = 0
            reward -= 100

        info["is_truncated"] = truncated
        info["is_terminated"] = terminated

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed (optional)
            options: Additional options (optional)
            
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        if self.verbose:
            print("Resetting env...")

        self.sim.resetSimulation()
        self.robobo.stopMotors()
        time.sleep(.1)
        self.robobo.moveTiltTo(115, speed=20, wait=True)

        observation = self._get_obs()
        info = self._get_info()

        # self.target_pos = get_cylinder_pos(self.sim, self.target_name)

        return observation, info

    def render(self):
        """Render environment (not implemented)"""
        pass

    def close(self):
        """Close connections to simulator"""
        self.robobo.disconnect()
        self.sim.disconnect()

    def _get_obs(self):
        """
        Get current observation from environment.
        
        Returns:
            Dictionary with "sector" and "ir_front" keys indicating where 
            target is visible and front obstacle proximity
        """
        red_x = np.array([self.robobo.readColorBlob(self.target_color).posx])
        if red_x == 0:
            sector = 5
        elif red_x == 100:
            sector = 4
        else:
            sector = red_x // 20

        ir_sensors = self.robobo.readAllIRSensor()
        
        if ir_sensors:
            # Average the front IR sensors (FrontLL, FrontL, FrontC, FrontR, FrontRR)
            front_values = [
                ir_sensors.get(IR.FrontLL.value, 0),
                ir_sensors.get(IR.FrontL.value, 0),
                ir_sensors.get(IR.FrontC.value, 0),
                ir_sensors.get(IR.FrontR.value, 0),
                ir_sensors.get(IR.FrontRR.value, 0)
            ]
            avg_front_ir = np.mean(front_values)
            
            if avg_front_ir < 10:
                ir_sector = 0
            elif avg_front_ir < 25:
                ir_sector = 1
            elif avg_front_ir < 50:
                ir_sector = 2
            else:
                ir_sector = 3
        else:
            ir_sector = 0

        return {
            "sector": np.array([sector], dtype=int).flatten(),
            "ir_front": np.array([ir_sector], dtype=int).flatten()
        }

    def _get_info(self):
        """Get additional information about current step"""
        return {}
