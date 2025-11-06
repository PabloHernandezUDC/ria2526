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
            - "ir_front_c": Discrete(4) representing front-center IR (0=far, 1=medium-far, 2=medium-close, 3=very close)
            - "ir_front_r": Discrete(2) representing front-right IR (0=far >25, 1=close <25)
            - "ir_front_l": Discrete(2) representing front-left IR (0=far >25, 1=close <25)
            - "ir_right": Discrete(2) representing right IR (0=far >25, 1=close <25)
            - "ir_left": Discrete(2) representing left IR (0=far >25, 1=close <25)
            - "ir_back_c": Discrete(2) representing back-center IR (0=far >25, 1=close <25)
            - "ir_back_r": Discrete(2) representing back-right IR (0=far >25, 1=close <25)
            - "ir_back_l": Discrete(2) representing back-left IR (0=far >25, 1=close <25)
        
    Action Space:
        Discrete(3): 0=forward, 1=turn left, 2=turn right
    
    Args:
        verbose: If True, prints step information (default: True)
        target_name: Name of the target object in simulator (default: "CYLINDERMIDBALL")
        alpha: Weight for angle vs distance in reward calculation (default: 0.4)
        penalty_strength: Strength of penalty for approaching (0, 0) (default: 0.0, disabled)
    """

    def __init__(self, verbose=True, target_name="CYLINDERMIDBALL", alpha=0.4, penalty_strength=0.0):
        # Observation space: visual sector (0-5) + IR sensor sectors
        self.observation_space = gym.spaces.Dict(
            {
                "sector": gym.spaces.Discrete(6),
                "ir_front_c": gym.spaces.Discrete(4),
                "ir_front_r": gym.spaces.Discrete(2),
                "ir_front_l": gym.spaces.Discrete(2),
                "ir_right": gym.spaces.Discrete(2),
                "ir_left": gym.spaces.Discrete(2),
                "ir_back_c": gym.spaces.Discrete(2),
                "ir_back_r": gym.spaces.Discrete(2),
                "ir_back_l": gym.spaces.Discrete(2)
            }
        )

        # Action space: 3 discrete actions
        self.action_space = gym.spaces.Discrete(3)

        # Map actions to wheel speeds
        speed = 5
        self._action_to_direction = {
            0: np.array([speed*4, speed*4]),   # Forward
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
        self.penalty_strength = penalty_strength

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
        robot_pos = get_robot_pos(self.sim)
        distance = get_distance_to_target(
            robot_pos,
            self.target_pos
        )

        angle = get_angle_to_target(
            robot_pos,
            self.target_pos
        )

        # Calculate reward
        reward = get_reward(distance, angle, alpha=self.alpha, robot_pos=robot_pos, penalty_strength=self.penalty_strength)

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
            Dictionary with "sector" and IR sensor keys indicating where target 
            is visible and obstacle proximity on all 8 IR sensors (binary: close/far)
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
            front_c_ir = ir_sensors.get(IR.FrontC.value, 0)
            front_r_ir = ir_sensors.get(IR.FrontR.value, 0)
            front_l_ir = ir_sensors.get(IR.FrontL.value, 0)
            right_ir = ir_sensors.get(IR.FrontRR.value, 0)
            left_ir = ir_sensors.get(IR.FrontLL.value, 0)
            back_c_ir = ir_sensors.get(IR.BackC.value, 0)
            back_r_ir = ir_sensors.get(IR.BackR.value, 0)
            back_l_ir = ir_sensors.get(IR.BackL.value, 0)
            
            # Front-center: 4-state discretization (0=far, 1=medium-far, 2=medium-close, 3=very close)
            if front_c_ir < 10:
                ir_front_c_sector = 0
            elif front_c_ir < 25:
                ir_front_c_sector = 1
            elif front_c_ir < 50:
                ir_front_c_sector = 2
            else:
                ir_front_c_sector = 3
            
            # Other sensors: Binary discretization (0=far >=25, 1=close <25)
            ir_front_r_sector = 1 if front_r_ir < 25 else 0
            ir_front_l_sector = 1 if front_l_ir < 25 else 0
            ir_right_sector = 1 if right_ir < 25 else 0
            ir_left_sector = 1 if left_ir < 25 else 0
            ir_back_c_sector = 1 if back_c_ir < 25 else 0
            ir_back_r_sector = 1 if back_r_ir < 25 else 0
            ir_back_l_sector = 1 if back_l_ir < 25 else 0
            
            # print(f"back_l {back_l_ir} | back_c {back_c_ir} | back_r {back_r_ir}")
        else:
            ir_front_c_sector = 0
            ir_front_r_sector = 0
            ir_front_l_sector = 0
            ir_right_sector = 0
            ir_left_sector = 0
            ir_back_c_sector = 0
            ir_back_r_sector = 0
            ir_back_l_sector = 0

        return {
            "sector": np.array([sector], dtype=int).flatten(),
            "ir_front_c": np.array([ir_front_c_sector], dtype=int).flatten(),
            "ir_front_r": np.array([ir_front_r_sector], dtype=int).flatten(),
            "ir_front_l": np.array([ir_front_l_sector], dtype=int).flatten(),
            "ir_right": np.array([ir_right_sector], dtype=int).flatten(),
            "ir_left": np.array([ir_left_sector], dtype=int).flatten(),
            "ir_back_c": np.array([ir_back_c_sector], dtype=int).flatten(),
            "ir_back_r": np.array([ir_back_r_sector], dtype=int).flatten(),
            "ir_back_l": np.array([ir_back_l_sector], dtype=int).flatten()
        }

    def _get_info(self):
        """Get additional information about current step"""
        return {}
