import random, time
import numpy as np
import gymnasium as gym
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim


class RoboboEnv(gym.Env):

    def __init__(self):
        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        # self._agent_location = np.array([-1, -1], dtype=np.int32)
        # self._target_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),   # [x, y] coordinates
        #         "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),  # [x, y] coordinates
        #     }
        # )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),  # Move down (negative y)
        }

'''
def discretize_state(state, bins, bounds):
    discrete = []
    for _, (val, (low, high), n_bins) in enumerate(zip(state, bounds, bins)):
        # Crear bins equiespaciados
        bin_width = (high - low) / n_bins
        bin_idx = int((val - low) / bin_width)
        # Mantener índices dentro del rango válido
        bin_idx = max(0, min(n_bins - 1, bin_idx))
        discrete.append(bin_idx)
    return tuple(discrete)

'''

def observe_robot(sim: RoboboSim):
    data = sim.getRobotLocation(0)
    
    # DEBUG
    print(data)
    # DEBUG
    
    pos_x = data["position"]["x"]
    # pos_y = data["position"]["y"]
    pos_z = data["position"]["z"]
    
    # rot_x = data["rotation"]["x"]
    rot_y = data["rotation"]["y"]
    # rot_z = data["rotation"]["z"]
    
    return ((pos_x, pos_z, rot_y))



def main():
    env = RoboboEnv()

    # Dirección IP del simulador
    ip = "localhost"

    # Inicializamos el robot y el simulador
    robobo = Robobo(ip)
    sim = RoboboSim(ip)
    robobo.connect()
    sim.connect()

    robobo.moveWheels(0, 15)

    for _ in range(1000):
        observe_robot(sim)
        time.sleep(.1)




    time.sleep(1000)

    # env.close()


if __name__ == "__main__":
    main()
