import random, time
import numpy as np
import gymnasium as gym
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from math import dist, log

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

def get_robot_pos(sim: RoboboSim):
    data = sim.getRobotLocation(0)
    
    pos_x = data["position"]["x"]
    pos_z = data["position"]["z"]    
    rot_y = data["rotation"]["y"]
    
    return {"x": pos_x, "z": pos_z, "y": rot_y}

def get_cylinder_pos(sim: RoboboSim):
    
    # IMPORTANTE: la posición del cilindro no se actualiza en tiempo real,
    # solo podemos saber la inicial
    
    data = sim.getObjectLocation("CYLINDERMIDBALL")

    pos_x = data["position"]["x"]
    pos_z = data["position"]["z"]
    
    return {"x": pos_x, "z": pos_z}

def get_distance_to_target(robot_pos: dict, target_pos: dict):
    rx, rz = robot_pos["x"], robot_pos["z"]
    tx, tz = target_pos["x"], target_pos["z"]
    return dist((rx, rz), (tx, tz))

def get_reward(robot_pos: dict, target_pos: dict):
    r = 1000 / get_distance_to_target(robot_pos, target_pos)
    return r


def get_robot_observation(rob: Robobo, target_color: BlobColor = BlobColor.RED):
    x = rob.readColorBlob(target_color).posx
    
    return x




def main():
    env = RoboboEnv()

    # Dirección IP del simulador
    ip = "localhost"

    # Inicializamos el robot y el simulador
    robobo = Robobo(ip)
    sim = RoboboSim(ip)
    robobo.connect()
    sim.connect()






    # robobo.moveWheels(15, 15)
    # for i in range(1000):
        # robot_pos = get_robot_pos(sim)
        # target_pos = get_cylinder_pos(sim)
        # print("robot is at", robot_pos)
        # print(f"distance to cylinder: {get_distance_to_target(robot_pos, target_pos)}")
        # print(f"reward: {get_reward(robot_pos, target_pos)}")

        # time.sleep(.1)




    time.sleep(1000)

    # env.close()


if __name__ == "__main__":
    main()

'''
TODO:

hacer el espacio de estados a partir de lo que observa el robot (por ejemplo, la posicion del blob rojo) (ya está medio empezado)
hacer el espacio de acciones (probablemente solo alante, atrás, girar izquierda, girar derecha)

añadir pa escoger acción aleatoria
añadir función step

añadir todo lo relevante al env de gymnasium en la clase RoboboEnv (probablemente lo que hay puesto ahí no sirva para nada)

... y hacer el resto de la práctica


'''