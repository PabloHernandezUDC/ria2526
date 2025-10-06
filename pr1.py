import random, time
import numpy as np
import gymnasium as gym
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from math import dist
from stable_baselines3 import A2C

class RoboboEnv(gym.Env):

    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                # "agent": gym.spaces.Box(-1000, 1000, shape=(3,), dtype=int),   # [x, z] coordinates and [y] rotation
                # "target": gym.spaces.Box(-1000, 1000, shape=(2,), dtype=int),  # [x, z] coordinates
                "red_x": gym.spaces.Box(0, 100, shape=(1,), dtype=int)
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        speed = 20
        self._action_to_direction = {
            0: np.array([speed, speed]),    # Forward
            1: np.array([0, speed]),        # Turn Left
            2: np.array([speed, 0]),        # Turn Right
            3: np.array([-speed, speed]),   # Go Backwards
        }
        
        ip = "localhost"
        self.robobo = Robobo(ip)
        self.sim = RoboboSim(ip)
        self.robobo.connect()
        self.sim.connect()
                
        self.target_pos = get_cylinder_pos(self.sim)
        self.target_color = BlobColor.RED
            
    def step(self, action):
        l_speed, r_speed = self._action_to_direction[action]
        
        duration = 0.5 # habría que adaptarlo si se acelera el simulador
        self.robobo.moveWheelsByTime(r_speed, l_speed, duration=duration, wait=False)
        time.sleep(duration)

        observation = self._get_obs()

        distance = get_distance_to_target(
            get_robot_pos(self.sim),
            self.target_pos
        )
        
        reward = get_reward(distance)

        terminated = distance <= 50
        
        # TODO
        truncated = False   
        
        info = self._get_info()
        
        print(f"Action: {parse_action(action)} | Reward: {(reward):.3f} | Distance: {(distance):.3f} | Obs: {observation}")
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        print("Resetting env...")
        
        self.sim.resetSimulation()
        self.robobo.stopMotors()
        time.sleep(2) # sin esto no funciona (???????????)
        self.robobo.moveTiltTo(105, speed=20, wait=False)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def render(self):
        # TODO
        ...
    
    def close(self):
        self.robobo.disconnect()
        self.sim.disconnect()
    
    def _get_obs(self):
        return {"red_x": self.robobo.readColorBlob(self.target_color).posx}
    
    def _get_info(self):
        return {}
        

def parse_action(action: int):
    if action == 0:
        return "↑"
    if action == 1:
        return "←"
    if action == 2:
        return "→"
    if action == 3:
        return "↓"
    else:
        return "unknown"


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

def get_reward(distance: float):
    return 1000 / distance

def main():
    gym.register(
        id="RoboboEnv",
        entry_point=RoboboEnv,
    )

    env = gym.make("RoboboEnv")
    
    model = A2C("MultiInputPolicy", env, verbose=1) # en el enunciado usa MlpPolicy
    model.learn(total_timesteps=10000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        # vec_env.render("human")




if __name__ == "__main__":
    main()






'''
TODO:

¿hecho? hacer el espacio de estados a partir de lo que observa el robot (por ejemplo, la posicion del blob rojo) (ya está medio empezado)
¿hecho? hacer el espacio de acciones (probablemente solo alante, atrás, girar izquierda, girar derecha)

¿hecho? añadir pa escoger acción aleatoria
añadir función step

añadir todo lo relevante al env de gymnasium en la clase RoboboEnv (probablemente lo que hay puesto ahí no sirva para nada)

... y hacer el resto de la práctica


'''