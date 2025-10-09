import random, time, math
import numpy as np
import gymnasium as gym
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from math import dist
from stable_baselines3 import PPO
from pr1 import *

def main():
    gym.register(
        id="RoboboEnv",
        entry_point=RoboboEnv,
    )

    env = gym.make("RoboboEnv")
    
    model = PPO("MultiInputPolicy", env, verbose=1)

    model = model.load("checkpoint.zip", env=env)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)


if __name__ == "__main__":
    main()

