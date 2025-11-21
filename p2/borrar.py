import time
import pickle
import gymnasium as gym
import numpy as np
import neat
from robobo_utils import RoboboEnv
import robobo_utils.neat_visualize as visualize
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim

ip = "localhost"
rob = Robobo(ip)
sim = RoboboSim(ip)

rob.connect()
sim.connect()

print(sim.getObjectLocation("CYLINDERBALL"))