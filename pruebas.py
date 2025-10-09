import random, time, math
import numpy as np
import gymnasium as gym
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from math import dist

from train import *

# ip = "localhost"
# sim = RoboboSim(ip)
# rob = Robobo(ip)
# sim.connect()
# rob.connect()

# print(sim.getRobots())

# for _ in range(1000):
#     angle = random.random()*360
#     angle = int(round(angle, 0))
#     data = sim.getRobotLocation(0)
#     data["rotation"]["y"] = angle
#     sim.setRobotLocation(0, rotation={})

for i in range(1, 101):
    print(i, i // 20)