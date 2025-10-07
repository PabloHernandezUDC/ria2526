import random, time, math
import numpy as np
import gymnasium as gym
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from math import dist

from pr1 import *

ip = "localhost"
sim = RoboboSim(ip)
rob = Robobo(ip)
sim.connect()
rob.connect()



def get_stuff():
    r = get_robot_pos(sim)
    t = get_cylinder_pos(sim)
    a = get_angle_to_target(r, t)


time.sleep(0.5)

get_stuff()

rob.moveWheelsByTime(0, 20, 1.85)

get_stuff()

rob.moveWheelsByTime(-20, -20, 6)

get_stuff()

rob.moveWheelsByTime(20, 0, 1.5)

get_stuff()