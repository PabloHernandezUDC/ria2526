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
