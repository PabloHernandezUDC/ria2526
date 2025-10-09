import random, time, math
import numpy as np
import gymnasium as gym
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from math import dist
from train import *
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        self.rewards = list()
        self.ep_lengths = list()
        self.current_episode_rewards = list()


    def _on_step(self) -> bool:
        # Access the info dict from the last step
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]  # For single environment
            if "step_reward" in info:
                self.current_episode_rewards.append(info["step_reward"])
        
        # Check if episode ended
        if len(self.locals.get("dones", [])) > 0:
            if self.locals["dones"][0]:  # Episode finished
                if self.current_episode_rewards:
                    self.rewards.append(sum(self.current_episode_rewards))
                    self.current_episode_rewards = list()
        
        return True


    def _on_training_end(self) -> None:
        # Plot the rewards
        if self.rewards:
            plt.clf()
            plt.plot(self.rewards)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Rewards per Episode")
            plt.savefig("episode_rewards.jpg")
            plt.close()
            
            print(f"\nTotal episodes: {len(self.rewards)}")
            print(f"Mean reward: {np.mean(self.rewards):.2f}")
            print(f"Std reward: {np.std(self.rewards):.2f}")


















def plot_moving_avg(data, title: str):
    mx = len(data)
    window_size = 100

    means = list()
    for i in range(window_size, mx):
        means.append(np.mean(data[i-window_size:i]))

    plt.clf()
    plt.plot(data)
    x = np.arange(window_size, mx)
    plt.plot(x, means)
    plt.savefig("borrar.jpg")


#  -----------------------------------------------------------------------------

# data = list()
# for i in range(mx):
#     data.append(random.random()*i)

#  -----------------------------------------------------------------------------

# def plot_evaluation_results():
#     data = np.load("eval_results/evaluations.npz")

#     timesteps = data["timesteps"]
#     rewards = data["results"]
#     ep_lengths = data["ep_lengths"]

#     mean_rewards = np.mean(rewards, axis=1)
#     std_rewards = np.std(rewards, axis=1)

#     mean_ep_lengths = np.mean(ep_lengths, axis=1)
#     std_ep_lengths = np.std(ep_lengths, axis=1)

#     for type, means, stds in (("reward", mean_rewards, std_rewards), ("episode_length", mean_ep_lengths, std_ep_lengths)):
#         plt.clf()
#         plt.errorbar(timesteps, means, yerr=stds, fmt="o-", capsize=4, color="black", ecolor="blue")
#         plt.xticks(timesteps)
#         plt.xlabel("Timesteps")
#         plt.ylabel(f"Mean {type}")
#         plt.suptitle(f"Mean and std. {type} over training")
#         plt.savefig(f"{type}s.jpg")

#  -----------------------------------------------------------------------------

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

# for i in range(1, 101):
#     print(i, i // 20)