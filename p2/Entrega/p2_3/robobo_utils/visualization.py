"""
Visualization utilities for training results.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_evaluation_results(eval_file="eval_results/evaluations.npz", output_dir="plots"):
    """
    Generate plots from periodic evaluations during training.
    
    Args:
        eval_file: Path to evaluation results .npz file
        output_dir: Directory to save plots
    """
    data = np.load(eval_file)

    timesteps = data["timesteps"]
    rewards = data["results"]
    ep_lengths = data["ep_lengths"]

    mean_rewards = np.mean(rewards, axis=1)
    std_rewards = np.std(rewards, axis=1)

    mean_ep_lengths = np.mean(ep_lengths, axis=1)
    std_ep_lengths = np.std(ep_lengths, axis=1)

    for type, means, stds in (("reward", mean_rewards, std_rewards), 
                               ("episode_length", mean_ep_lengths, std_ep_lengths)):
        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.errorbar(timesteps, means, yerr=stds, fmt="o-",
                    capsize=4, color="black", ecolor="blue")
        ax.set_xticks(timesteps)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(f"Mean {type}")
        fig.suptitle(f"Mean and std. {type} over training")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/eval_{type}s.jpg")
        plt.close()
