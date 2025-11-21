"""
Custom callbacks for Stable Baselines3 training.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    """
    Custom callback to collect metrics during training.
    
    Records rewards, episode lengths, robot trajectories, and generates
    visualizations at the end of training.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        """Initialize lists to store metrics"""
        self.rewards = list()
        self.ep_lengths = list()
        self.current_episode_rewards = list()
        self.positions = list()
        self.current_episode_positions = list()
        self.episode_outcomes = list()

    def _on_step(self) -> bool:
        """Called after each environment step"""
        # Collect reward and position from current step
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            if "step_reward" in info:
                self.current_episode_rewards.append(info["step_reward"])
            if "robot_pos" in info:
                self.current_episode_positions.append(
                    (info["robot_pos"]["x"], info["robot_pos"]["z"]))

        # Process episode completion
        dones = self.locals.get("dones", [])
        if len(dones) > 0 and dones[0]:
            if self.current_episode_rewards:
                self.rewards.append(sum(self.current_episode_rewards))
                self.current_episode_rewards = list()
            if self.current_episode_positions:
                self.positions.append(self.current_episode_positions.copy())
                self.current_episode_positions = list()

            # Determine episode completion type
            infos = self.locals.get("infos", [])
            if len(infos) > 0:
                info = infos[0]
                if info.get("is_truncated", False):
                    self.episode_outcomes.append('truncated')
                elif info.get("is_terminated", False):
                    self.episode_outcomes.append('terminated')
                else:
                    self.episode_outcomes.append('other')
            else:
                self.episode_outcomes.append('other')

        return True

    def _on_training_end(self) -> None:
        """Generate visualizations at the end of training"""
        # Plot rewards with moving average
        if self.rewards:
            plt.clf()
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot raw rewards
            episodes = np.arange(len(self.rewards))
            ax.plot(episodes, self.rewards, alpha=0.3,
                    label='Episode Reward', color='blue')

            # Calculate and plot moving average
            window_size = 10
            if len(self.rewards) >= window_size:
                moving_avg = []
                for i in range(len(self.rewards)):
                    if i < window_size - 1:
                        # For early episodes, use average of all episodes so far
                        moving_avg.append(np.mean(self.rewards[:i+1]))
                    else:
                        # Use last 10 episodes
                        moving_avg.append(
                            np.mean(self.rewards[i-window_size+1:i+1]))

                ax.plot(episodes, moving_avg, linewidth=2,
                        label=f'Moving Average (last {window_size} episodes)', color='red')

            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.set_title("Rewards per Episode with Moving Average")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("plots/episode_rewards.jpg")
            plt.close()

            print(f"\nTotal episodes: {len(self.rewards)}")
            print(f"Mean reward: {np.mean(self.rewards):.2f}")
            print(f"Std reward: {np.std(self.rewards):.2f}")

        # Plot robot trajectories
        if self.positions:
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 10))

            num_episodes = len(self.positions)

            # Color gradient from red to green to indicate temporal progress
            for i, episode_positions in enumerate(self.positions):
                if len(episode_positions) > 0:
                    # Color: red (1, 0, 0) -> green (0, 1, 0)
                    ratio = i / max(1, num_episodes - 1)
                    color = (1 - ratio, ratio, 0)

                    # Extract x, z coordinates
                    xs = [pos[0] for pos in episode_positions]
                    zs = [pos[1] for pos in episode_positions]

                    # Draw trajectory
                    ax.plot(xs, zs, '-', color=color, alpha=0.6, linewidth=1)
                    
                    # Mark final position according to completion type
                    if i < len(self.episode_outcomes):
                        outcome = self.episode_outcomes[i]
                        if outcome == 'terminated':
                            # Star for success (target reached)
                            ax.plot(xs[-1], zs[-1], '*', color="gold", markersize=12,
                                    alpha=1, markeredgecolor='black', markeredgewidth=0.5)
                        elif outcome == 'truncated':
                            # Cross for truncated episode
                            ax.plot(xs[-1], zs[-1], 'x', color="red",
                                    markersize=8, alpha=1, markeredgewidth=2)
                    else:
                        # Default marker
                        ax.plot(xs[-1], zs[-1], 'o', color="black",
                                markersize=4, alpha=1)

            ax.set_xlim(-1000, 1000)
            ax.set_ylim(-1000, 1000)
            ax.set_xlabel("X Position")
            ax.set_ylabel("Z Position")
            ax.set_title(
                f"Robot Trajectories (Red=Early Episodes, Green=Late Episodes)")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            # Legend for markers
            legend_elements = [
                Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                       markeredgecolor='black', markersize=10, label='Target Found'),
                Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                       markeredgecolor='red', markersize=8, label='Truncated', markeredgewidth=2)
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            plt.savefig("plots/robot_trajectories.jpg", dpi=150)
            plt.close()

            print(f"Saved trajectory plot with {num_episodes} episodes")
