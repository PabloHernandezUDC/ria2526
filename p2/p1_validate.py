"""
Validation script for Practice 1 - Robobo Navigation with PPO

Validates a trained PPO model by running multiple episodes and generating
detailed statistics and visualizations.
"""

import random
import time
import sys
import io
import warnings
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from pathlib import Path

from robobo_utils import (
    RoboboEnv,
    parse_action,
    get_robot_pos,
    get_distance_to_target,
    get_angle_to_target
)

# Suppress Gymnasium warnings
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium')

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import locale
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

# Configure plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ValidationMetrics:
    """Class to store and manage validation metrics"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []  # Final distance to target
        self.episode_successes = []  # Whether target was reached
        self.episode_trajectories = []  # Robot (x, z) trajectories
        self.episode_target_positions = []  # Cylinder positions
        self.episode_step_rewards = []  # Step rewards for each episode
        self.episode_actions = []  # Actions taken in each episode
        
    def add_episode(self, total_reward, length, final_distance, success, 
                    trajectory, target_pos, step_rewards, actions):
        """Add complete episode data"""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        self.episode_distances.append(final_distance)
        self.episode_successes.append(success)
        self.episode_trajectories.append(trajectory)
        self.episode_target_positions.append(target_pos)
        self.episode_step_rewards.append(step_rewards)
        self.episode_actions.append(actions)
        
    def get_statistics(self):
        """Calculate descriptive statistics"""
        return {
            'num_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'median_reward': np.median(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'std_length': np.std(self.episode_lengths),
            'mean_final_distance': np.mean(self.episode_distances),
            'std_final_distance': np.std(self.episode_distances),
            'success_rate': np.mean(self.episode_successes) * 100,
            'num_successes': np.sum(self.episode_successes),
        }
        
    def save_to_file(self, filepath):
        """Save metrics to .npz file"""
        np.savez(
            filepath,
            episode_rewards=np.array(self.episode_rewards),
            episode_lengths=np.array(self.episode_lengths),
            episode_distances=np.array(self.episode_distances),
            episode_successes=np.array(self.episode_successes),
            episode_trajectories=np.array(self.episode_trajectories, dtype=object),
            episode_target_positions=np.array(self.episode_target_positions, dtype=object),
            episode_step_rewards=np.array(self.episode_step_rewards, dtype=object),
            episode_actions=np.array(self.episode_actions, dtype=object)
        )


def run_validation_episode(model, env, episode_num, deterministic=True, max_steps=200):
    """
    Run a complete validation episode
    
    Args:
        model: Trained StableBaselines3 model
        env: Gymnasium environment
        episode_num: Current episode number
        deterministic: If True, use deterministic policy (no exploration)
        max_steps: Maximum steps per episode
        
    Returns:
        Tuple with (total_reward, length, final_distance, success, trajectory,
                   target_pos, step_rewards, actions)
    """
    obs, info = env.reset()
    
    episode_reward = 0
    episode_length = 0
    trajectory = []
    step_rewards = []
    actions = []
    
    # Get initial target position
    actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    target_pos = actual_env.target_pos
    
    # Lists to store step information
    step_info = []
    
    for step in range(max_steps):
        # Get action from model
        action, _states = model.predict(obs, deterministic=deterministic)
        
        # Save current robot position
        actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        robot_pos = get_robot_pos(actual_env.sim)
        trajectory.append((robot_pos['x'], robot_pos['z']))
        actions.append(int(action))
        
        # Calculate distance and angle BEFORE step
        distance_before = get_distance_to_target(robot_pos, target_pos)
        angle_before = get_angle_to_target(robot_pos, target_pos)
        
        # Execute action (suppressing environment prints)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            obs, reward, terminated, truncated, info = env.step(int(action))
        finally:
            sys.stdout = old_stdout
        
        episode_reward += reward
        episode_length += 1
        step_rewards.append(reward)
        
        # Store step information
        step_info.append({
            'step': step + 1,
            'action': parse_action(int(action)),
            'reward': reward,
            'distance': distance_before,
            'angle': angle_before,
            'cyl_x': target_pos['x'],
            'cyl_z': target_pos['z']
        })
        
        done = terminated or truncated
        
        if done:
            # Save final position
            actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
            robot_pos = get_robot_pos(actual_env.sim)
            trajectory.append((robot_pos['x'], robot_pos['z']))
            
            final_distance = get_distance_to_target(robot_pos, target_pos)
            success = terminated and final_distance <= 100
            
            # Display episode summary table
            print(f"\n{'='*90}")
            print(f"{'EPISODE ' + str(episode_num):^90}")
            print(f"{'='*90}")
            print(f"{'Step':>5} | {'Action':>7} | {'Reward':>10} | {'Distance':>10} | {'Angle':>8} | {'Cyl.X':>8} | {'Cyl.Z':>8}")
            print("-"*90)
            for info_step in step_info:
                print(f"{info_step['step']:>5} | {info_step['action']:>7} | "
                      f"{info_step['reward']:>10.3f} | {info_step['distance']:>10.2f} | "
                      f"{info_step['angle']:>7.2f}° | {info_step['cyl_x']:>8.2f} | {info_step['cyl_z']:>8.2f}")
            print("-"*90)
            
            # Episode result
            if success:
                print(f"[SUCCESS] Robot reached target")
            elif truncated:
                print(f"[TRUNCATED] Too many steps without seeing target")
            else:
                print(f"[TERMINATED] Episode ended")
                
            print(f"Total reward: {episode_reward:.2f}")
            print(f"Length: {episode_length} steps")
            print(f"Final distance: {final_distance:.2f}")
            
            return (episode_reward, episode_length, final_distance, success, 
                    trajectory, target_pos, step_rewards, actions)
    
    # If maximum steps reached
    actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    robot_pos = get_robot_pos(actual_env.sim)
    final_distance = get_distance_to_target(robot_pos, target_pos)
    
    # Display episode summary table
    print(f"\n{'='*90}")
    print(f"{'EPISODE ' + str(episode_num):^90}")
    print(f"{'='*90}")
    print(f"{'Step':>5} | {'Action':>7} | {'Reward':>10} | {'Distance':>10} | {'Angle':>8} | {'Cyl.X':>8} | {'Cyl.Z':>8}")
    print("-"*90)
    for info_step in step_info:
        print(f"{info_step['step']:>5} | {info_step['action']:>7} | "
              f"{info_step['reward']:>10.3f} | {info_step['distance']:>10.2f} | "
              f"{info_step['angle']:>7.2f}° | {info_step['cyl_x']:>8.2f} | {info_step['cyl_z']:>8.2f}")
    print("-"*90)
    
    print(f"[WARNING] Maximum steps reached")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Final distance: {final_distance:.2f}")
    
    return (episode_reward, episode_length, final_distance, False, 
            trajectory, target_pos, step_rewards, actions)


def plot_validation_results(metrics, output_dir="plots", file_prefix="p1_validate"):
    """
    Generate all validation visualizations
    
    Args:
        metrics: ValidationMetrics object with data
        output_dir: Directory to save plots
        file_prefix: Prefix for filenames
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Rewards per episode plot
    plt.figure(figsize=(14, 6))
    episodes = np.arange(1, len(metrics.episode_rewards) + 1)
    
    # Color mask based on success
    colors = ['green' if s else 'red' for s in metrics.episode_successes]
    
    plt.subplot(1, 2, 1)
    plt.bar(episodes, metrics.episode_rewards, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=np.mean(metrics.episode_rewards), color='blue', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(metrics.episode_rewards):.2f}')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Rewards per Validation Episode', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Rewards box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(metrics.episode_rewards, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Reward Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_prefix}_results.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{file_prefix}_results.jpg")
    
    # 2. 2D Robot trajectories
    plt.figure(figsize=(12, 12))
    
    for i, trajectory in enumerate(metrics.episode_trajectories):
        if len(trajectory) > 0:
            xs = [pos[0] for pos in trajectory]
            zs = [pos[1] for pos in trajectory]
            
            # Color based on success
            if metrics.episode_successes[i]:
                color = 'green'
                alpha = 0.8
                linewidth = 2
                label = f'Ep {i+1} (✓)'
            else:
                color = 'red'
                alpha = 0.4
                linewidth = 1
                label = f'Ep {i+1} (✗)'
            
            # Draw trajectory
            plt.plot(xs, zs, '-', color=color, alpha=alpha, linewidth=linewidth, label=label)
            
            # Mark start and end
            plt.plot(xs[0], zs[0], 'o', color='blue', markersize=8, 
                    markeredgecolor='black', markeredgewidth=1)
            plt.plot(xs[-1], zs[-1], 's', color=color, markersize=10, 
                    markeredgecolor='black', markeredgewidth=1.5, alpha=1)
            
            # Mark target position
            target = metrics.episode_target_positions[i]
            plt.plot(target['x'], target['z'], '*', color='gold', markersize=20,
                    markeredgecolor='black', markeredgewidth=2, alpha=0.9)
    
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)
    plt.xlabel('X Position', fontsize=12)
    plt.ylabel('Z Position', fontsize=12)
    plt.title('2D Robot Trajectories in Validation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=8, label='Start', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
               markersize=8, label='End (Success)', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=8, label='End (Failure)', markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
               markersize=12, label='Target', markeredgecolor='black')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_prefix}_trajectories_2d.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{file_prefix}_trajectories_2d.jpg")
    
    # 3. Comparative statistics with boxplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Rewards
    axes[0, 0].boxplot(metrics.episode_rewards, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral', alpha=0.7),
                       medianprops=dict(color='darkred', linewidth=2))
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Episode lengths
    axes[0, 1].boxplot(metrics.episode_lengths, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='darkblue', linewidth=2))
    axes[0, 1].set_ylabel('Number of Steps')
    axes[0, 1].set_title('Episode Length Distribution')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Final distances
    axes[1, 0].boxplot(metrics.episode_distances, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7),
                       medianprops=dict(color='darkgreen', linewidth=2))
    axes[1, 0].set_ylabel('Distance (units)')
    axes[1, 0].set_title('Final Distance Distribution')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Success rate
    success_count = np.sum(metrics.episode_successes)
    fail_count = len(metrics.episode_successes) - success_count
    axes[1, 1].bar(['Success', 'Failure'], [success_count, fail_count], 
                   color=['green', 'red'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Number of Episodes')
    axes[1, 1].set_title(f'Success Rate: {np.mean(metrics.episode_successes)*100:.1f}%')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add values above bars
    for i, (label, value) in enumerate(zip(['Success', 'Failure'], [success_count, fail_count])):
        axes[1, 1].text(i, value + 0.1, str(value), ha='center', va='bottom', 
                       fontsize=12, fontweight='bold')
    
    plt.suptitle('Validation Statistics', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_prefix}_boxplots.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{file_prefix}_boxplots.jpg")
    
    # 4. Action histogram
    plt.figure(figsize=(10, 6))
    all_actions = []
    for episode_actions in metrics.episode_actions:
        all_actions.extend(episode_actions)
    
    action_labels = {0: 'Forward ↑', 1: 'Left ←', 2: 'Right →'}
    action_counts = [all_actions.count(i) for i in range(3)]
    
    bars = plt.bar(range(3), action_counts, color=['blue', 'orange', 'green'], 
                   alpha=0.7, edgecolor='black', linewidth=2)
    plt.xticks(range(3), [action_labels[i] for i in range(3)], fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Action Distribution During Validation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add percentages above bars
    total_actions = sum(action_counts)
    for i, (bar, count) in enumerate(zip(bars, action_counts)):
        percentage = (count / total_actions) * 100
        plt.text(bar.get_x() + bar.get_width()/2, count + max(action_counts)*0.01, 
                f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_prefix}_actions.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/{file_prefix}_actions.jpg")


def save_statistics_to_file(stats, filepath):
    """Save statistics to text file"""
    with open(filepath, 'w+', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("VALIDATION STATISTICS\n")
        f.write("="*70 + "\n")
        f.write(f"Number of episodes:           {stats['num_episodes']}\n")
        f.write(f"Successful episodes:          {stats['num_successes']} ({stats['success_rate']:.1f}%)\n")
        f.write(f"\nRewards:\n")
        f.write(f"  Mean:                       {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}\n")
        f.write(f"  Min:                        {stats['min_reward']:.2f}\n")
        f.write(f"  Max:                        {stats['max_reward']:.2f}\n")
        f.write(f"  Median:                     {stats['median_reward']:.2f}\n")
        f.write(f"\nEpisode lengths:\n")
        f.write(f"  Mean:                       {stats['mean_length']:.1f} ± {stats['std_length']:.1f} steps\n")
        f.write(f"\nFinal distance to target:\n")
        f.write(f"  Mean:                       {stats['mean_final_distance']:.2f} ± {stats['std_final_distance']:.2f}\n")
        f.write("="*70 + "\n")
    print(f"\nStatistics saved to: {filepath}")


def main():
    """Main validation function"""
    # Get current script name
    SCRIPT_NAME = Path(__file__).stem
    OUTPUT_DIR = "plots"
    
    print("\n" + "="*70)
    print("STARTING MODEL VALIDATION")
    print("="*70)
    
    # Configuration
    MODEL_PATH = "checkpoints/checkpoint.zip"
    NUM_EPISODES = 10  # Number of validation episodes
    MAX_STEPS = 200     # Maximum steps per episode
    DETERMINISTIC = True  # Use deterministic policy (no exploration)
    
    # Register environment
    gym.register(
        id="RoboboEnv-v0",
        entry_point=RoboboEnv,
    )
    
    # Create environment
    print(f"\nCreating validation environment...")
    env = gym.make("RoboboEnv-v0")
    
    # Load trained model
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = PPO.load(MODEL_PATH, env=env)
        print("Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Create metrics object
    metrics = ValidationMetrics()
    
    # Run validation episodes
    print(f"\nRunning {NUM_EPISODES} validation episodes...")
    print(f"Deterministic policy: {DETERMINISTIC}")
    print(f"Maximum steps per episode: {MAX_STEPS}")
    
    start_time = time.time()
    
    for episode in range(1, NUM_EPISODES + 1):
        # Run episode
        (reward, length, distance, success, trajectory, 
         target_pos, step_rewards, actions) = run_validation_episode(
            model, env, episode, deterministic=DETERMINISTIC, max_steps=MAX_STEPS
        )
        
        # Store metrics
        metrics.add_episode(reward, length, distance, success, trajectory, 
                          target_pos, step_rewards, actions)
    
    elapsed_time = time.time() - start_time
    
    # Close environment
    env.close()
    
    print(f"\n{'='*80}")
    print(f"VALIDATION COMPLETED IN {elapsed_time:.2f} SECONDS")
    print(f"Average time per episode: {elapsed_time/NUM_EPISODES:.2f} seconds")
    print(f"{'='*80}")
    
    # Display summary table of all episodes
    print(f"\n{'='*90}")
    print("SUMMARY OF ALL EPISODES")
    print(f"{'='*90}")
    print(f"{'Episode':>8} | {'Reward':>11} | {'Steps':>6} | {'Final Dist':>11} | {'Success?':>8}")
    print("-"*90)
    for i in range(len(metrics.episode_rewards)):
        result = "Success" if metrics.episode_successes[i] else "Failure"
        print(f"{i+1:>8} | {metrics.episode_rewards[i]:>11.2f} | {metrics.episode_lengths[i]:>6} | "
              f"{metrics.episode_distances[i]:>11.2f} | {result:>8}")
    print(f"{'='*90}")
    
    # Global statistics
    stats = metrics.get_statistics()
    
    print(f"\n{'='*80}")
    print("GLOBAL STATISTICS")
    print(f"{'='*80}")
    print(f"Completed episodes:           {stats['num_episodes']}")
    print(f"Success rate:                 {stats['success_rate']:.1f}% ({stats['num_successes']}/{stats['num_episodes']})")
    print(f"\nRewards:")
    print(f"  Mean:                       {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"  Range:                      [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
    print(f"\nEpisode lengths:")
    print(f"  Mean:                       {stats['mean_length']:.1f} steps")
    print(f"\nFinal distance to target:")
    print(f"  Mean:                       {stats['mean_final_distance']:.2f} units")
    print(f"{'='*80}")
    
    # Save statistics to text file
    stats_file = f"{OUTPUT_DIR}/{SCRIPT_NAME}_statistics.txt"
    save_statistics_to_file(stats, stats_file)
    
    # Save metrics to file
    print(f"\nSaving validation data...")
    data_file = f"{OUTPUT_DIR}/{SCRIPT_NAME}_data.npz"
    metrics.save_to_file(data_file)
    print(f"Data saved to: {data_file}")
    
    # Generate visualizations
    print(f"\nGenerating plots...")
    plot_validation_results(metrics, output_dir=OUTPUT_DIR, file_prefix=SCRIPT_NAME)
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETED")
    print("="*70)
    print(f"\nResults saved in directory: {OUTPUT_DIR}/")
    print(f"  - {SCRIPT_NAME}_results.jpg          (Rewards and distribution)")
    print(f"  - {SCRIPT_NAME}_trajectories_2d.jpg  (2D trajectories)")
    print(f"  - {SCRIPT_NAME}_boxplots.jpg         (Comparative statistics)")
    print(f"  - {SCRIPT_NAME}_actions.jpg          (Action distribution)")
    print(f"  - {SCRIPT_NAME}_data.npz             (Complete data)")
    print(f"  - {SCRIPT_NAME}_statistics.txt       (Text statistics)")
    print()


if __name__ == "__main__":
    main()
