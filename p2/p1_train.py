"""
Training script for Practice 1 - Robobo Navigation with PPO

Trains a PPO agent to navigate a Robobo robot towards a red cylinder
using only visual sector information.
"""

import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from robobo_utils import RoboboEnv, CustomCallback, plot_evaluation_results

# Seed for reproducibility
seed = 42


def main():
    """Main training function"""
    # Register custom environment
    id = "RoboboEnv"
    gym.register(
        id=id,
        entry_point=RoboboEnv,
    )

    # Create training environment
    train_env = Monitor(gym.make(id))
    
    # Create PPO model
    model = PPO(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        seed=seed,
        n_steps=512,
    )

    # Configure callbacks for evaluation and monitoring
    eval_env = Monitor(gym.make(id))
    eval_callback = EvalCallback(
        eval_env,
        log_path="eval_results/",
        eval_freq=512,
        n_eval_episodes=5
    )

    custom_callback = CustomCallback(verbose=1)
    callback_list = CallbackList([eval_callback, custom_callback])

    # Train the model
    start = time.time()
    model.learn(
        total_timesteps=8192,
        callback=callback_list,
        progress_bar=True)
    learning_time = time.time() - start

    # Generate evaluation plots
    plot_evaluation_results()

    print(f"Training took {(learning_time):.2f} seconds.")

    # Save trained model
    model.save("checkpoints/checkpoint.zip")


if __name__ == "__main__":
    main()
