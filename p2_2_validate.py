import time
import pickle
import gymnasium as gym
import numpy as np
import neat
from robobo_utils import RoboboEnv

CONFIG_FILE = "checkpoints/2_2/config"
WINNER_FILE = "checkpoints/2_2/winner_genome.pkl"


def run_best_genome(genome, config, num_episodes=5, verbose=True):
    """
    Run the best genome to visualize performance.
    
    Args:
        genome: Best genome from evolution
        config: NEAT configuration object
        num_episodes: Number of episodes to run (default: 5)
        verbose: Whether to print detailed output (default: True)
    """
    print("\n*** Running best genome ***")
    
    id = "RoboboEnv"
    env = gym.make(id, verbose=verbose, target_name="CYLINDERBALL", alpha=0.5)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    episode_rewards = []
    episode_steps = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        
        # Set robot to custom starting position
        env.unwrapped.sim.setRobotLocation(0, position={'x': -1000.0, 'y': 39.0, 'z': -400.0})
        time.sleep(0.1)
        
        episode_reward = 0.0
        done = False
        steps = 0
        max_steps = 200
        
        print(f"\n--- Episode {episode + 1} ---")
        while not done and steps < max_steps:
            sector = obs["sector"][0]
            ir_front = obs["ir_front"][0]
            
            nn_input = [0.0] * 10  # 6 for sector + 4 for IR
            if sector < 6:
                nn_input[sector] = 1.0
            if ir_front < 4:
                nn_input[6 + ir_front] = 1.0
            
            output = net.activate(nn_input)
            action = np.argmax([output[0] < 0.33, 
                               0.33 <= output[0] < 0.67, 
                               output[0] >= 0.67])
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        
        status = "SUCCESS" if terminated else "TRUNCATED"
        print(f"Episode {episode + 1} - {status}: reward={episode_reward:.2f}, steps={steps}")
    
    env.close()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("*** Evaluation Summary ***")
    print(f"Total episodes: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average steps: {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")
    print(f"Best reward: {np.max(episode_rewards):.2f}")
    print(f"Worst reward: {np.min(episode_rewards):.2f}")
    print("="*50)


def main(config_file=CONFIG_FILE, winner_file=WINNER_FILE, num_episodes=5, verbose=True):
    """
    Main validation function.
    
    Args:
        config_file: Path to NEAT configuration file
        winner_file: Path to saved winner genome pickle file
        num_episodes: Number of episodes to evaluate (default: 5)
        verbose: Whether to print detailed output (default: True)
    """
    # Register custom environment
    id = "RoboboEnv"
    gym.register(
        id=id,
        entry_point=RoboboEnv,
    )

    # Load NEAT configuration
    print(f"Loading config from: {config_file}")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # Load the winner genome
    print(f"Loading winner genome from: {winner_file}")
    with open(winner_file, 'rb') as f:
        winner = pickle.load(f)
    
    print(f"\n*** Winner Genome Info ***")
    print(f"Key: {winner.key}")
    print(f"Fitness: {winner.fitness}")
    print(f"Number of nodes: {len(winner.nodes)}")
    print(f"Number of connections: {len(winner.connections)}")

    # Run evaluation
    run_best_genome(winner, config, num_episodes=num_episodes, verbose=verbose)


if __name__ == "__main__":
    main()
