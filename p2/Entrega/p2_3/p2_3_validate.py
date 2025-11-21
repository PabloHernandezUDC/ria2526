import time
import pickle
import gymnasium as gym
import numpy as np
import neat
from robobo_utils import RoboboEnv

CONFIG_FILE = "checkpoints/2_3/config"
WINNER_FILE = "checkpoints/2_3/winner_genome.pkl"

def run_best_genome(genome, config, num_episodes=5, verbose=True):
    """
    Run the best genome to visualize performance (2.3).
    """
    print("\n*** Running best genome (2.3) ***")
    id = "RoboboEnv"
    env = gym.make(id, verbose=verbose, target_name="CYLINDERBALL", alpha=0.35, penalty_strength=1000.0, target_pos={'x': 1102.0, 'z': 0.0})
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    episode_rewards = []
    episode_steps = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        # No cambio de posición inicial, se usa la predeterminada
        episode_reward = 0.0
        done = False
        steps = 0
        max_steps = 200
        print(f"\n--- Episode {episode + 1} ---")
        while not done and steps < max_steps:
            sector = obs["sector"][0]
            ir_front_c = obs["ir_front_c"][0]
            ir_right = obs["ir_right"][0]
            ir_left = obs["ir_left"][0]
            ir_back_r = obs["ir_back_r"][0]
            ir_back_l = obs["ir_back_l"][0]
            
            # 6 for sector + 4 for ir_front_c + 4 other IR sensors (binary)
            nn_input = [0.0] * 14
            
            # Sector encoding (one-hot)
            if sector < 6:
                nn_input[sector] = 1.0
            
            # Front-center IR (one-hot encoding for 4 states)
            if ir_front_c < 4:
                nn_input[6 + ir_front_c] = 1.0
            
            # Other IR sensors encoding (binary)
            nn_input[10] = float(ir_right)
            nn_input[11] = float(ir_left)
            nn_input[12] = float(ir_back_r)
            nn_input[13] = float(ir_back_l)
            
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
    print("\n" + "="*50)
    print("*** Evaluation Summary (2.3) ***")
    print(f"Total episodes: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average steps: {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")
    print(f"Best reward: {np.max(episode_rewards):.2f}")
    print(f"Worst reward: {np.min(episode_rewards):.2f}")
    print("="*50)

def main(config_file=CONFIG_FILE, winner_file=WINNER_FILE, num_episodes=5, verbose=True):
    """
    Main validation function (2.3).
    """
    id = "RoboboEnv"
    gym.register(
        id=id,
        entry_point=RoboboEnv,
    )
    print(f"Loading config from: {config_file}")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )
    print(f"Loading winner genome from: {winner_file}")
    with open(winner_file, 'rb') as f:
        winner = pickle.load(f)
    print(f"\n*** Winner Genome Info (2.3) ***")
    print(f"Key: {winner.key}")
    print(f"Fitness: {winner.fitness}")
    print(f"Number of nodes: {len(winner.nodes)}")
    print(f"Number of connections: {len(winner.connections)}")
    run_best_genome(winner, config, num_episodes=num_episodes, verbose=verbose)

if __name__ == "__main__":
    main()
