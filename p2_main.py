import time
import pickle
import gymnasium as gym
import numpy as np
import neat
from robobo_utils import RoboboEnv
import robobo_utils.neat_visualize as visualize

CONFIG_FILE = "checkpoints/2_1/config"

# ------------------------------------------------------------------------------

def eval_genomes(genomes, config):
    """
    Evaluate all genomes in the population.
    
    Args:
        genomes: List of (genome_id, genome) tuples
        config: NEAT configuration object
    """
    # Create environment with verbose=False to reduce output
    id = "RoboboEnv"
    env = gym.make(id, verbose=False)
    
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Run multiple episodes to get average fitness
        num_episodes = 3
        # total_fitness = 0.0
        fitness_values = list()
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            done = False
            steps = 0
            max_steps = 50
            
            while not done and steps < max_steps:
                # Get sector from observation (0-5)
                sector = obs["sector"][0]
                
                # Convert sector to neural network input (one-hot encoding)
                nn_input = [0.0] * 6
                if sector < 6:
                    nn_input[sector] = 1.0
                
                # Get action from neural network
                output = net.activate(nn_input)
                
                # Convert neural network output to action
                # Output is a single value, map to 3 discrete actions
                action = np.argmax([output[0] < 0.33, 
                                   0.33 <= output[0] < 0.67, 
                                   output[0] >= 0.67])
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                steps += 1
            
            fitness_values.append(episode_reward)
        
        # Average fitness over episodes
        genome.fitness = np.mean(fitness_values)
        print(f"Genome {genome_id} fitness: {genome.fitness:.2f} ({fitness_values})")
    
    env.close()


def run_best_genome(genome, config):
    """
    Run the best genome to visualize performance.
    
    Args:
        genome: Best genome from evolution
        config: NEAT configuration object
    """
    print("\n*** Running best genome ***")
    
    id = "RoboboEnv"
    env = gym.make(id, verbose=False)  # Verbose for final evaluation
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    num_episodes = 5
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        while not done and steps < 200:
            sector = obs["sector"][0]
            nn_input = [0.0] * 6
            if sector < 6:
                nn_input[sector] = 1.0
            
            output = net.activate(nn_input)
            action = np.argmax([output[0] < 0.33, 
                               0.33 <= output[0] < 0.67, 
                               output[0] >= 0.67])
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        
        print(f"Episode {episode + 1} reward: {episode_reward:.2f}, steps: {steps}")
    
    env.close()


def main(config_file):
    """
    Main training function for NEAT algorithm.
    
    Args:
        config_file: Path to NEAT configuration file
    """
    # Register custom environment
    id = "RoboboEnv"
    gym.register(
        id=id,
        entry_point=RoboboEnv,
    )

    # Load NEAT configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )

    # Create the population, which is the top-level object for a NEAT run
    p = neat.Population(config)

    # Add reporters to show progress in the terminal
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='checkpoints/2_1/checkpoint-'))

    # Run for up to 300 generations
    print("\n*** Starting NEAT evolution ***")
    start_time = time.time()
    
    winner = p.run(eval_genomes, 5)
    
    training_time = time.time() - start_time
    print(f"\n*** Training completed in {training_time:.2f} seconds ***")

    # Display the winning genome
    print('\n*** Best genome ***')
    print(f'Key: {winner.key}')
    print(f'Fitness: {winner.fitness}')
    print(f'Nodes: {winner.nodes}')
    print(f'Connections: {[conn for conn in winner.connections.values()]}')

    # Save the winner
    winner_route = "checkpoints/2_1/winner_genome.pkl"
    with open(winner_route, 'wb') as f:
        pickle.dump(winner, f)
    print(f'\nWinner genome saved to {winner_route}')

    # Visualize the winner network
    plot_route = "plots/2_1_winner_network.gv"
    visualize.draw_net(config, winner, view=False, 
                      filename=plot_route)
    print(f'Winner network visualized in {plot_route}')

    # Plot statistics
    visualize.plot_stats(stats, view=False, 
                        filename="plots/2_1_fitness_stats.svg")
    visualize.plot_species(stats, view=False, 
                          filename="plots/2_1_neat_species.svg")
    print('Statistics plotted in plots/')

    # Run the best genome
    run_best_genome(winner, config)


if __name__ == "__main__":
    main(config_file=CONFIG_FILE)
    
    