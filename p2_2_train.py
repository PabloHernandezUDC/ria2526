import time
import pickle
import gymnasium as gym
import numpy as np
import neat
from robobo_utils import RoboboEnv
import robobo_utils.neat_visualize as visualize

CONFIG_FILE = "checkpoints/2_2/config"

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
    env = gym.make(id, verbose=False, target_name="CYLINDERBALL", alpha=0.5)
    
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Run multiple episodes to get average fitness
        num_episodes = 1
        fitness_values = list()
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            
            env.unwrapped.sim.setRobotLocation(0, position={'x': -1000.0, 'y': 39.0, 'z': -400.0})
            time.sleep(0.1)
            
            episode_reward = 0.0
            done = False
            steps = 0
            max_steps = 50
            
            while not done and steps < max_steps:
                # Get sector and IR from observation
                sector = obs["sector"][0]
                ir_front = obs["ir_front"][0]
                
                # Convert to neural network input (one-hot encoding for both)
                nn_input = [0.0] * 10  # 6 for sector + 4 for IR
                if sector < 6:
                    nn_input[sector] = 1.0
                if ir_front < 4:
                    nn_input[6 + ir_front] = 1.0
                
                # Get action from neural network
                output = net.activate(nn_input)
                
                # Convert neural network output to action
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
        print(f"Genome {genome_id} fitness: {genome.fitness:.2f} ({np.round(fitness_values, 2)})")
    
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
    p.add_reporter(neat.Checkpointer(5, filename_prefix='checkpoints/2_2/checkpoint-'))

    # Run for up to 300 generations
    print("\n*** Starting NEAT evolution ***")
    start_time = time.time()
    
    winner = p.run(eval_genomes, 10)
    
    training_time = time.time() - start_time
    print(f"\n*** Training completed in {training_time:.2f} seconds ***")

    # Display the winning genome
    print('\n*** Best genome ***')
    print(f'Key: {winner.key}')
    print(f'Fitness: {winner.fitness}')
    print(f'Nodes: {winner.nodes}')
    print(f'Connections: {[conn for conn in winner.connections.values()]}')

    # Save the winner
    winner_route = "checkpoints/2_2/winner_genome.pkl"
    with open(winner_route, 'wb') as f:
        pickle.dump(winner, f)
    print(f'\nWinner genome saved to {winner_route}')

    # Visualize the winner network
    plot_route = "plots/2_2_winner_network.gv"
    visualize.draw_net(config, winner, view=False, 
                      filename=plot_route)
    print(f'Winner network visualized in {plot_route}')

    # Plot statistics
    visualize.plot_stats(stats, view=False, 
                        filename="plots/2_2_fitness_stats.svg")
    visualize.plot_species(stats, view=False, 
                          filename="plots/2_2_neat_species.svg")
    print('Statistics plotted in plots/')

    print('\n*** Training complete! Run p2_2_validate.py to evaluate the winner. ***')


if __name__ == "__main__":
    main(config_file=CONFIG_FILE)
