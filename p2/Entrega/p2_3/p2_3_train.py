import time
import pickle
import gymnasium as gym
import numpy as np
import neat
from robobo_utils import RoboboEnv
import robobo_utils.neat_visualize as visualize

CONFIG_FILE = "checkpoints/2_3/config"

# ------------------------------------------------------------------------------

def eval_genomes(genomes, config):
    """
    Evaluate all genomes in the population for scenario 2.3.
    """
    id = "RoboboEnv"

    env = gym.make(
        id,
        verbose=False,
        target_name="CYLINDERBALL",
        alpha=0.35,
        penalty_strength=500,
        target_pos={'x': 1102.0, 'z': 0.0}
    )

    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        num_episodes = 2
        fitness_values = list()

        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            done = False
            steps = 0
            max_steps = 50

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
            fitness_values.append(episode_reward)
        genome.fitness = np.mean(fitness_values)
        print(f"Genome {genome_id} fitness: {genome.fitness:.2f} ({np.round(fitness_values, 2)})")
    env.close()


def main(config_file):
    """
    Main training function for NEAT algorithm (2.3).
    """
    id = "RoboboEnv"
    gym.register(
        id=id,
        entry_point=RoboboEnv,
    )
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='checkpoints/2_3/checkpoint-'))
    print("\n*** Starting NEAT evolution (2.3) ***")
    start_time = time.time()
    winner = p.run(eval_genomes, 100)
    training_time = time.time() - start_time
    print(f"\n*** Training completed in {training_time:.2f} seconds ***")
    print('\n*** Best genome ***')
    print(f'Key: {winner.key}')
    print(f'Fitness: {winner.fitness}')
    print(f'Nodes: {winner.nodes}')
    print(f'Connections: {[conn for conn in winner.connections.values()]}')
    winner_route = "checkpoints/2_3/winner_genome.pkl"
    with open(winner_route, 'wb') as f:
        pickle.dump(winner, f)
    print(f'\nWinner genome saved to {winner_route}')
    plot_route = "plots/2_3_winner_network.gv"
    visualize.draw_net(config, winner, view=False, 
                      filename=plot_route)
    print(f'Winner network visualized in {plot_route}')
    visualize.plot_stats(stats, view=False, 
                        filename="plots/2_3_fitness_stats.svg")
    visualize.plot_species(stats, view=False, 
                          filename="plots/2_3_neat_species.svg")
    print('Statistics plotted in plots/')
    print('\n*** Training complete! Run p2_3_validate.py to evaluate the winner. ***')


if __name__ == "__main__":
    main(config_file=CONFIG_FILE)
