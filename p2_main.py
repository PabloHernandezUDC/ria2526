import gymnasium as gym
from robobo_utils import RoboboEnv
import neat
import robobo_utils.neat_visualize as visualize

CONFIG_FILE = "checkpoints/neat/config"

# ------------------------------------------------------------------------------

def eval_genomes(genomes, config):
    ...


def main(config_file):
    id = "RoboboEnv"
    
    gym.register(
        id=id,
        entry_point=RoboboEnv,
    )
    
    train_env = gym.make(id)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
        )

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    # winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))



if __name__ == "__main__":
    main(config_file=CONFIG_FILE)
    
    