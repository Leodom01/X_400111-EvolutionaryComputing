import math
import neat
import numpy as np
import time
from NEAT_controller import player_controller
from NEAT_population import EvomanPopulation
from custom_crossover import whole_arithmetic_recombination
from environment import training_environment, Environment
from leap_ec import ops
from leap_ec.algorithm import generational_ea
from leap_ec.individual import IdentityDecoder
from leap_ec.probe import FitnessStatsCSVProbe, pairwise_squared_distance_metric
from leap_ec.problem import FunctionProblem
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.representation import Representation
from matplotlib import shutil

# HYPER PARAMETERS
MAX_GENERATIONS = 30
POPULATION_SIZE = 100
ELITISM = 0
NUM_RUNS = 10
# END HYPER PARAMETERS

# META PARAMETERS
TRIES = 10
ENEMY = 3
# END META PARAMETERS

def nn_out_to_controls(nn_outputs):
    if nn_outputs[0] > 0.5:
        left = 1
    else:
        left = 0

    if nn_outputs[1] > 0.5:
        right = 1
    else:
        right = 0

    if nn_outputs[2] > 0.5:
        jump = 1
    else:
        jump = 0

    if nn_outputs[3] > 0.5:
        shoot = 1
    else:
        shoot = 0

    if nn_outputs[4] > 0.5:
        release = 1
    else:
        release = 0

    return [left, right, jump, shoot, release]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness, genome.player_energy, genome.enemy_energy, genome.individual_gain = run(env, net)

    return genome.fitness, genome.player_energy, genome.enemy_energy, genome.individual_gain


def run(env,net):
    fitness, player_energy, enemy_energy, t = env.play(pcont=net)
    gain = player_energy-enemy_energy

    return fitness, player_energy, enemy_energy, gain


# To run pygame headless
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

root_dir = f"./data.enemy{ENEMY}"


if __name__ == "__main__":

    experiment_name = 'neat'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    for run_number in range(1, NUM_RUNS + 1):
        env = Environment(experiment_name=experiment_name,
                          enemies=[ENEMY],
                          playermode="ai",
                          player_controller=player_controller(),
                          enemymode="static",
                          randomini='yes',
                          level=2,
                          speed="fastest",
                          savelogs="no",
                          logs="off"
                          )

        # Load configuration.
        config = neat.Config(neat.DefaultGenome,
                             neat.DefaultReproduction,
                             neat.DefaultSpeciesSet,
                             neat.DefaultStagnation,
                             'neat_config')

        # Create the population, which is the top-level object for a NEAT run.
        p = EvomanPopulation(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))

        # Run until a solution is found or max generation reached
        best_ever, best_last_gen = p.run(eval_genomes, MAX_GENERATIONS)
        evoman_reporter.plot_report()

        # Display the winning genome.
        print("\nbest_ever: {!s}".format(best_ever.fitness))

        base_path = os.path.join(RUNS_DIR, "enemy_" + str(ENEMY))
        os.makedirs(
            base_path,
            exist_ok=True,
        )

        # Dumping the updated winners list to the file
        best_file_name = 'best_individual_run_%d' % (run_number)

        with open(os.path.join(base_path, best_file_name), 'wb') as file_out:
            pickle.dump(best_ever, file_out)
