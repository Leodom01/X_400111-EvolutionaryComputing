from leap_ec.algorithm import generational_ea
from leap_ec.individual import IdentityDecoder
from leap_ec.representation import Representation
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.problem import FunctionProblem
from leap_ec import ops
from leap_ec.probe import FitnessStatsCSVProbe, pairwise_squared_distance_metric

from matplotlib import shutil
import numpy as np
from toolz import curry

from environment import training_environment

import collections
import time
import math

@curry
@ops.iteriter_op
def whole_arithmetic_recombination(next_individual, alpha = 0.5):
    def _whole_arithmetic_recombination(parent1, parent2, alpha):
        tmp = alpha * parent1.genome + (1 - alpha) * parent2.genome
        parent2.genome = (1 - alpha) * parent1.genome + alpha * parent2.genome
        parent1.genome = tmp

        return parent1, parent2

    while True:
        parent1 = next(next_individual)
        parent2 = next(next_individual)

        child1, child2 = _whole_arithmetic_recombination(
            parent1, 
            parent2, 
            alpha
        )

        child1.fitness = child2.fitness = None

        yield child1
        yield child2

# HYPER PARAMETERS
MAX_GENERATIONS = 50
POPULATION_SIZE = 100
ELITISM = 0
# END HYPER PARAMETERS

# META PARAMETERS
TRIES = 10
# END META PARAMETERS

# EXPERIMENTS

experiments = [
    ("2pt_crossover", ops.n_ary_crossover(num_points=2)),
    ("uniform_crossover", ops.uniform_crossover),
    ("whole_arithmetic_recombination", whole_arithmetic_recombination),
    ("whole_arithmetic_recombination_07", whole_arithmetic_recombination(
        alpha=0.7
    )), # type: ignore
]

# To run pygame headless
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

neurons_number, env = training_environment([2])

def evaluate(phenome):
    f, _, _, _ = env.play(phenome)
    return f

problem = FunctionProblem(
    evaluate,
    maximize=True
)

# Bound for the sampling
a = 10_0000
bounds = [(-a, a)] * neurons_number
representation = Representation(
    decoder=IdentityDecoder(),
    # We inizialize sampling from a normal distribution
    initialize=create_real_vector(bounds)
)

def run_experiment(experiment_name, crossover):

    dir = f"./data/{experiment_name}"
    if os.path.exists(dir):
        shutil.rmtree(dir)
        
    os.makedirs(dir)

    for try_number in range(TRIES):

        print(f"Starting run {try_number} for experiment {experiment_name}")
        
        stream = open(f"{dir}/run-{try_number}.csv", "w+")
        fitness_probe = FitnessStatsCSVProbe(
            stream=stream,
            extra_metrics={
                # Compute the genetic diversity by distance between vectors
                "sq_disance_diversity": pairwise_squared_distance_metric
            }
        )

        start_time = time.time()
        
        out = generational_ea(
            max_generations=MAX_GENERATIONS,
            pop_size=POPULATION_SIZE,
            problem=problem,
            representation=representation,
            k_elites=ELITISM,
            # Evolution Pipeline
            pipeline=[
                fitness_probe,

                ops.tournament_selection(k=10),

                ops.clone,

                crossover,
                mutate_gaussian(std=0.05, expected_num_mutations='isotropic'),

                ops.evaluate,
                ops.pool(size=POPULATION_SIZE),
            ]
        )

        best_individual = None
        best_fitness = -1
        for _, best in out:
            if best.fitness > best_fitness:
                best_individual = best.decode()
        
        if best_individual is not None:
            np.savetxt(f"{dir}/individual-{try_number}.txt", best_individual)

        end_time = time.time()
        duration = end_time - start_time
        

        print(f"Finished run {try_number} for experiment {experiment_name}"
              f" took {math.floor(duration)} seconds")

        stream.close()

for experiment_name, crossover in experiments:
    print(f"Running experiment: {experiment_name}")
    run_experiment(experiment_name, crossover)
