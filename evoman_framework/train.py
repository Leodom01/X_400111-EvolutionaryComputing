from leap_ec.algorithm import generational_ea
from leap_ec.individual import IdentityDecoder
from leap_ec.representation import Representation
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.problem import FunctionProblem
from leap_ec import ops
from leap_ec.probe import FitnessStatsCSVProbe, pairwise_squared_distance_metric

from matplotlib import shutil

from environment import training_environment

import collections
import time
import math


# PARAMETERS
MAX_GENERATIONS = 150
POPULATION_SIZE = 100
ELITISM = 1
# END PARAMETERS

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

# EXPERIMENT_CONFIG
TRIES = 5
EXPERIMENT_NAME = "2pt_crossover"
# END EXPERIMENT_CONFIG

dir = f"./data/{EXPERIMENT_NAME}"
if os.path.exists(dir):
    shutil.rmtree(dir)
    
os.makedirs(dir)

for try_number in range(TRIES):

    print(f"Starting run {try_number} for experiment {EXPERIMENT_NAME}")
    
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

            ops.proportional_selection(offset='pop-min'),

            ops.clone,

            #ops.uniform_crossover,
            ops.n_ary_crossover(num_points=2),
            mutate_gaussian(std=0.05, expected_num_mutations='isotropic'),

            ops.evaluate,
            ops.pool(size=POPULATION_SIZE),
        ]
    )
    collections.deque(out, maxlen=0)

    end_time = time.time()
    duration = end_time - start_time
    

    print(f"Finished run {try_number} for experiment {EXPERIMENT_NAME}"
          f" took {math.floor(duration)} seconds")

    stream.close()

