from leap_ec.algorithm import generational_ea
from leap_ec.individual import IdentityDecoder
from leap_ec.representation import Representation
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.problem import FunctionProblem
from leap_ec import ops
from leap_ec.probe import FitnessStatsCSVProbe

import numpy as np


from matplotlib import pyplot as plt

from environment import training_environment

# PARAMETERS

MAX_GENERATIONS = 30
POPULATION_SIZE = 100
ELITISM = 5

# END PARAMETERS

# To run pygame headless
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

neurons_number, env = training_environment([1])

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


fitness_probe = FitnessStatsCSVProbe()

out = generational_ea(
    max_generations=MAX_GENERATIONS,
    pop_size=POPULATION_SIZE,
    problem=problem,
    representation=representation,
    k_elites=ELITISM,
    # Evolution Pipeline
    pipeline=[
        fitness_probe,

	#ops.tournament_selection(k=2),
        ops.proportional_selection(offset='pop-min'),

        ops.clone,

	ops.uniform_crossover,
        mutate_gaussian(std=0.5, expected_num_mutations=1),

        ops.evaluate,
        ops.pool(size=POPULATION_SIZE),
    ]
)

for i, best in out:
    np.savetxt("train.txt", best.decode())

