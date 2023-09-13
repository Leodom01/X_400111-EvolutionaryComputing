from leap_ec.algorithm import generational_ea
from leap_ec.individual import IdentityDecoder
from leap_ec.representation import Representation
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.problem import FunctionProblem
from leap_ec import ops

import numpy as np

from environment import training_environment

# PARAMETERS

MAX_GENERATIONS = 300
POPULATION_SIZE = 100

# END PARAMETERS

# To run pygame headless
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

neurons_number, env = training_environment()

def evaluate(phenome):
    f, _, _, _ = env.play(phenome)
    return f

problem = FunctionProblem(
    evaluate,
    maximize=True
)

# Bound for the sampling
a = 100000000000
bounds = [(-a, a)] * neurons_number
representation = Representation(
    decoder=IdentityDecoder(),
    # We inizialize sampling from a normal distribution
    initialize=create_real_vector(bounds)
)

out = generational_ea(
    MAX_GENERATIONS,
    POPULATION_SIZE,
    problem,
    representation,
    # Evolution Pipeline
    [
	ops.tournament_selection(k=2),
        ops.clone,
	ops.uniform_crossover,
        mutate_gaussian(std=0.5, expected_num_mutations=1),
        ops.evaluate,
        ops.pool(size=POPULATION_SIZE),
    ]
)

for i, best in out:
    print(f"{i}, {best}")

input("ENTER TO SHOW THE PLAY")
np.savetxt("train.txt", best.decode())

