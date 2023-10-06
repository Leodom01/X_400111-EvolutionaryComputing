import os
import multiprocessing
import numpy as np

import deap.cma as cma
impoe
from deap import base, creator

from environment import training_environment

# BEGIN META PARAMETERS
# ENEMIES = [1, 3, 4, 6, 7]
ENEMIES = range(1, 9)
NGEN = 500
# END META PARAMETERS

# BEGIN HYPER PARAMETERS
INITIAL_SIGMA = 0.2
NPOP = 100
# END HYPER PARAMETERS

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

neuron_number, env = training_environment(ENEMIES)

try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 1
pool = multiprocessing.Pool(processes=cpus)

def evaluate(phenone):
    def run_single(enemy):
	f, _, _, _ = env.run_single(pcont=phenone, enemyn=enemy, econt=None)
	return f

    return tuple(pool.map(run_single, ENEMIES))

# Fitness function
creator.create(
    "FitnessMulti", 
    base.Fitness, 
    weights=(1.0,) * len(ENEMIES)
)

# Individuals
creator.create(
    "Individual", 
    np.ndarray, 
    fitness=creator.FitnessMulti, # type: ignore
    player_life=100, 
    enemy_life=100
)

strategy = cma.StrategyMultiObjective(
    [
	creator.Individual(x) # type: ignore
	for x in (np.random.uniform(-1000, 1000, (NPOP, neuron_number)))
    ],
    sigma = 5
)

toolbox = base.Toolbox()
toolbox.register("generate", strategy.generate, creator.Individual) # type: ignore
toolbox.register("update", strategy.update)
toolbox.register("evaluate", evaluate)
toolbox.register("map", pool.map)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)

# for i in range(NGEN):
i = 0
while i < NGEN:

    i += 1

np.savetxt(f"agent-multi-fit-{engine.result[1]}.txt", engine.result[0])
# np.savetxt(f"agent-custom-fitness-fit-{engine.result[1]}.txt", engine.result[0])
# np.savetxt(f"agent-custom-enemies-fit-{engine.result[1]}.txt", engine.result[0])
