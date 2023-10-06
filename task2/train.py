import os
import cma
import multiprocessing
import numpy as np

from environment import training_environment

# BEGIN META PARAMETERS
# ENEMIES = [1, 3, 4, 6, 7]
ENEMIES = range(1, 9)
NGEN = 1000
# END META PARAMETERS

# BEGIN HYPER PARAMETERS
INITIAL_SIGMA = 0.2
NPOP = 100
# END HYPER PARAMETERS

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

neuron_number, env = training_environment(ENEMIES)

def fitness_signle(p, e, t):
    phi = 0.9
    alpha = 0.1
    # phi = 0.8
    # alpha = 0.2
    return phi * (100 - e) + alpha * p - np.log(t)

def evaluate(phenone):
    f, p, e, t = env.play(phenone)
    # fitness = fitness_signle(p, e, t)
    return -f

engine = cma.CMAEvolutionStrategy(
    [0] * neuron_number,
    INITIAL_SIGMA,
    {
	"popsize": NPOP,
    }
)

try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 1
pool = multiprocessing.Pool(processes=cpus)

# for i in range(NGEN):
i = 0
while i < NGEN:
    solutions = engine.ask()
    fitness = pool.map(evaluate, solutions)
    engine.tell(solutions, fitness)

    current_best_fitness = -min(fitness)
    global_best = -engine.result[1]
    print(f"Generation {i} max_fitness: {current_best_fitness}"
	  f"best of all time: {global_best}")
    np.savetxt(f"tmp-agent.txt", engine.result[0])

    i += 1

np.savetxt(f"agent--fit-{engine.result[1]}.txt", engine.result[0])
# np.savetxt(f"agent-custom-fitness-fit-{engine.result[1]}.txt", engine.result[0])
# np.savetxt(f"agent-custom-enemies-fit-{engine.result[1]}.txt", engine.result[0])
