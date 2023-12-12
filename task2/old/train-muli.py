import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import operator
import numpy as np

import deap.cma as cma
from deap import base, creator, tools

from environment import training_environment

# BEGIN META PARAMETERS
# ENEMIES = [1, 3, 4, 6, 7]
ENEMIES = range(1, 9)
NGEN = 100
BOUNDS = 1
# END META PARAMETERS

# BEGIN HYPER PARAMETERS
INITIAL_SIGMA = 0.2
NPOP = 30
# END HYPER PARAMETERS

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

neuron_number, env = training_environment(ENEMIES)
def evaluate(phenone):
  def run_single(enemy):
    f, _, e, _ = env.run_single(pcont=phenone, enemyn=enemy, econt=None)
    return f, e
  
  fitnesses = []
  killed = 0
  for (f, e) in map(run_single, ENEMIES):
    fitnesses.append(f)
    if e == 0: killed += 1

  return tuple(fitnesses), killed


def same(individual_1, individual_2):
  return operator.eq(individual_1, individual_2).all()

try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 1

print(f"[DEBUG] Using {cpus} cpus")


creator.create(
  "FitnessMulti", 
  base.Fitness, 
  weights=(1.0,) * len(ENEMIES)
)
creator.create(
  "Individual", 
  np.ndarray, 
  fitness=creator.FitnessMulti # type: ignore
)

toolbox = base.Toolbox()
toolbox.register("evaluate", evaluate)

halloffame = tools.ParetoFront(same)

stats = tools.Statistics(lambda ind: ind.kills)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()
logbook.header = ["gen", "nevals"] + stats.fields # type: ignore

best = 0
agent = []

# with ProcessPoolExecutor(cpus) as pool:
with multiprocessing.Pool(cpus) as pool:
  
  toolbox.register("map", pool.map)
  # toolbox.register("map", map)
  def eval_population(population):
    fitnesses = toolbox.map(toolbox.evaluate, population) # type: ignore 
    for ind, (fit, kills) in zip(population, fitnesses):
      ind.fitness.values = fit
      ind.kills = kills

    #return np.max(np.average(fitnesses))

  population = [
    creator.Individual(x) # type: ignore
    for x in np.random.uniform(-BOUNDS, BOUNDS, (NPOP, neuron_number))
  ]
  max_fitness = eval_population(population)

  strat = cma.StrategyMultiObjective(
    population,
    mu=NPOP,
    lambda_=130,
    sigma=0.02,
  )

  toolbox.register("generate", strat.generate, creator.Individual) # type: ignore
  toolbox.register("update", strat.update)

  for gen in range(NGEN):

    population = toolbox.generate() # type: ignore
    eval_population(population)

    best_killer = max(population, key=lambda x: x.kills) # type: ignore
    if best_killer.kills > best: # type: ignore
      print(f"New best killer {best_killer.kills}")
      agent = best_killer
      np.save("best-killer-multi.txt", agent)
      # print(agent)

    
    halloffame.update(population)
    record = stats.compile(population)
    logbook.record(gen=gen, nevals=len(population), **record)

    print(logbook.stream)

    toolbox.update(population) # type: ignore

print(f"Hall of fame size {len(halloffame)}")
for idx, best in enumerate(halloffame):
  np.savetxt(f"best_multi/best-multi-best{idx}.txt", best)

# np.savetxt(f"agent-multi-fit-{engine.result[1]}.txt", engine.result[0])
# np.savetxt(f"agent-custom-fitness-fit-{engine.result[1]}.txt", engine.result[0])
# np.savetxt(f"agent-custom-enemies-fit-{engine.result[1]}.txt", engine.result[0])
