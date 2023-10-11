import os
import cma
import multiprocessing
import numpy as np
from functools import partial
from environment import training_environment

# BEGIN META PARAMETERS
# ENEMIES = [1, 3, 4, 6, 7]
ENEMIES = range(1, 9)
NGEN = 300
# END META PARAMETERS

# BEGIN HYPER PARAMETERS
INITIAL_SIGMA = 0.2
NPOP = 100
# END HYPER PARAMETERS

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

neuron_number, env = training_environment(ENEMIES)

def run_single(phenome, enemy):
  f, p, e, _ = env.run_single(pcont=phenome, enemyn=enemy, econt=None)
  return f, p, e

def run(phenome):
  return list(map(partial(run_single, phenome), ENEMIES))

def compute_weights(runs):
  tot_enemy_gain = [0] * len(ENEMIES)
  for run in runs:
    for enemy_number, (_, p_energy, e_energy) in enumerate(run):
      tot_enemy_gain[enemy_number] += (e_energy - p_energy)

  tot_gain = sum(tot_enemy_gain)
  return list(map(lambda gain: gain / tot_gain, tot_enemy_gain))

def compute_fitness(runs, weights, generation_number):
  weights = compute_weights(runs)

  def compute(run):
    fitnesses = []
    n_kills = 0
    n_deaths = 0
    n_timeouts = 0
    for f, p_energy, e_energy in run:
      fitnesses.append(f)
      if p_energy == 0:
        n_deaths += 1
      if e_energy == 0:
        n_kills += 1
      if p_energy != 0 and e_energy != 0:
        n_timeouts += 1
    
    base_fit = np.average(fitnesses, weights=weights)
    return - base_fit

  return list(map(compute, runs))

def compute_stats(runs):
  def compute(run):
    fitnesses = []
    n_kills = 0
    n_deaths = 0
    n_timeouts = 0
    for f, p_energy, e_energy in run:
      fitnesses.append(f)
      if p_energy == 0:
        n_deaths += 1
      if e_energy == 0:
        n_kills += 1
      if p_energy != 0 and e_energy != 0:
        n_timeouts += 1
    
    base_fit = np.average(fitnesses) - np.std(fitnesses)
    return (- base_fit, n_kills)

  return list(map(compute, runs))

def main():
  init = [0] * neuron_number

  engine = cma.CMAEvolutionStrategy(
    init,
    INITIAL_SIGMA,
    {
      "popsize": NPOP,
    }
  )

  try:
    cpus = multiprocessing.cpu_count()
  except NotImplementedError:
    cpus = 1

  with multiprocessing.Pool(processes=cpus) as pool:

    for i in range(NGEN):
      solutions = engine.ask()
      
      results = pool.map(run, solutions)
     
      stats = compute_stats(results)
      classical_fitness, kills = map(list, zip(*stats))

      weights = compute_weights(results)
      fitness = compute_fitness(results, weights, i)

      engine.tell(solutions, fitness)

      max_kills = np.max(kills)
      current_best_fitness = - np.min(fitness)
      current_average_fitness = - np.average(fitness)
      current_best_classical_fitness = - np.min(classical_fitness)
      current_average_classical_fitness = - np.average(classical_fitness)

      global_best = - engine.result[1]

      print(
        f"Generation {i}\t"
        f"max fitness: {current_best_fitness} "
        f"(classic: {current_best_classical_fitness})\t"
        f"average fitness: {current_average_fitness} "
        f"(classic: {current_average_classical_fitness})\t"
        f"kills: {max_kills}\t"
        f"best of all time: {global_best}"
      )

      np.savetxt(f"tmp-agent.txt", engine.result[0])

    np.savetxt(f"agent-weights-fit-{engine.result[1]}.txt", engine.result[0])

if __name__ == "__main__":
  multiprocessing.freeze_support()
  main()
