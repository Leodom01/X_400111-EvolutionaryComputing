import os
import cma
import multiprocessing
import numpy as np
from functools import partial
from environment import training_environment
from scipy.spatial.distance import pdist

# BEGIN META PARAMETERS
# ENEMIES = [1, 3, 4, 6, 7]
ENEMIES = range(1, 9)
NGEN = 300
RUN_NUMBER = 5
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

def compute_weights(run):
  tot_enemy_gain = [0] * len(ENEMIES)
  for enemy_number, (_, p_energy, e_energy) in enumerate(run):
    tot_enemy_gain[enemy_number] += (e_energy - p_energy) + 100

  tot_gain = sum(tot_enemy_gain)
  return list(map(lambda gain: gain / tot_gain, tot_enemy_gain))

def compute_fitness(runs, weights, generation_number):
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
    
    # std_avg = np.average(fitnesses)
    w_avg = np.average(fitnesses, weights=weights) \
          - np.sqrt(np.cov(fitnesses, aweights=weights))
    # blend = generation_number / NGEN
    base_fit = w_avg #std_avg * (1 - blend)  + w_avg * blend * 2 
    base_fit += 100 * n_kills
    return - base_fit

  return list(map(compute, runs))

def compute_gain(run):
  tot_gain = []
  # print(run)
  for (_, p_energy, e_energy) in run:
    tot_gain.append(p_energy - e_energy)
  return np.average(tot_gain)

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


def compute_diversity(solutions):
  pop = np.array(solutions)
  distances = pdist(pop, "sqeuclidean")
  return np.sum(distances)



def main():
  # init = [0] * neuron_number

  for _ in range(RUN_NUMBER):

    init = np.loadtxt("./tmp-agent.txt")

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

    pool = multiprocessing.Pool(processes=cpus)

    # for i in range(NGEN):
    i = 0
    while i < NGEN:
      solutions = engine.ask()
      fitness = list(pool.map(evaluate, solutions))
      engine.tell(solutions, fitness)

      current_best_fitness = - np.min(fitness)
      average_fitness = - np.average(fitness)
      global_best = - engine.result[1]

      print(
        f"Generation {i}\t"
        f"max fitness: {current_best_fitness}\t"
        f"average fitness: {average_fitness}\t"
        f"best of all time: {global_best}"
      )

      if current_best_fitness >= global_best:
        np.savetxt(f"tmp-agent.txt", engine.result[0])

      i += 1

    np.savetxt(f"agent--fit-{engine.result[1]}.txt", engine.result[0])
    pool.close()
    # np.savetxt(f"agent-custom-fitness-fit-{engine.result[1]}.txt", engine.result[0])
    # np.savetxt(f"agent-custom-enemies-fit-{engine.result[1]}.txt", engine.result[0])

  np.savetxt(f"leo-best.txt", engine.result[0])


if __name__ == "__main__":
  multiprocessing.freeze_support()
  main()
