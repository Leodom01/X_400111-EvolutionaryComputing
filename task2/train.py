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

def evaluate(phenone):

  def run_single(enemy):
    f, p, e, _ = env.run_single(pcont=phenone, enemyn=enemy, econt=None)
    return f, p, e

  fitnesses = []
  kills = 0
  deaths = 0
  out_of_time = 0
  for (f, p, e) in map(run_single, ENEMIES):
    fitnesses.append(f)
    if e == 0: kills += 1
    if p == 0: deaths += 1
    if p != 0 and e != 0: out_of_time += 1

  classic_fitness = np.average(fitnesses) - np.std(fitnesses)
  f = classic_fitness + 50 * kills - 30 * out_of_time - 30 * deaths
  return -f

def main():
  # init = [0] * neuron_number

  for run_number in range(RUN_NUMBER):

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

    print("Running on ", cpus, " processors!")

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
        print("New agent saved in tmp-agent.txt")
        best_agent = engine.result[0]

      i += 1

    np.savetxt(f"agent--fit-{engine.result[1]}.txt", engine.result[0])
    pool.close()
    # np.savetxt(f"agent-custom-fitness-fit-{engine.result[1]}.txt", engine.result[0])
    # np.savetxt(f"agent-custom-enemies-fit-{engine.result[1]}.txt", engine.result[0])
    print("Completed run number: ", run_number)

  print("Agent saved in leo-best.txt")
  np.savetxt(f"leo-best.txt", best_agent)


if __name__ == "__main__":
  multiprocessing.freeze_support()
  main()
