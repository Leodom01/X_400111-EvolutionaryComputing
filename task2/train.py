# type: ingore
import os
import cma
import multiprocessing
import numpy as np
from functools import partial
from environment import training_environment
from scipy.spatial.distance import pdist
import pandas as pd
import pathlib


# BEGIN HYPER PARAMETERS
INITIAL_SIGMA = 0.2
NPOP = 100
NGEN = 500
# END HYPER PARAMETERS

FITNESS_FUNCTION = os.environ["FITNESS_FUNCTION"]
N_RUN = os.environ["N_RUN"]
ENEMIESNAME = os.environ["ENEMIES"]
ENEMIES = ENEMIESNAME.split("-")

if FITNESS_FUNCTION not in ("custom", "classic"):
  print(f"Invalid fitness function: {FITNESS_FUNCTION}")
  exit(-1)

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

neuron_number, env = training_environment(ENEMIES)

def save_run_data(data, best_agent, gain):
  savedir = f"./data/{ENEMIESNAME}/{FITNESS_FUNCTION}"
  pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

  header = [
    "step", 
    "bsf", 
    "mean_fitness_classic", 
    "min_fitness_classic", 
    "max_fitness_classic", 
    "mean_fitness_custom", 
    "min_fitness_custom", 
    "max_fitness_custom", 
    "sq_distance_diversity"
  ]

  df = pd.DataFrame(data, columns=header)
  df.to_csv(os.path.join(savedir, f"{N_RUN}.csv"), index=False)
  np.savetxt(os.path.join(savedir, f"{N_RUN}.txt"), best_agent)
  np.savetxt(os.path.join(savedir, f"{N_RUN}.gain"), gain)

def run_single(phenome, enemy):
  f, p, e, _ = env.run_single(pcont=phenome, enemyn=enemy, econt=None)
  return f, p, e

def run(phenome):
  return list(map(partial(run_single, phenome), ENEMIES))

def compute_weights(run):
  tot_enemy_gain = [0] * len(ENEMIES)
  for enemy_number, (_, p_energy, e_energy) in enumerate(run):
    tot_enemy_gain[enemy_number] += e_energy - p_energy + 100

  tot_gain = sum(tot_enemy_gain)
  if tot_gain == 0:
    return [1 / len(ENEMIES)] * len(ENEMIES)

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
   
    w_avg = np.average(fitnesses, weights=weights) \
          - np.sqrt(np.cov(fitnesses, aweights=weights))
    # blend = generation_number / NGEN
    base_fit = w_avg #std_avg * (1 - blend)  + w_avg * blend * 2
    base_fit += 50 * n_kills - 30*n_timeouts - 30*n_deaths
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

def extract_gain(xx):
  a = []
  for en in xx:
    _, p, e = en
    a.append(p - e)

  return a 

def main():
  init = [0] * neuron_number

  data = np.empty((0,9))

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

  weights = [1 / len(ENEMIES)] * len(ENEMIES)
  pool = multiprocessing.Pool(processes=cpus)

  for i in range(NGEN):
    print(f"Gen {i}")
    solutions = engine.ask()
    
    results = pool.map(run, solutions)
   
    stats = compute_stats(results)
    custom_fitness = compute_fitness(results, weights, i)
    best_idx = np.argmax(custom_fitness)

    classical_fitness, kills = map(list, zip(*stats))

    weights = compute_weights(results[best_idx])
    
    if FITNESS_FUNCTION == "custom":
      engine.tell(solutions, custom_fitness)
    else:
      engine.tell(solutions, classical_fitness)

    
    # header = [
    #   "step", 
    #   "bsf", 
    #   "mean_fitness_classic", 
    #   "min_fitness_classic", 
    #   "max_fitness_classic", 
    #   "mean_fitness_custom", 
    #   "min_fitness_custom", 
    #   "max_fitness_custom", 
    #   "sq_distance_dsumsumiversity"
    # ]
    res = [[
      i,
      engine.result[1],
      # CLASSIC
      - np.mean(classical_fitness),
      - np.max(classical_fitness),
      - np.min(classical_fitness),
      # CSUTOM
      - np.mean(custom_fitness),
      - np.max(custom_fitness),
      - np.min(custom_fitness),
      compute_diversity(solutions)
    ]]

    data = np.append(
      data,
      res,
      axis=0
    )

  xx = run(engine.result[0])
  gain = extract_gain(xx)

  save_run_data(data, engine.result[0], gain)
  


if __name__ == "__main__":
  multiprocessing.freeze_support()
  main()
