# type: ingore
import os
import cma
import multiprocessing
import numpy as np
from functools import partial
from environment import training_environment
from scipy.spatial.distance import pdist
import pandas as pd

# BEGIN META PARAMETERS

# ENEMIES = [1, 3, 4, 6, 7]
ENEMIES = [2, 5, 8]
# ENEMIES = range(1, 9)

NGEN = 10

FITNESS_FUNCTION = "custom"
# FITNESS_FUNCTION = "classic"

# END META PARAMETERS

# BEGIN HYPER PARAMETERS
INITIAL_SIGMA = 0.2
NPOP = 100
# END HYPER PARAMETERS

if FITNESS_FUNCTION not in ("custom", "classic"):
  print(f"Invalid fitness function: {FITNESS_FUNCTION}")
  exit(-1)

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

neuron_number, env = training_environment(ENEMIES)

def create_data_folder_if_not_existant(name):
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists("data/"+name):
        os.makedirs("data/"+name)

def dump_it(run_numb,matrix_of_run,name):
       
    header = ["step", "bsf", "mean_fitness", "std_fitness", "min_fitness", "max_fitness", "sq_distance_diversity"]
    
    df = pd.DataFrame(matrix_of_run,columns=header)
    df.to_csv(os.path.join("data/"+name, "run-"+str(run_numb)+".csv"), index=False)

def run_single(phenome, enemy):
  f, p, e, _ = env.run_single(pcont=phenome, enemyn=enemy, econt=None)
  return f, p, e

def run(phenome):
  return list(map(partial(run_single, phenome), ENEMIES))

def compute_weights(run):
  tot_enemy_gain = [0] * len(ENEMIES)
  for enemy_number, (_, p_energy, e_energy) in enumerate(run):
    tot_enemy_gain[enemy_number] += e_energy

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
    
    w_avg = np.average(fitnesses, weights=weights) \
          - np.sqrt(np.cov(fitnesses, aweights=weights))
    # blend = generation_number / NGEN
    base_fit = w_avg #std_avg * (1 - blend)  + w_avg * blend * 2 
    # base_fit += 100 * n_kills
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
  init = [0] * neuron_number

  temp_matrix_for_dumping = np.empty((0,7))

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
  with multiprocessing.Pool(processes=cpus) as pool:

    for i in range(NGEN):
      solutions = engine.ask()
      
      results = pool.map(run, solutions)
     
      stats = compute_stats(results)
      fitness = compute_fitness(results, weights, i)
      best_idx = np.argmax(fitness)

      classical_fitness, kills = map(list, zip(*stats))

      weights = compute_weights(results[best_idx])
      
      if FITNESS_FUNCTION == "custom":
        engine.tell(solutions, fitness)
      else:
        engine.tell(solutions, classical_fitness)

      max_kills = np.max(kills)
      current_best_fitness = - np.min(fitness)
      current_average_fitness = - np.average(fitness)
      current_best_classical_fitness = - np.min(classical_fitness)
      current_average_classical_fitness = - np.average(classical_fitness)
      diversity = compute_diversity(solutions)
      current_best_gain = compute_gain(results[best_idx])

      global_best = - engine.result[1]

      print(
        f"Generation {i}\t"
        f"max custom fitness: {current_best_fitness} "
        f"(classic: {current_best_classical_fitness}, gain:  {current_best_gain})\t"
        f"average custom fitness: {current_average_fitness} "
        f"(classic: {current_average_classical_fitness})\t"
        f"kills: {max_kills}\t"
        f"diversity: {diversity}\t"
        f"best of all time: {global_best}"
      )

      temp_matrix_for_dumping = np.append(temp_matrix_for_dumping, [[i,np.max(fitness),np.mean(fitness),np.std(fitness),np.min(fitness),np.max(fitness),diversity]], axis=0) 

      create_data_folder_if_not_existant("test")
      #np.savetxt(f"data/tmp-agent.txt", engine.result[0])
      np.savetxt("data/test/individual-"+str(i)+".txt", engine.result[0])

    np.savetxt(f"data/test/agent-weights-fit-{engine.result[1]}.txt", engine.result[0])
    #TODO is there a master script calling various times so I can change the first variable fixed at 1?
    dump_it(1,temp_matrix_for_dumping,"test")

if __name__ == "__main__":
  multiprocessing.freeze_support()
  main()
