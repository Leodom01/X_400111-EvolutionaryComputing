import os
from environment import training_environment
from functools import partial
import numpy as np

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

ENEMIES = range(1, 9)
neuron_number, env = training_environment(ENEMIES)

def run_single(phenome, enemy):
  f, p, e, _ = env.run_single(pcont=phenome, enemyn=enemy, econt=None)
  return f, p, e

def run(phenome):
  return list(map(partial(run_single, phenome), ENEMIES))

def compute_kills(runs):
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
   
    return n_kills

  return compute(runs)

for enemy in ["1-2-3-4-5-6-7-8", "1-2-6"]:
  print(enemy)
  for fitness in ["classic", "custom"]:
    print(fitness)
    dir = f"data/{enemy}/{fitness}"
    experiments_files = os.listdir(dir)
    experiments = []
    gains = []

    for experiment_file in experiments_files:
      if not experiment_file.endswith(".txt"): continue
      
      agent = np.loadtxt(f"{dir}/{experiment_file}")

      kills = compute_kills(run(agent))

      print(kills)
