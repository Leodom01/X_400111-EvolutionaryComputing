import os

from pymoo.problems.functional import FunctionalProblem
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

import numpy as np

from environment import training_environment

# BEGIN META PARAMETERS
# ENEMIES = [1, 3, 4, 6, 7]
ENEMIES = range(1, 9)
NGEN = 100
BOUND = 1000
# END META PARAMETERS

# BEGIN HYPER PARAMETERS
# END HYPER PARAMETERS

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

neuron_number, env = training_environment(ENEMIES)

def run_single(enemy, phenome):
  f, _, _, _ = env.run_single(pcont=phenome, enemyn=enemy, econt=None)
  return f

objectives = [
  lambda x: -run_single(enemy, x)
  for enemy in ENEMIES
]

problem = FunctionalProblem(
  n_var=neuron_number,
  objs=objectives,
  xl=np.array([-BOUND] * neuron_number),
  xu=np.array([+BOUND] * neuron_number),
)

ref_dirs = get_reference_directions("das-dennis", len(objectives), n_partitions=12)

from pymoo.algorithms.moo.sms import SMSEMOA
# from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.rvea import RVEA

# algorithm = NSGA2(ref_dirs)
algorithm = SMSEMOA()

results = minimize(
  problem,
  algorithm,
  ("n_gen", NGEN),
  verbose=True
)

for i, ind in enumerate(results.opt):
  np.savetxt(f"pyymo-{i}.txt", ind.X)
