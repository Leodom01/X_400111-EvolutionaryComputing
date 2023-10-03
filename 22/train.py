from leap_ec.algorithm import generational_ea
from leap_ec.individual import IdentityDecoder
from leap_ec.representation import Representation
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.problem import FunctionProblem
from leap_ec import ops
from leap_ec.probe import FitnessStatsCSVProbe, pairwise_squared_distance_metric

from matplotlib import shutil
import numpy as np

from environment import training_environment
from custom_crossover import whole_arithmetic_recombination

import time
import math

# HYPER PARAMETERS
MAX_GENERATIONS = 30
POPULATION_SIZE = 100
ELITISM = 0
# END HYPER PARAMETERS

# META PARAMETERS
TRIES = 10
ENEMY = 3
# END META PARAMETERS

# EXPERIMENTS
experiments = [
    ("2pt_crossover", ops.n_ary_crossover(num_points=2)),
    # ("uniform_crossover", ops.uniform_crossover),
    ("whole_arithmetic_recombination", whole_arithmetic_recombination),
    # ("whole_arithmetic_recombination_07", whole_arithmetic_recombination(
    #     alpha=0.7
    # )), # type: ignore
    # ("whole_arithmetic_recombination_09", whole_arithmetic_recombination(
    #     alpha=0.9
    # )), # type: ignore
    # ("random_arithmetic_recombination", random_arithmetic_recombination),
    # ("no_crossover", no_crossover),
]
# END EXPERIMENTS

# To run pygame headless
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

neurons_number, env = training_environment([ENEMY])

def evaluate(phenome):
    f, _, _, _ = env.play(phenome)
    return f

problem = FunctionProblem(
    evaluate,
    maximize=True
)

# Bound for the sampling
a = 10_0000
bounds = [(-a, a)] * neurons_number
representation = Representation(
    decoder=IdentityDecoder(),
    # We inizialize sampling from a normal distribution
    initialize=create_real_vector(bounds)
)

root_dir = f"./data.enemy{ENEMY}"

def run_experiment(dir, experiment_name, crossover):

    if os.path.exists(dir):
        shutil.rmtree(dir)
        
    os.makedirs(dir)

    for try_number in range(TRIES):

        print(f"Starting run {try_number} for experiment {experiment_name}")
        
        stream = open(f"{dir}/run-{try_number}.csv", "w+")
        fitness_probe = FitnessStatsCSVProbe(
            stream=stream,
            extra_metrics={
                # Compute the genetic diversity by distance between vectors
                "sq_disance_diversity": pairwise_squared_distance_metric
            }
        )

        start_time = time.time()
        
        out = generational_ea(
            max_generations=MAX_GENERATIONS,
            pop_size=POPULATION_SIZE,
            problem=problem,
            representation=representation,
            k_elites=ELITISM,
            # Evolution Pipeline
            pipeline=[
                fitness_probe,

                ops.tournament_selection(k=10),

                ops.clone,

                crossover,
                mutate_gaussian(std=0.6, expected_num_mutations='isotropic'),

                ops.evaluate,
                ops.pool(size=POPULATION_SIZE),
            ]
        )

        best_individual = None
        best_fitness = -1
        for _, best in out:
            if best.fitness > best_fitness:
                best_individual = best.decode()
        
        if best_individual is not None:
            np.savetxt(f"{dir}/individual-{try_number}.txt", best_individual)

        end_time = time.time()
        duration = end_time - start_time
        

        print(f"Finished run {try_number} for experiment {experiment_name}"
              f" took {math.floor(duration)} seconds")

        stream.close()

def collect_data(dir):
    summary = []
    for file in os.listdir(dir):
        if not file.startswith("individual"): continue

        individual = np.loadtxt(f"{dir}/{file}")

        _, pl, el, _ = env.play(pcont=individual) # type: ignore

        summary.append(pl - el)

    print("Summary of the best individuals: ", summary)
    np.savetxt(f"{dir}/summary.txt", summary)

# Running the experiments
for experiment_name, crossover in experiments:
    dir = f"{root_dir}/{experiment_name}"

    print(f"Running experiment: {experiment_name}")
    run_experiment(dir, experiment_name, crossover)

    print(f"Collecting data for: {experiment_name}")
    collect_data(dir)

