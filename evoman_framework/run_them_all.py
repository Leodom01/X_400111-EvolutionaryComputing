import genetic_algorithm_1
import subprocess

#make 2 runs against enemy 1 with 2point crossover
genetic_algorithm_1.run_experiments(2, 1, True, False)

#make 2 runs against enemy 1 with whole aritm crossover
genetic_algorithm_1.run_experiments(2, 1, False, False)

#make 2 runs against enemy 1 with komma selection (generational)
genetic_algorithm_1.run_experiments(2, 1, True, False)

#make 2 runs against enemy 1 with plus selection (steady state)
genetic_algorithm_1.run_experiments(2, 1, False, True)

subprocess.call(["python", "plot.py"])
