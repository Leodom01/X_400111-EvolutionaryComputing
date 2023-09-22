from environment import eval_environment
import numpy as np

_, env = eval_environment([6])

base = "./data/2pt_crossover"
for i in range(10):
	solution = np.loadtxt(f"{base}/individual-{i}.txt")

	f, _, _, _ = env.play(pcont=solution) # type:ignore
	print(f"Individual {i} fitness: {f}")

