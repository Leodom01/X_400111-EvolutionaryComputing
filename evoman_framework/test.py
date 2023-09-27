from environment import eval_environment, training_environment
import numpy as np

enemies = [3, 6, 7]

for enemy in enemies:
	print("Enemy ", enemy)
	_, env = training_environment([enemy])

	# base = f"./data.enemy{enemy}/whole_arithmetic_recombination"
	base = f"./data.enemy{enemy}/2pt_crossover"
	for i in range(10):
		solution = np.loadtxt(f"{base}/individual-{i}.txt")
		a = env.play(pcont=solution) # type:ignore
		print(f"Individual {i} fitness: {a}")

