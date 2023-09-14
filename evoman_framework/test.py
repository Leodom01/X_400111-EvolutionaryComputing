from environment import eval_environment
import numpy as np

_, env = eval_environment([1])
solution = np.loadtxt("trained/train-enemy1.txt")

print("Loaded solution:", solution)

print(env.play(pcont=solution))

