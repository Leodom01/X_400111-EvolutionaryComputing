from environment import eval_environment
import numpy as np

_, env = eval_environment([2])
#solution = np.loadtxt("trained/train-enemy2.txt")
solution = np.loadtxt("train.txt")

print("Loaded solution:", solution)

print(env.play(pcont=solution))

