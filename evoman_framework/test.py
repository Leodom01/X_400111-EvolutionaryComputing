from environment import eval_environment
import numpy as np

_, env = eval_environment()

solution = np.loadtxt("train.txt")
print("Loaded solution:", solution)

print(env.play(pcont=solution))

