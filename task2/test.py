from environment import eval_environment
import numpy as np

enemies = range(1, 9)
agent = np.loadtxt("./exporing_agents/kill-8.txt")
_, env = eval_environment(enemies)

enemy_killed = 0
for enemy in enemies:
  f, p, e, t = env.run_single(pcont=agent, enemyn=enemy, econt=None)
  print(f"Enemy {enemy}, fitness: {f}, player: {p}, enemy: {e}")
  if e == 0:
    enemy_killed += 1

print(f"We killed {enemy_killed} enemies")
	
