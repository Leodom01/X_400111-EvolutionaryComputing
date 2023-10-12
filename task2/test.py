from environment import eval_environment
import numpy as np

enemies = range(1, 9)
agent = np.loadtxt("./tmp-agent.txt")
_, env = eval_environment(enemies)

enemy_killed = 0
tot_player_life = 0
for enemy in enemies:
  f, p, e, t = env.run_single(pcont=agent, enemyn=enemy, econt=None)
  print(f"Enemy {enemy}, fitness: {f}, player: {p}, enemy: {e}")
  if e == 0:
    enemy_killed += 1
  tot_player_life += p

print("Tmp Agent:")
print(f"We killed {enemy_killed} enemies with total player life {tot_player_life}")

agent = np.loadtxt("./leo-best.txt")
_, env = eval_environment(enemies)

enemy_killed = 0
tot_player_life = 0
for enemy in enemies:
  f, p, e, t = env.run_single(pcont=agent, enemyn=enemy, econt=None)
  print(f"Enemy {enemy}, fitness: {f}, player: {p}, enemy: {e}")
  if e == 0:
    enemy_killed += 1
  tot_player_life += p

print("Leo best:")
print(f"We killed {enemy_killed} enemies with total player life {tot_player_life}")
	
