from environment import eval_environment
import numpy as np

enemies = range(1, 9)
agent = np.loadtxt("./tmp-agent.txt")
_, env = eval_environment(enemies)

enemy_killed = 0
tot_player_life = 0
duration = 0
gain = []
for enemy in enemies:
  f, p, e, t = env.run_single(pcont=agent, enemyn=enemy, econt=None)
  print(f"Enemy {enemy}, fitness: {f}, player: {p}, enemy: {e}")
  if e == 0:
    enemy_killed += 1
  duration += t
  tot_player_life += p
  gain.append(p - e)
gain = np.sum(gain)

print(f"We killed {enemy_killed} enemies with total player life {int(tot_player_life)} (max: {800}) in {duration} seconds")
print(f"Gain value is {int(gain)}")