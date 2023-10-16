from environment import eval_environment
import os
import numpy as np

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

_, env = eval_environment(range(1, 9))

def run_single(agent, enemy):
	return env.run_single(pcont=agent, enemyn=enemy, econt=None)

enemies = range(1, 9)
max_kill = 0
best = []
for agent in os.listdir("./aa"):
	agent = np.loadtxt(f"./aa/{agent}")

	enemy_killed = 0
	for enemy in enemies:

		fitness, player_hp, enemy_hp, time = run_single(agent, enemy)
		#print(f"Enemy {enemy}, fitness: {fitness}, player: {player_hp}, enemy: {enemy_hp}")
		if enemy_hp == 0:
			enemy_killed += 1

	if enemy_killed > max_kill:
		print(f"We killed {enemy_killed} enemies")
		max_kill = enemy_killed
		best = agent

print(f"Max kileld: {max_kill}")
np.savetxt("best-multi.txt", best)
