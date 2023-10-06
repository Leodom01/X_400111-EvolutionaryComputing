from environment import eval_environment
import numpy as np

enemies = range(1, 9)
agent = np.loadtxt("./tmp-agent.txt")

enemy_killed = 0
for enemy in enemies:
	_, env = eval_environment([enemy])
	fitness, player_hp, enemy_hp, time = env.play(agent) # type: ignore
	print(f"Enemy {enemy}, fitness: {fitness}, player: {player_hp}, enemy: {enemy_hp}")
	if enemy_hp == 0:
		enemy_killed += 1

print(f"We killed {enemy_killed} enemies")
	
