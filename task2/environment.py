from evoman.environment import Environment
from controller import player_controller

_HIDDEN_NEURONS_NUMBER = 10

def training_environment(enemies=[8]):
	'''
	Constructs a training environment that is headless and with the fastest
	simulation speed.
	'''
	env = Environment(
		experiment_name="demo/MAIN",
		enemies=enemies,
		playermode="ai",
		player_controller=player_controller(_HIDDEN_NEURONS_NUMBER),
		enemymode="static",
		level=2,
		speed="fastest",
		multiplemode="yes",
	    visuals=False
	)

	neuron_number = (env.get_num_sensors() + 1) * _HIDDEN_NEURONS_NUMBER \
					+ (_HIDDEN_NEURONS_NUMBER + 1) * 5

	return neuron_number, env

def eval_environment(enemies=[8]):
	'''
	Constructs an evaluation environment that has video output and with the 
	standard simulation speed.
	'''
	env = Environment(
		experiment_name="demo/MAIN_VIS",
		enemies=enemies,
		playermode="ai",
		player_controller=player_controller(_HIDDEN_NEURONS_NUMBER),
		enemymode="static",
		level=2,
		speed="fastest",
		# speed="normal",
		multiplemode="yes",
	    visuals=True
	)

	neuron_number = (env.get_num_sensors() + 1) * _HIDDEN_NEURONS_NUMBER \
					+ (_HIDDEN_NEURONS_NUMBER + 1) * 5

	return neuron_number, env
