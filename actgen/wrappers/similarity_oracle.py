import gym
import numpy as np


class SimilarityOracle(gym.Wrapper):
	"""
	provide additional information on how actions in the environment are similar to each other
	currently, this wrapper is only implemented for the 18 action breakout.
	there are only 4 types of actions:
		left
		right
		noop, down, up, 4 diagonal actions
		fire, all 8 other fire actions
	"""
	def __init__(self, env):
		super().__init__(env)
	
	def get_action_similarity_scores(self, a):
		"""
		return a list of similarity scores of 'a' with all the actions in the actions_space
		"""
		NOOP = [0, 2, 5, 6, 7, 8, 9]
		FIRE = [1, 10, 11, 12, 13, 14, 15, 16, 17]
		LEFT = [4]
		RIGHT = [3]
		action_type = None
		# NO-OP action type
		if a in NOOP:
			action_type = NOOP
		# FIRE action type:
		elif a in FIRE:
			action_type = FIRE
		# LEFT action type:
		elif a in LEFT:
			action_type = LEFT
		# RIGHT action type:
		elif a in RIGHT:
			action_type = RIGHT
		else:
			raise RuntimeError('action {} not in action space of 18-action Breakout'.format(a))

		similarity_score = [0] * 18
		for a in action_type:
			similarity_score[a] = 1

		return similarity_score


def test_similarity_oracle_env():
	env = SimilarityOracle(gym.envs.atari.atari_env.AtariEnv('breakout', obs_type='image', frameskip=1, full_action_space=True))

	assert 17 in env.action_space
	assert 0 in env.action_space
	assert 18 not in env.action_space

	assert env.action_space == gym.spaces.Discrete(18)

	assert env.action_space.n == 18

	noop_similarity = env.get_action_similarity_scores(0)
	assert noop_similarity[0] == 1
	assert noop_similarity[2] == 1
	assert sum(noop_similarity) == 7

	fire_similarity = env.get_action_similarity_scores(1)
	assert fire_similarity[1] == 1
	assert fire_similarity[10] == 1
	assert fire_similarity[17] == 1
	assert sum(fire_similarity) == 9

	left_similarity = env.get_action_similarity_scores(4)
	assert left_similarity[4] == 1
	assert sum(left_similarity) == 1

	right_similarity = env.get_action_similarity_scores(3)
	assert right_similarity[3] == 1
	assert sum(right_similarity) == 1


if __name__ == '__main__':
	test_similarity_oracle_env()
