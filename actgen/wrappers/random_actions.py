import math
import random

import gym


class RandomActions(gym.Wrapper):
	"""
	duplicate actions in a gym environment. The first n actions are the same as
	the original env, but the rest are all random actions, with uniform probability
	for each of the original actions.
	"""
	def __init__(self, env, n_dup):
		"""
        :param env: a gym environment that has discrete actions
        :param n_dup: the number of times to duplicate the original actions
        """
		super(RandomActions, self).__init__(env)
		self.n_dup = n_dup
		self.action_space = gym.spaces.Discrete(n_dup * env.action_space.n)
	
	def step(self, action):
		"""
        :param action: a number, in range (0, n_dup)
        :return: (next_state, reward, done, info)
        """
		if action not in self.action_space:
			raise RuntimeError("trying to take action not in action space")
		# the corresponding original action of the original env
		if action < self.env.action_space.n:
			return self.env.step(action)
		else:
			random_action = random.randrange(0, self.env.action_space.n, 1)
			return self.env.step(random_action)
	
	def get_duplicate_actions(self, a):
		"""
		return all the actions that's a duplicate 'a' in a list
		all the random actions are duplicates of each other, no other actions are duplicates
		"""
		# one of the original acitons
		if a < self.env.action_space.n:
			return [a]
		else:
			# a random action
			return list(range(self.env.action_space.n, self.n_dup * self.env.action_space.n))


def test_random_actions_env():
	env = RandomActions(gym.make("CartPole-v0"), 3)  # env with 6 total actions

	assert 6 not in env.action_space
	assert 3 in env.action_space

	assert env.action_space == gym.spaces.Discrete(6)

	assert env.action_space.n == 6

	assert env.observation_space == gym.make("CartPole-v0").observation_space
	

if __name__ == '__main__':
	test_random_actions_env()
	
