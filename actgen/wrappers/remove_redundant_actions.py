import gym


class RemoveRedundantActions(gym.Wrapper):
	"""
	remove the redundant actions for a gym environment
	currently, this wrapper is only implemented for MsPacman
	"""
	def __init__(self, env):
		super().__init__(env)
		if 'MsPacman' not in env.spec.id:
			raise NotImplementedError('remove redundancy action wrapper is only implemented for MsPacman')
		self.action_space = gym.spaces.Discrete(5)
		
	def step(self, action):
		if action not in self.action_space:
			raise RuntimeError("trying to take action not in action space")
		return self.env.step(action)


def test_remove_redundant_actions_env():
	env = gym.make('MsPacmanNoFrameskip-v4')
	env = RemoveRedundantActions(env)

	for action in range(5):
		assert action in env.action_space
	assert 9 not in env.action_space
	assert 5 not in env.action_space
	assert 6 not in env.action_space

	assert env.action_space.n == 5

	assert 0 <= env.action_space.sample() <= 4

	assert env.observation_space == gym.make('MsPacmanNoFrameskip-v4').observation_space


if __name__ == "__main__":
	test_remove_redundant_actions_env()