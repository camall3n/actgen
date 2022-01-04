import gym


class IdentityWrapper(gym.Wrapper):
	"""
	simply provide a dummy get_action_similarity_scores() method for the env
	the env should be a original gym environment, where no actions are considered similar to each other
	"""
	def __init__(self, env):
		super().__init__(env)
	
	def get_action_similarity_scores(self, a):
		"""
		a dummy function (this should never be called, the reason it's here is to
		conform with the API of other wrappers, so that the code doesn't have a null
		pointer in train.py)
		"""
		pass


def test_identity_env():
	env = IdentityWrapper(gym.make('SpaceInvadersNoFrameskip-v4'))

	assert 0 in env.action_space
	assert 5 in env.action_space
	assert 6 not in env.action_space
	assert env.action_space.n == 6

	assert env.action_space == gym.spaces.Discrete(6)


if __name__ == '__main__':
	test_identity_env()
