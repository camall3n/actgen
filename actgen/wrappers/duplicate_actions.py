import gym


class DuplicateActions(gym.Wrapper):
    """
    duplicate the original actions in a gym environment
    """
    def __init__(self, env, n_dup):
        """
        :param env: an unwrapped gym environment that has discrete actions
        :param n_dup: the number of times to duplicate the original actions
                        e.g. n_dup = 3 in a cartpole environment will result in a total of 6 actions
                            in the order of (L1, R1, L2, R2, L3, R3)
        """
        super(DuplicateActions, self).__init__(env)
        self.n_dup = n_dup
        self.num_actions = n_dup * env.action_space.n
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = env.observation_space
        self.env = env
        self._max_episode_steps = env._max_episode_steps

    def step(self, action):
        """
        :param action: a number, in range (0, n_dup)
        :return: (next_state, reward, done, info)
        """
        original_a = action % self.env.action_space.n  # the corresponding original action of the original env
        return self.env.step(original_a)


def test_duplicat_action_env():
    da = DuplicateActions(gym.make("CartPole-v0"), 2)
    print(da.action_space.n)
    print(da.action_space.sample())
    print(da.observation_space)


if __name__ == '__main__':
    test_duplicat_action_env()
