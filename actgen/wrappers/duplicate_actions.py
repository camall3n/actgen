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
        self.action_space = gym.spaces.Discrete(n_dup * env.action_space.n)

    def step(self, action):
        """
        :param action: a number, in range (0, n_dup)
        :return: (next_state, reward, done, info)
        """
        if action in self.env.action_space:
            original_a = action % self.env.action_space.n  # the corresponding original action of the original env
            return self.env.step(original_a)
        else:
            raise RuntimeError("trying to take action not in action space")


def test_duplicate_action_env():
    env = DuplicateActions(gym.make("CartPole-v0"), 2)  # env with 4 actions

    assert 5 not in env.action_space
    assert 1 in env.action_space

    assert env.action_space == gym.spaces.Discrete(4)

    assert env.action_space.n == 4

    assert 1 <= env.action_space.sample() <= 4

    assert env.observation_space == gym.make("CartPole-v0").observation_space


if __name__ == '__main__':
    test_duplicate_action_env()
