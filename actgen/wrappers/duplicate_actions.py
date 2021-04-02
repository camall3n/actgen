import gym

import math


class DuplicateActions(gym.Wrapper):
    """
    duplicate the original actions in a gym environment
    """
    def __init__(self, env, n_dup):
        """
        :param env: an unwrapped gym environment that has discrete actions
        :param n_dup: the number of times to duplicate the original actions
                        e.g. n_dup = 3 in a cartpole environment will result in a total of 6 actions
                            in the order of (L1, L2, L3, R1, R2, R3)
        """
        super(DuplicateActions, self).__init__(env)
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
        original_a = math.floor(action / self.n_dup)
        return self.env.step(original_a)


def test_duplicate_action_env():
    env = DuplicateActions(gym.make("CartPole-v0"), 2)  # env with 4 actions

    assert 5 not in env.action_space
    assert 1 in env.action_space

    assert env.action_space == gym.spaces.Discrete(4)

    assert env.action_space.n == 4

    assert 0 <= env.action_space.sample() < 4

    assert env.observation_space == gym.make("CartPole-v0").observation_space


if __name__ == '__main__':
    test_duplicate_action_env()
