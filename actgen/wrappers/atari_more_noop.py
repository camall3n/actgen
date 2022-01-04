import gym
import numpy as np


class AtariMoreNoops(gym.Wrapper):
    """
    add a bunch of Noop acitons to an atari gym environment
    """
    def __init__(self, env, n_dup) -> None:
        """
        if the original environment has m discrete actions, this wrapped environment
        gives n_dup*m actions, of which the first m are the original m action, and
        the rest are all no-ops.
        """
        super().__init__(env)
        self.n_dup = n_dup
        self.env = env
        assert type(env.action_space) == gym.spaces.Discrete, "can only handle atari discrete action space"
        self.action_space = gym.spaces.Discrete(n_dup * env.action_space.n)
    
    def step(self, action):
        if action not in self.action_space:
            raise RuntimeError("trying to take action not in action space")
        if action < self.env.action_space.n:
            # original actions
            return self.env.step(action)
        else:
            # do no-op
            return self.env.step(0)


def test_atari_more_noop_env():
    env = AtariMoreNoops(gym.make("MsPacman-v0"), 3)

    assert 27 not in env.action_space
    assert 1 in env.action_space

    assert env.action_space == gym.spaces.Discrete(3 * 9)

    assert env.action_space.n == 3 * 9

    assert env.observation_space == gym.make("MsPacman-v0").observation_space


if __name__ == "__main__":
    test_atari_more_noop_env()