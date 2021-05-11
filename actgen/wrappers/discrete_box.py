import gym
import numpy as np

class DiscreteBox(gym.Wrapper):
    """
    Discretize a continuous "Box" action space into a discrete action space
    """
    def __init__(self, env):
        """
        :param env: an unwrapped gym environment that has a Box action space of shape (n,)
        """
        super().__init__(env)
        self.shape = env.action_space.shape
        self.n_dims = self.shape[-1]
        self.high = env.action_space.high
        self.low = env.action_space.low
        if len(self.high.shape) == 1 or len(self.low.shape) == 1:
            self.high = np.expand_dims(self.high, 0)
            self.low = np.expand_dims(self.low, 0)
        self.mid = (self.low + self.high)/2
        self.n_dup = self.high.shape[0]
        all_atomic_actions = np.concatenate([self.low, self.mid, self.high], axis=0).T
        self.actions = np.array(np.meshgrid(*all_atomic_actions)).T.reshape(-1, self.n_dims)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def step(self, action):
        """
        :param action: integer in the range [0, 3^n_dims)
        :return: (next_state, reward, done, info)
        """
        if action not in self.action_space:
            raise RuntimeError("trying to take action not in action space")
        original_a = self.actions[action]  # the corresponding original action of the original env
        return self.env.step(original_a)


def test_discrete_box():
    env = DiscreteBox(gym.make("Pendulum-v0"))
    assert env.action_space.n == 3
    env.reset()
    for a in range(env.action_space.n):
        env.step(a)

    env = gym.make("Pendulum-v0")
    env.action_space = gym.spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)
    env = DiscreteBox(env)
    assert env.action_space.n == 9, env.action_space.n


if __name__ == '__main__':
    test_discrete_box()
