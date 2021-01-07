import gym
import numpy as np
import torch

class TorchSpace(gym.Space):
    def __init__(self, space):
        self._space = space
        self.shape = torch.Size(space.shape)
        seed_ = space.seed()[0]
        self.dtype = self.sample().dtype
        space.seed(seed_)
        try:
            self.low = torch.as_tensor(space.low)
            self.high = torch.as_tensor(space.high)
        except AttributeError:
            pass
        try:
            self.n = space.n
        except AttributeError:
            pass

    def contains(self, x):
        return self._space.contains(x.detach().numpy())

    def sample(self):
        x = self._space.sample()
        return torch.as_tensor(x)

    def seed(self, seed=None):
        return self._space.seed(seed)

class TorchInterface(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = TorchSpace(env.action_space)
        self.observation_space = TorchSpace(env.observation_space)

    def reset(self):
        state = self.env.reset()
        return torch.as_tensor(state)

    def step(self, action):
        np_action = np.asarray(action)
        state, reward, done, info = self.env.step(np_action)
        state = torch.as_tensor(state, dtype=self.observation_space.dtype)
        reward = torch.as_tensor(reward, dtype=self.action_space.dtype)
        done = torch.as_tensor(done)
        return state, reward, done, info

def test_torch_interface():
    env = gym.make('CartPole-v0')
    env = TorchInterface(env)

    s = env.reset()
    assert isinstance(s, torch.Tensor)

    A = env.action_space
    assert A.dtype is torch.int64

    S = env.observation_space
    assert S.dtype is torch.float32

    a = A.sample()
    assert a.dtype == A.dtype

    s, r, done, info = env.step(a)
    assert s.dtype == S.dtype
    assert isinstance(r, torch.Tensor)
    assert done.dtype == torch.bool

if __name__ == "__main__":
    test_torch_interface()
