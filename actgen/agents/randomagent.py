import torch


class RandomAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def save(self, fname, dir, is_best):
        pass

    def act(self, state, testing=False):
        return self.action_space.sample()

    def store(self, experience):
        pass

    def update(self):
        loss = torch.as_tensor(0)
        return loss
