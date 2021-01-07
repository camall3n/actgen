import torch

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state, testing=False):
        return self.action_space.sample()

    def store(self, experience):
        pass

    def update(self):
        loss = torch.as_tensor(0)
        return loss
