import random

from ..utils import Experience


# Adapted from Pytorch docs
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.memory = []
        self.position = 0

    def push(self, experience):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, itemize=False):
        samples = random.sample(self.memory, batch_size)
        samples = list(zip(*samples))
        batch = Experience(*samples)
        return batch

    def __len__(self):
        return len(self.memory)
