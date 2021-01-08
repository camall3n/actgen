import torch
from collections import deque


class DQNAgent:
    def __init__(self,
                 action_space,
                 batch_size=32,
                 buffer_size=500,
                 exploration_max=1,
                 exploration_min=0.05):
        self.action_space = action_space
        self.exp_buffer = deque(maxlen=buffer_size)  # a list of experience tuples
        self.batch_size = batch_size
        self.exploration_rate = exploration_max
        self.model = None  # TODO:

    def act(self, state, is_testing=False):
        if is_testing:
            # when testing, choose the greedy action
            pass  # TODO
        else:
            # when training, choose the e-greedy action
            pass  # TODO:

    def store(self, experience):
        """
        this store the generated experience into a buffer
        :param experience: an experience tuple
        """
        self.exp_buffer.append(experience)

    def exp_replay(self):
        """
        handles experience replay. this samples a batch from the buffer to train on
        :returns (s, a, r, terminal, sp), each of which is a list of length batch_size if the buffer is large enough
                else the length is the buffer list length
        a list of Experience(s, a, r, terminal, sp)
        """
        import random
        if len(self.exp_buffer) < self.batch_size:
            # not enough experience to sample from
            states, actions_taken, rewards, terminals, sp = list(zip(*self.exp_buffer))
        else:
            # enough experience to sample from
            experience_pairs = random.sample(self.exp_buffer, self.batch_size)
            states, actions_taken, rewards, terminals, sp = list(zip(*experience_pairs))
        return states, actions_taken, rewards, terminals, sp

    def update(self):
        # TODO:
        pass
