import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
from collections import deque


class DQNAgent:
    def __init__(self,
                 action_space,
                 batch_size=32,
                 buffer_size=500,
                 exploration_max=1,
                 exploration_min=0.05,
                 exploration_decay=0.98,
                 reward_discount=0.99):
        self.action_space = action_space
        self.exp_buffer = deque(maxlen=buffer_size)  # a list of experience tuples
        self.batch_size = batch_size
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.gamma = reward_discount
        self.q = None  # TODO:
        self.q_target = None  # TODO:
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=0.001)
        self.q_target_optim = torch.optim.Adam(self.q.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

    def act(self, state, is_testing=False):
        """
        given a state, the agent chooses one of the actions from the action space
        :param state: a tensor representing the state
        :param is_testing: whether during train of test
        :return: an integer in range [1, action_space.n], representing one action
        """
        # when testing, choose the greedy action
        if is_testing:
            q_values = self.q.forward(state)
            return np.argmax(q_values) + 1  # + 1 because action_space starts index at 1

        # when training, choose the e-greedy action
        else:
            if np.random.randn() < self.exploration_rate:
                return random.randrange(self.action_space)
            else:
                q_values = self.q.forward(state)
                return np.argmax(q_values) + 1

    def store(self, experience):
        """
        this store the generated experience into a buffer
        :param experience: an experience tuple
        """
        self.exp_buffer.append(experience)

    def exp_replay(self):
        """
        this function is used in self.update()
        handles experience replay. this samples a batch from the buffer to train on
        :returns (s, a, r, terminal, sp), each of which is a list of length batch_size if the buffer is large enough
                else the length is the buffer list length
        a list of Experience(s, a, r, terminal, sp)
        """
        import random
        if len(self.exp_buffer) < self.batch_size:
            # not enough experience to sample from
            # states, actions_taken, rewards, terminals, sp = list(zip(*self.exp_buffer))
            return
        else:
            # enough experience to sample from
            experience_pairs = random.sample(self.exp_buffer, self.batch_size)
            states, actions_taken, rewards, terminals, sp = list(zip(*experience_pairs))
        return states, actions_taken, rewards, terminals, sp

    def update(self):
        """
        update the model parameters of the q network and target_q network
        also update the exploration rate
        :return: the scalar loss of a batch
        """
        batch = self.exp_replay()
        if batch is None:
            return
        running_loss = 0
        for i, exp_tuple in enumerate(batch):

            # get the bootstrap q_update
            if exp_tuple.done:
                q_update = exp_tuple.reward
            else:
                q_update = exp_tuple.reward + self.gamma * np.max(self.q_target.forward(exp_tuple.next_state))

            # q values as predicted by the q network
            q_values = self.q.forward(exp_tuple.state)

            # the target q values we want to hit
            target_q_values = q_values
            target_q_values[exp_tuple.action] = q_update

            # train the q network
            self.q_optim.zero_grad()
            loss = self.loss(q_values, target_q_values)
            running_loss += loss.item()
            loss.backward()
            self.q_optim.step()

            # train the target network
            # TODO:

        # update exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)

        return running_loss


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
