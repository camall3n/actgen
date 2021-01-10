import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
from collections import deque


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


class DQNAgent:
    def __init__(self,
                 obser_space: int,
                 action_space: int,
                 seed=0,
                 batch_size=32,
                 buffer_size=500,
                 exploration_max=1,
                 exploration_min=0.05,
                 exploration_decay=0.98,
                 reward_discount=0.99,
                 soft_update_factor=0.001):
        self.action_space = action_space
        self.exp_buffer = deque(maxlen=buffer_size)  # a list of experience tuples
        self.batch_size = batch_size
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.gamma = reward_discount
        self.tau = soft_update_factor
        self.q = DQN(obser_space, action_space, seed=seed).to(device)
        self.q_target = DQN(obser_space, action_space, seed=seed).to(device)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=0.001)
        self.q_target_optim = torch.optim.Adam(self.q.parameters(), lr=0.001)
        self.loss = nn.MSELoss()

    def act(self, state, testing=False):
        """
        given a state, the agent chooses one of the actions from the action space
        :param state: a 1-d tensor or 1-d np.ndarray representing the state
        :param testing: whether during train of test
        :return: an integer in range [1, action_space.n], representing one action
        """
        if type(state) == np.ndarray:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        else:
            state = state.float().unsqueeze(0).to(device)

        self.q.eval()
        with torch.no_grad():
            q_values = self.q(state)
        self.q.train()

        # when testing, choose the greedy action
        if testing:
            return torch.argmax(q_values).item()

        # when training, choose the e-greedy action
        else:
            if np.random.randn() < self.exploration_rate:
                return random.randrange(self.action_space)
            else:
                return torch.argmax(q_values).item()

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
        :returns (s, a, r, terminal, sp), each of which is a tensor of length batch_size if the buffer is large enough
                else the length is the buffer list length
        """
        import random
        if len(self.exp_buffer) < self.batch_size:
            # not enough experience to sample from
            # states, actions_taken, rewards, sp, terminals = list(zip(*self.exp_buffer))
            return
        else:
            # enough experience to sample from
            experience_pairs = random.sample(self.exp_buffer, self.batch_size)
            states, actions_taken, rewards, sp, terminals = list(zip(*experience_pairs))

        # make all into tensors of the right shape
        states = torch.cat(states).view(self.batch_size, -1).to(device, dtype=torch.float32)
        actions_taken = torch.from_numpy(np.array(actions_taken)).to(device)
        rewards = torch.tensor(rewards).to(device, dtype=torch.float32)
        sp = torch.cat(sp).view(self.batch_size, -1).to(device, dtype=torch.float32)
        terminals = torch.tensor(terminals).to(device)

        return states, actions_taken, rewards, sp, terminals

    def update(self):
        """
        update the model parameters of the q network and target_q network using a batch of experience
        also update the exploration rate
        :return: the loss of a batch
        """
        replay = self.exp_replay()
        if replay is None:
            return
        states, actions, rewards, sp, terminals = replay  # each is of size (batch_sz, -1)

        # get the bootstrapped q_update
        self.q_target.eval()
        q_update = []
        for i, terminal in enumerate(terminals):
            if terminal:
                q_update.append(rewards[i])
            else:
                next_state = torch.unsqueeze(sp[i], dim=0)
                q_update.append(rewards[i] + self.gamma * torch.max(self.q_target.forward(next_state)))
        q_update = torch.tensor(q_update)

        # q values as predicted by the q network
        self.q.eval()
        q_values = self.q.forward(states)  # (batch_sz, action_space)

        # target q values we want to hit
        target_q_values = q_values.clone()
        for i, q in enumerate(q_update):
            target_q_values[i][actions[i].item()] = q

        # train the q network
        self.q.train()
        self.q_optim.zero_grad()
        loss = self.loss(q_values, target_q_values).to(device)
        loss.backward()
        self.q_optim.step()

        # soft update the target network
        # target = tau * local + (1 - tau) * target
        for target_param, local_param in zip(self.q_target.parameters(), self.q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1-self.tau) * target_param.data)

        # update exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)

        return loss


class DQN(nn.Module):

    def __init__(self, obser_space, act_space, h1=32, h2=64, seed=0):
        """
        :param obser_space: an int, the dimension of observation space, 4 for cartpole
        :param act_space: an int, the dimension of the action space
        :param h: the size of the hidden layer
        :param seed: random seed
        """
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(obser_space, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, act_space)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: of shape (batch_sz, obser_space)
        :return: of shape (batch_sz, act_space)
        """
        batch_sz = x.size(0)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        assert x.size(0) == batch_sz
        return x


def test_unpack():
    from ..utils import Experience
    a = Experience('s', 'a', 'r', 'ss', False)
    b = Experience('ss', 'aa', 'rr', 'sss', True)
    ls = [a, b]
    packed = list(zip(*ls))

    assert len(packed) == 5
    assert packed[0] == ('s', 'ss')


def test_dqn():
    net = DQN(obser_space=3, act_space=5, seed=0).to(device)

    x = torch.rand((20, 3)).to(device)
    y = net.forward(x)

    assert y.size(0) == 20
    assert y.size(1) == 5


def test_dqn_agent():
    from ..utils import Experience
    agent = DQNAgent(obser_space=3, action_space=5, batch_size=4)

    # test act()
    state = torch.rand((3,)).to(device)
    a1 = agent.act(state, testing=False)
    a2 = agent.act(state, testing=True)
    assert 0 <= a1 <= 4
    assert 0 <= a2 <= 4

    # test store()
    assert len(agent.exp_buffer) == 0
    dummy_state = torch.tensor([1, 2, 3])
    dummy_action = 2
    dummy_reward = 9.7
    a = Experience(dummy_state, dummy_action, dummy_reward, dummy_state, False)
    b = Experience(dummy_state, dummy_action, dummy_reward, dummy_state, True)
    agent.store(a)
    assert len(agent.exp_buffer) == 1
    agent.store(b)
    assert len(agent.exp_buffer) == 2
    assert list(agent.exp_buffer) == [a, b]

    # test exp_replay()
    re1 = agent.exp_replay()
    assert re1 is None
    for _ in range(5):
        agent.store(a)
    states, actions, rewards, sp, terminals = agent.exp_replay()
    assert len(states) == 4
    assert len(actions) == 4
    assert len(rewards) == 4
    assert len(sp) == 4
    assert len(terminals) == 4

    # test update()
    loss = agent.update()
    assert loss.item() > 0


if __name__ == '__main__':
    test_unpack()
    test_dqn()
    test_dqn_agent()
