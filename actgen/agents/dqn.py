import numpy as np
import torch

from ..nnutils import Network, Sequential, Reshape, extract
from .replaymemory import ReplayMemory


class QNet(Network):
    def __init__(self, n_features, n_actions, n_hidden_layers, n_units_per_layer):
        super().__init__()
        self.n_features = n_features
        self.n_actions = n_actions

        assert n_hidden_layers >= 1

        layers = [
            Reshape(-1, n_features),
            torch.nn.Linear(n_features, n_units_per_layer),
            torch.nn.ReLU()
        ] + [
            torch.nn.Linear(n_units_per_layer, n_units_per_layer),
            torch.nn.ReLU()
        ] * (n_hidden_layers - 1) + [
            torch.nn.Linear(n_units_per_layer, n_actions)
        ]
        self.model = Sequential(*layers)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class DQNAgent():
    def __init__(self, observation_space, action_space, params):
        self.observation_space = observation_space
        self.action_space = action_space
        self.params = params

        self.replay = ReplayMemory(self.params['replay_buffer_size'])

        self.n_training_steps = 0
        assert len(self.observation_space.shape) == 1
        n_features = self.observation_space.shape[0]
        self.q = self._make_qnet(n_features, action_space.n, self.params)
        self.q_target = self._make_qnet(n_features, action_space.n, self.params)
        self.q_target.hard_copy_from(self.q)
        self.replay.reset()
        params = list(self.q.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.params['learning_rate'])

    def save(self, is_best, seed):
        self.q.save('qnet' + f'_seed{seed}', 'results/', is_best)

    def act(self, x, testing=False):
        if ((len(self.replay) < self.params['replay_warmup_steps']
                or np.random.uniform() < self._get_epsilon(testing=testing))):
            a = self.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self._get_q_values_for_state(x)
                a = torch.argmax(q_values, dim=-1)[0]
        return a

    def store(self, experience):
        self.replay.push(experience)

    def update(self):
        if len(self.replay) < self.params['replay_warmup_steps']:
            return 0.0

        batch = self.replay.sample(self.params['batch_size'])

        self.q.train()
        self.optimizer.zero_grad()

        q_values = self._get_q_predictions(batch)
        q_targets = self._get_q_targets(batch)

        loss = torch.nn.functional.smooth_l1_loss(input=q_values, target=q_targets)
        loss.backward()
        self.optimizer.step()

        self.n_training_steps += 1
        self.q_target.soft_copy_from(self.q, self.params['target_copy_tau'])
        # if self.n_training_steps % self.params['target_copy_period'] == 0:
        #     self.q_target.hard_copy_from(self.q)

        return loss.detach().item()

    def _get_epsilon(self, testing=False):
        if testing:
            epsilon = self.params['epsilon_final']
        else:
            alpha = (len(self.replay) - self.params['replay_warmup_steps']) / self.params['epsilon_decay_period']
            alpha = np.clip(alpha, 0, 1)
            epsilon = self.params['epsilon_final'] * alpha + 1 * (1 - alpha)
        return epsilon

    def _get_q_targets(self, batch):
        with torch.no_grad():
            # Compute Double-Q targets
            next_state = torch.stack(batch.next_state).float()
            ap = torch.argmax(self.q(next_state), dim=-1)
            vp = self.q_target(next_state).gather(-1, ap.unsqueeze(-1)).squeeze(-1)
            not_done_idx = ~torch.stack(batch.done)
            targets = torch.stack(batch.reward) + self.params['gamma'] * vp * not_done_idx
        return targets

    def _get_q_predictions(self, batch):
        q_values = self.q(torch.stack(batch.state).float())
        q_acted = extract(q_values, idx=torch.stack(batch.action).long(), idx_dim=-1)
        return q_acted

    def _get_q_values_for_state(self, x):
        return self.q(torch.as_tensor(x).float())

    def _make_qnet(self, n_features, n_actions, params):
        return QNet(n_features=n_features,
                    n_actions=n_actions,
                    n_hidden_layers=params['n_hidden_layers'],
                    n_units_per_layer=params['n_units_per_layer'])
