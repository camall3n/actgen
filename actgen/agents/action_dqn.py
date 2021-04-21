import torch

from ..nnutils import one_hot, MLP
from .dqn import DQNAgent


class ActionDQNAgent(DQNAgent):
    def __init__(self, observation_space, action_space, get_duplicate_actions, params):
         super().__init__(observation_space, action_space, get_duplicate_actions, params)

    def _get_q_targets(self, batch):
        with torch.no_grad():
            # Compute Double-Q targets
            next_states = torch.stack(batch.next_state, dim=0).float().to(self.params['device'])
            aps = []
            for s in next_states:
                q_values = self._get_q_values_for_state(s).squeeze().float()
                ap = torch.argmax(q_values, dim=0)
                aps.append(ap)
            aps = torch.as_tensor(aps).to(self.params['device'])
            aps_one_hot = one_hot(aps, self.action_space.n)
            assert aps_one_hot.shape == (len(aps), self.action_space.n)
            vp = self.q_target(torch.cat([next_states, aps_one_hot], dim=-1).float()).squeeze(-1)
            not_done_idx = ~torch.stack(batch.done).to(self.params['device'])
            targets = torch.stack(batch.reward) + self.params['gamma'] * vp * not_done_idx
        return targets.unsqueeze(-1)

    def _get_q_predictions(self, batch):
        states = torch.stack(batch.state, dim=0).float().to(self.params['device'])
        actions = one_hot(torch.stack(batch.action, dim=0).to(self.params['device']), depth=self.action_space.n).float()
        q_values = self.q(torch.cat([states, actions], dim=-1).float())
        return q_values

    def _get_all_one_hot_actions(self):
        return torch.eye(self.action_space.n).float().to(self.params['device'])

    def _get_q_values_for_state(self, x):
        states = torch.as_tensor(x).float().repeat(self.action_space.n, 1).to(self.params['device'])
        assert states.shape == (self.action_space.n, len(x))
        qnet_input = torch.cat([states, self._get_all_one_hot_actions()], dim=-1).float()
        return self.q(qnet_input)

    def _make_qnet(self, n_features, n_actions, params):
        dropout = params['dropout_rate']
        return MLP(n_inputs=n_features+n_actions,
                   n_outputs=1,
                   n_hidden_layers=params['n_hidden_layers'],
                   n_units_per_layer=params['n_units_per_layer'],
                   device=params['device'],
                   dropout=dropout)
