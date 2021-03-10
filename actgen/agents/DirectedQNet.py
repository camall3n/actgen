import numpy as np
import torch

from .dqn import QNet


class DirectedQNet(QNet):
    """
    this represents a Q network that we can manipulate directed update on certain q(s, a)
    """
    def __init__(self, n_features, n_actions, n_hidden_layers, n_units_per_layer, lr, optim='sgd'):
        super().__init__(n_features, n_actions, n_hidden_layers, n_units_per_layer)
        assert optim in ['sgd', 'adam']
        if optim == 'sgd':
            self.optimizer = torch.optim.SGD(list(self.parameters()), lr=lr)
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(list(self.parameters()), lr=lr)

    def directed_update(self, states, actions, delta_update, n_updates, baseline_qnet):
        """
        update the q network towards the specified target for the states and actions
        update q(s, a) towards target for all s in states, a in actions

        :param
            states: a list of states (s1, s2, ..., sn)
            actions: a list of actions (a1, a2, ..., am)
            delta_update: update q(s,a) towards q(s,a) + delta_update
            n_updates: number of times to update each q(s, a)
        :return:
            q_delta for each state-action-updated pair,
            an np array of shape (len(states), len(actions), self.n_actions)
            each q_delta is a 1D array of (updated_q_val - original_q_val) for all possible actions in q_net
        """
        q_delta = np.zeros((len(states), len(actions), self.n_actions))
        for s_idx, s in enumerate(states):
            for a_idx, a in enumerate(actions):
                # reset the q_net for each state
                self.hard_copy_from(baseline_qnet)

                with torch.no_grad():
                    original_q_values = self.forward(s.float())
                    q_target = original_q_values.clone()
                    q_target[0][a] = float(original_q_values[0][a]) + delta_update

                # perform updates for q(s, a)
                self.train()
                for i in range(n_updates):
                    current_q_vals = self.forward(s.float())  # q(s, a') for all a'

                    self.optimizer.zero_grad()
                    loss = torch.nn.functional.smooth_l1_loss(input=current_q_vals, target=q_target)
                    loss.backward()
                    self.optimizer.step()

                self.eval()
                new_q_values = self.forward(s.float())
                q_delta[s_idx, a_idx, :] = new_q_values.detach().numpy()[0] - original_q_values.detach().numpy()[0]
        return q_delta

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
