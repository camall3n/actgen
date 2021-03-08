import numpy as np
import torch

from .dqn import QNet


class DirectedQNet(QNet):
    def __init__(self, n_features, n_actions, n_hidden_layers, n_units_per_layer, lr):
        super().__init__(n_features, n_actions, n_hidden_layers, n_units_per_layer)
        self.optimizer = torch.optim.Adam(list(self.parameters()), lr=lr)

    def directed_update(self, states, actions, target, n_updates):
        """
        update the q network towards the specified target for the states and actions
        update q(s, a) towards target for all s in states, a in actions

        :param
            states: a list of states (s1, s2, ..., sn)
            actions: a list of actions (a1, a2, ..., am)
            target: the target towards which to update the q(s,a) to
            n_updates: number of times to update each q(s, a)
        :return:
            q_delta for each state-action-updated pair,
            an np array of shape (len(states), len(actions), self.n_actions)
            each q_delta is a 1D array of (updated_q_val - original_q_val) for all possible actions in q_net
        """
        q_delta = np.zeros((len(states), len(actions), self.n_actions))
        for s_idx, s in enumerate(states):
            for a_idx, a in enumerate(actions):

                # get original q-values & update target
                original_q_values = self.forward(s.float())  # q(s, a') for all a'
                q_target = original_q_values.clone()
                q_target[0][a] = target

                # perform n updates for q(s, a)
                self.train()
                for _ in range(n_updates):
                    self.optimizer.zero_grad()
                    loss = torch.nn.functional.smooth_l1_loss(input=original_q_values, target=q_target)
                    loss.backward()
                    self.optimizer.step()

                self.eval()
                new_q_values = self.forward(s.float())
                q_delta[s_idx, a_idx, :] = new_q_values.detach().numpy()[0]
        return q_delta
