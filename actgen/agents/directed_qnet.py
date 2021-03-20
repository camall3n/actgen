import numpy as np
import torch

from ..nnutils import MLP, one_hot


class DirectedQNet(MLP):
    """
    this represents a Q network that we can manipulate directed update on certain q(s, a)
    This Q network has a normal structure of (s) -> q(s, a) for all a
    """
    def __init__(self, n_inputs, n_outputs, n_hidden_layers, n_units_per_layer, 
                 lr, agent_type, optim='sgd'):
        super().__init__(n_inputs, n_outputs, n_hidden_layers, n_units_per_layer)
        assert optim in ['sgd', 'adam']
        self.optim = optim
        self.lr = lr
        assert agent_type in ['dqn', 'action_dqn']
        self.agent_type = agent_type

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
        q_delta = np.zeros((len(states), len(actions), len(actions)))
        for s_idx, s in enumerate(states):
            for a_idx, a in enumerate(actions):
                # reset the q_net for each state
                self.hard_copy_from(baseline_qnet)
                if self.optim == 'sgd':
                    optimizer = torch.optim.SGD(list(self.parameters()), lr=self.lr)
                if self.optim == 'adam':
                    optimizer = torch.optim.Adam(list(self.parameters()), lr=self.lr)

                # update for normal DQN
                if self.agent_type == 'dqn':
                    # get q_target
                    with torch.no_grad():
                        original_q_values = self.forward(s.float())
                        q_target = original_q_values.clone()
                        q_target[0][a] = float(original_q_values[0][a]) + delta_update

                    # perform updates for q(s, a)
                    self.train()
                    for i in range(n_updates):
                        current_q_vals = self.forward(s.float())  # q(s, a') for all a'

                        optimizer.zero_grad()
                        loss = torch.nn.functional.smooth_l1_loss(input=current_q_vals, target=q_target)
                        loss.backward()
                        optimizer.step()

                    self.eval()
                    new_q_values = self.forward(s.float())
                    q_delta[s_idx, a_idx, :] = new_q_values.detach().numpy()[0] - original_q_values.detach().numpy()[0]
                
                # udpate for flipped DQN
                elif self.agent_type == 'action_dqn':

                    one_hot_a = one_hot(torch.as_tensor([a]), depth=len(actions)).float().squeeze()

                    # get q_target
                    with torch.no_grad():
                        original_q_values = self.forward(torch.cat([s.float(), one_hot_a], dim=-1))
                        q_target = original_q_values.clone()
                        q_target[0] = float(original_q_values[0]) + delta_update
                    
                    # perform udpates for q(s, a)
                    self.train()
                    for i in range(n_updates):
                        current_q_val = self.forward(torch.cat([s.float(), one_hot_a], dim=-1))

                        optimizer.zero_grad()
                        loss = torch.nn.functional.smooth_l1_loss(input=current_q_val, target=q_target)
                        loss.backward()
                        optimizer.step()
                    
                    # new q(s, a) for all a
                    self.eval()
                    states = torch.as_tensor(s).float().repeat(len(actions), 1)
                    assert states.shape == (len(actions), len(s))
                    qnet_input = torch.cat([states, torch.eye(len(actions))], dim=-1).float()
                    new_q_values = self.forward(qnet_input).squeeze()
                    assert new_q_values.shape == (len(actions), )
                    q_delta[s_idx, a_idx, :] = (new_q_values.detach().numpy() - original_q_values.detach().numpy()).squeeze()
        return q_delta

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
