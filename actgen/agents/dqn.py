import math

import numpy as np
import torch

from ..nnutils import extract, MLP, one_hot
from .replaymemory import ReplayMemory
from ..models import NatureDQN
from .inverse_predictor import InversePredictor

class DQNAgent():
    def __init__(self, observation_space, action_space, get_duplicate_actions_fn, params):
        self.observation_space = observation_space
        self.action_space = action_space
        self.get_duplicate_actions_fn = get_duplicate_actions_fn
        self.params = params

        self.replay = ReplayMemory(self.params['replay_buffer_size'])

        self.n_training_steps = 0
        if len(self.observation_space.shape) == 1:
            input_shape = self.observation_space.shape[0]
        else:
            input_shape = self.observation_space.shape
        self.q = self._make_qnet(input_shape, self.action_space.n, self.params)
        self.q_target = self._make_qnet(input_shape, self.action_space.n, self.params)
        self.q_target.hard_copy_from(self.q)
        self.replay.reset()
        params = list(self.q.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.params['learning_rate'])

        if self.params['inv_model']:
            self.inverse_predictor = InversePredictor(action_space.n, self.params, discrete=True)

    def save(self, fname, dir, is_best):
        self.q.save(fname, dir, is_best)
        if self.params['inv_model']:
            self.inverse_predictor.save(fname, dir, is_best)

    def act(self, x, testing=False):
        if ((len(self.replay) < self.params['replay_warmup_steps']
                or np.random.uniform() < self._get_epsilon(testing=testing))):
            a = self.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self._get_q_values_for_state(x)
                a = torch.argmax(q_values.squeeze())
        return a

    def store(self, experience):
        self.replay.push(experience)

    def update(self):
        if len(self.replay) >= self.params['batch_size']:
            batch = self.replay.sample(self.params['batch_size'])

            # always update inverse model
            if self.params['inv_model']:
                self.inverse_predictor.update(batch, encoder=self.q.encoder)

        if len(self.replay) < self.params['replay_warmup_steps']:
            return 0.0

        self.q.train()
        self.optimizer.zero_grad()

        q_values = self._get_q_predictions(batch).to(self.params['device'])
        q_targets = self._get_q_targets(batch).to(self.params['device'])

        loss = torch.nn.functional.smooth_l1_loss(input=q_values, target=q_targets)
        param = torch.cat([x.view(-1) for x in self.q.parameters()])
        if self.params['regularization'] == 'l1':
            loss += self.params['regularization_weight_l1'] * torch.norm(param, 1)
        elif self.params['regularization'] == 'l2':
            loss += self.params['regularization_weight_l2'] * torch.norm(param, 2) ** 2
        loss.backward()
        self.optimizer.step()

        self.n_training_steps += 1

        if self.params['use_soft_copy']:
            # soft copy
            self.q_target.soft_copy_from(self.q, self.params['target_copy_tau'])
        else:
            # hard copy
            if self.n_training_steps % self.params['target_copy_period'] == 0:
                self.q_target.hard_copy_from(self.q)

        return loss.detach().item()

    def _get_epsilon(self, testing=False):
        if testing:
            epsilon = self.params['epsilon_during_eval']
        else:
            epsilon = (self.params['epsilon_final'] + (1 - self.params['epsilon_final'])
                 * math.exp(-1. * self.n_training_steps / self.params['epsilon_decay_period']))
        return epsilon

    def _get_action_similarities(self, batch):
        """
        return tensor of size (batch_size, n_actions) with values between 0 and 1.
        1 if "fully similar"; 0 if "fully different".
        """
        action_taken = torch.tensor(batch.action).to(self.params['device'])
        similarity_mat = one_hot(action_taken, self.action_space.n)  # (batch_size, n_actions)
        if self.params['oracle']:
            # all duplicate actions are fully similar
            # for each action taken in the batch
            for i, acted in enumerate(action_taken):
                action_similarity_scores = self.get_duplicate_actions_fn(acted)
                similarity_mat[i] = torch.as_tensor(action_similarity_scores)  # update each row in similarity_mat
        elif self.params['inv_model']:
            # don't use the inverse model in early stages of training
            if self.n_training_steps < self.params['inv_warmup_steps']:
                return similarity_mat
            # use the inverse model
            action_probs = self.inverse_predictor.predict(batch, encoder=self.q.encoder)  # (batch_size, n_actions)
            # scale the probabilities linearly so that the actual action taken has probability 1
            scale = 1 / extract(action_probs, action_taken, idx_dim=-1).view((self.params['batch_size'], 1))
            scaled_action_probs = scale * action_probs
            # those with probabilities lower than 0.5 are deemed none-similar actions
            clipped_action_probs = torch.where(scaled_action_probs < self.params['inv_clip_threshold'], 0, scaled_action_probs)
            similarity_mat = clipped_action_probs
        return similarity_mat

    def _get_q_targets(self, batch):
        with torch.no_grad():
            # Compute Double-Q targets
            next_state = torch.stack(batch.next_state).float().to(self.params['device'])  # (batch_size, dim_state)
            ap = torch.argmax(self.q(next_state), dim=-1)  # (batch_size, )
            vp = self.q_target(next_state).gather(-1, ap.unsqueeze(-1)).squeeze(-1)  # (batch_size, )
            not_done_idx = ~torch.stack(batch.done).to(self.params['device'])  # (batch_size, )
            q_targets = torch.stack(batch.reward).to(self.params['device']) + self.params['gamma'] * vp * not_done_idx  # (batch_size, )
            if self.params['dqn_train_pin_other_q_values']:
                q_predictions = self._get_q_predictions(batch)  # (batch_size, n_actions)
                action_similarities = self._get_action_similarities(batch)  # (batch_size, n_actions)
                td_error = q_targets.unsqueeze(-1) - q_predictions  # (batch_size, n_actions)
                all_action_q_targets = q_predictions + action_similarities * td_error  # (batch_size, n_actions)
                return all_action_q_targets  # (batch_size, n_actions)
        return q_targets

    def _get_q_predictions(self, batch):
        q_values = self.q(torch.stack(batch.state).float().to(self.params['device']))  # (batch_size, n_actions)
        if self.params['dqn_train_pin_other_q_values']:
            return q_values
        q_acted = extract(q_values, idx=torch.stack(batch.action).long(), idx_dim=-1)  #(batch_size,)
        return q_acted

    def _get_q_values_for_state(self, x):
        return self.q(torch.as_tensor(x).float().to(self.params['device']))

    def _make_qnet(self, n_features, n_actions, params):
        dropout = params['dropout_rate']
        if params['architecture'] == 'mlp':
            return MLP(n_inputs=n_features,
                       n_outputs=n_actions,
                       n_hidden_layers=params['n_hidden_layers'],
                       n_units_per_layer=params['n_units_per_layer'],
                       dropout=dropout).to(params['device'])
        elif params['architecture'] == 'nature_dqn':
            return NatureDQN(n_actions=n_actions).to(params['device'])
        else:
            raise NotImplementedError('"{}" is not a known network architecture'.format(params['architecture']))
