import os
import logging
import shutil

import numpy as np
import torch
import pfrl

from ..model import make_pfrl_nature_dqn
from ..nnutils import one_hot


class PfrlDQNAgent():
	"""
	source: https://github.com/pfnet/pfrl/blob/master/examples/atari/train_dqn_batch_ale.py
	"""
	def __init__(self, observation_space, action_space, params):
		self.observation_space = observation_space
		self.action_space = action_space
		self.params = params

		self.q_func = make_pfrl_nature_dqn(n_actions=action_space.n)
		self.optimizer = torch.optim.RMSprop(
			self.q_func.parameters(),
			lr=self.params['learning_rate'],
			alpha=0.95,
			momentum=0.0,
			eps=1e-2,
			centered=True,
		)
		# self.optimizer = torch.optim.Adam(params, lr=self.params['learning_rate'])
		self.replay = pfrl.replay_buffers.ReplayBuffer(capacity=self.params['replay_buffer_size'])
		explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
			1.0,
			self.params['epsilon_final'],
			self.params['replay_warmup_steps'],
			lambda: np.random.randint(action_space.n)
		)
		# explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=params['epsilon_final'], random_action_func=action_space.sample)

		# feature extractor
		def phi(x):
			return np.asarray(x, dtype=np.float32) / 255

		# Set the device id to use GPU. To use CPU only, set it to -1.
		gpu = -1 if self.params['device'] == 'cpu' else 0
		gpu = -1

		self.agent = pfrl.agents.DoubleDQN(
			self.q_func,
			self.optimizer,
			self.replay,
			self.params['gamma'],
			explorer,
			replay_start_size=self.params['replay_warmup_steps'],
			update_interval=self.params['updates_per_env_step'],
			target_update_interval=self.params['target_copy_period'],
			batch_accumulator="sum",
			phi=phi,
			gpu=gpu,
		)

	# def save(self, fname, dir, is_best):
	# 	os.makedirs(dir, exist_ok=True)
	# 	model_file = os.path.join(dir, '{}_latest.pytorch'.format(fname))
	# 	self.q_func.save(model_file)
	# 	logging.info('Model saved to {}'.format(model_file))
	# 	if is_best:
	# 		best_file = os.path.join(dir, '{}_best.pytorch'.format(fname))
	# 		shutil.copyfile(model_file, best_file)
	# 		logging.info('New best model! Model copied to {}'.format(best_file))
	def save(self, save_dir):
		return self.agent.save(save_dir)
	
	def act(self, x, testing=False):
		return self.agent.act(x)

	def batch_act(self, batch_obs):
		return self.agent.batch_act(batch_obs)
	
	def observe(self, obs, reward, done, reset):
		return self.agent.observe(obs, reward, done, reset)
	
	def get_statistics(self):
		return self.agent.get_statistics()

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
				all_duplicate_actions = self.get_duplicate_actions_fn(acted)
				similarity_mat[i, all_duplicate_actions] = 1  # update each row in similarity_mat
		return similarity_mat