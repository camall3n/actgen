import argparse
import copy
import csv
import logging
import os
import random

import gym
import numpy as np
import seeding
import torch
from tqdm import tqdm

from . import utils
from . import wrappers as wrap
from .agents import RandomAgent, DQNAgent, DirectedQNet, ActionDQNAgent
from .utils import Experience
from .gscore import calc_g_score, build_confusion_matrix

logging.basicConfig(level=logging.INFO)


class Trial:
    def __init__(self, test=True):
        args = self.parse_args()
        self.params = self.load_hyperparams(args)
        if self.params['test'] or test:
            self.params['test'] = True
            self.params['max_env_steps'] = 1000
            self.params['eval_every_n_steps'] = 100
            self.params['epsilon_decay_period'] = 250
            self.params['n_eval_episodes'] = 2
            self.params['replay_warmup_steps'] = 50
            self.params['gscore'] = True
        self.setup()

    def parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # yapf: disable
        parser.add_argument('--env_name', type=str, default='CartPole-v0',
                            help='Which gym environment to use')
        parser.add_argument('--agent', type=str, default='dqn',
                            choices=['dqn', 'random', 'action_dqn'],
                            help='Which agent to use')
        parser.add_argument('--regularization', type=str, default='None',
                            choices=['None', 'l1', 'l2', 'dropout'],
                            help='what regularization method to use during training')
        parser.add_argument('--duplicate', '-d', type=int, default=5,
                            help='Number of times to duplicate actions')
        parser.add_argument('--seed', '-s', type=int, default=0,
                            help='Random seed')
        parser.add_argument('--hyperparams', type=str, default='hyperparams/defaults.csv',
                            help='Path to hyperparameters csv file')
        parser.add_argument('--test', default=False, action='store_true',
                            help='Enable test mode for quickly checking configuration works')
        parser.add_argument('--gscore', default=False, action='store_true',
                            help='Calculate the g-score vs time as training proceeds')
        parser.add_argument('--oracle', default=False, action='store_true',
                            help='use our expert domain knowledge to perform oracle update')
        parser.add_argument('--results_dir', type=str, default='./results/',
                            help='Path to the result directory to save model files')
        parser.add_argument('--tag', type=str, default='default_exp',
                            help='A tag for the current experiment, used as a subdirectory name for saving models')
        args, unknown = parser.parse_known_args()
        other_args = {
            (utils.remove_prefix(key, '--'), val)
            for (key, val) in zip(unknown[::2], unknown[1::2])
        }
        args.other_args = other_args
        # yapf: enable
        return args

    def load_hyperparams(self, args):
        params = utils.load_hyperparams(args.hyperparams)
        for arg_name, arg_value in vars(args).items():
            if arg_name == 'hyperparams':
                continue
            params[arg_name] = arg_value
        for arg_name, arg_value in args.other_args:
            utils.update_param(params, arg_name, arg_value)
        return params

    def setup(self):
        seeding.seed(0, random, torch, np)
        env = gym.make(self.params['env_name'])
        env = wrap.FixedDurationHack(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = wrap.DiscreteBox(env)
        env = wrap.DuplicateActions(env, self.params['duplicate'])
        env = wrap.TorchInterface(env)
        test_env = copy.deepcopy(env)
        seeding.seed(self.params['seed'], gym, env)
        seeding.seed(1000 + self.params['seed'], gym, test_env)
        self.env = env
        self.test_env = test_env

        if self.params['oracle']:
	        self.params['dqn_train_pin_other_q_values'] = True

        if self.params['agent'] == 'random':
            self.agent = RandomAgent(env.observation_space, env.action_space)
        elif self.params['agent'] == 'dqn':
            self.agent = DQNAgent(env.observation_space, env.action_space, self.params)
        elif self.params['agent'] == 'action_dqn':
            self.agent = ActionDQNAgent(env.observation_space, env.action_space, self.params)
        self.best_score = -np.inf

        if self.params['regularization'] == 'l1':
            regularizer = self.params['regularization'] + "_" + str(self.params['regularization_weight_l1'])
        elif self.params['regularization'] == 'l2':
            regularizer = self.params['regularization'] + "_" + str(self.params['regularization_weight_l2'])
        elif self.params['regularization'] == 'dropout':
            regularizer = self.params['regularization'] + "_" + str(self.params['dropout_rate'])
        else:
            regularizer = 'none'
        self.file_name = self.params['agent'] + '_' \
                    + 'seed' + str(self.params['seed']) + '_' \
                    + regularizer
        self.experiment_dir = self.params['results_dir'] + self.params['tag'] + '/'
        os.makedirs(self.experiment_dir, exist_ok=True)

        utils.save_hyperparams(self.experiment_dir+self.file_name+'_hyperparams.csv', self.params)

        self.gscores = []

    def teardown(self):
        pass

    def update_agent(self):
        loss = []
        for count in range(self.params['updates_per_env_step']):
            temp = self.agent.update()
            loss.append(temp)
        loss = sum(loss) / len(loss)
        return loss

    def evaluate(self, step):
        ep_scores = []
        for ep in range(self.params['n_eval_episodes']):
            s, G, done, t = self.test_env.reset(), 0, False, 0
            while not done:
                a = self.agent.act(s, testing=True)
                sp, r, done, _ = self.test_env.step(a)
                s, G, t = sp, G + r, t + 1
            ep_scores.append(G.detach())
        avg_episode_score = (sum(ep_scores) / len(ep_scores)).item()
        logging.info("Evaluation: step {}, average score {}".format(
            step, avg_episode_score))
        if avg_episode_score > self.best_score:
            self.best_score = avg_episode_score
            is_best = True
        else:
            is_best = False
        # saving the model file
        self.agent.save(self.file_name, self.experiment_dir, is_best)

    def gscore_callback(self, step):
        # make a net qnet to test manipulated q updates on
        if self.params['agent'] == 'dqn':
            n_inputs = self.env.observation_space.shape[0]
            n_outputs = self.env.action_space.n
        if self.params['agent'] == 'action_dqn':
            n_inputs = self.env.observation_space.shape[0] + self.env.action_space.n
            n_outputs = 1
        q_net = DirectedQNet(n_inputs=n_inputs,
                             n_outputs=n_outputs,
                             n_hidden_layers=self.params['n_hidden_layers'],
                             n_units_per_layer=self.params['n_units_per_layer'],
                             lr=self.params['learning_rate'],
                             agent_type=self.params['agent'],
                             optim=self.params['gscore_optimizer'],
                             pin_other_q_values=self.params['pin_other_q_values'])

        # perform directed q-update for each action, across multiple states
        actions = list(range(self.test_env.action_space.n))
        states = []
        s, done = self.test_env.reset(), False
        for _ in range(self.params['n_gscore_states']):
            # figure out what the nest state is
            action_taken = self.agent.act(s)
            sp, r, done, _ = self.test_env.step(action_taken)
            s = sp if not done else self.test_env.reset()
            states.append(s)
        q_deltas = q_net.directed_update(states, actions, self.params['delta_update'],
                                         self.params['num_update'], self.agent.q)

        # get g-score
        cfn_mat_all_states = np.zeros((len(states),
                                       self.test_env.action_space.n,
                                       self.test_env.action_space.n))
        for i, _ in enumerate(states):
            cfn_mat_all_states[i, :, :] = build_confusion_matrix(q_deltas[i, :, :], self.params['duplicate'])
        avg_cfn_mat = np.mean(cfn_mat_all_states, axis=0)
        plus_g, minus_g = calc_g_score(avg_cfn_mat, self.params['duplicate'])

        # store the g-score at this time step
        self.gscores.append((step, plus_g, minus_g))

    def save_gscores(self):
        with open(self.experiment_dir + self.file_name + "_training_gscore.csv", 'w') as f:
            csv_writer = csv.writer(f)
            for g in self.gscores:
                csv_writer.writerow([g[0], g[1], g[2]])

    def run(self):
        s, done, t = self.env.reset(), False, 0
        for step in tqdm(range(self.params['max_env_steps'])):
            a = self.agent.act(s)
            sp, r, done, _ = self.env.step(a)
            t = t + 1
            terminal = torch.as_tensor(False) if t == self.env.unwrapped._max_episode_steps else done
            self.agent.store(Experience(s, a, r, sp, terminal))
            loss = self.update_agent()
            if done:
                s = self.env.reset()
            else:
                s = sp
            utils.every_n_times(self.params['eval_every_n_steps'], step, self.evaluate, step)
            if self.params['gscore']:
                utils.every_n_times(self.params['gscore_every_n_steps'], step, self.gscore_callback, step)
                self.save_gscores()
        self.teardown()


def main(test=False):
    trial = Trial(test=test)
    trial.run()


if __name__ == "__main__":
    main()
