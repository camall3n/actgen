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
from .agents import RandomAgent, DQNAgent, DirectedQNet, ActionDQNAgent
from .utils import Experience, Trial
from .gscore import calc_g_score, build_confusion_matrix

logging.basicConfig(level=logging.INFO)


class TrainTrial(Trial):
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
        train_parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            parents=[self.parse_common_args()]
        )
        # train args
        train_parser.add_argument('--regularization', type=str, default='None',
                            choices=['None', 'l1', 'l2', 'dropout'],
                            help='what regularization method to use during training')
        train_parser.add_argument('--gscore', default=False, action='store_true',
                            help='Calculate the g-score vs time as training proceeds')
        train_parser.add_argument('--oracle', default=False, action='store_true',
                            help='to perform oracle action generalization')
        args = self.parse_unknown_args(train_parser)
        return args

    def setup(self):
        seeding.seed(0, random, torch, np)
        env = self.make_gym_env()
        test_env = copy.deepcopy(env)
        seeding.seed(self.params['seed'], gym, env)
        seeding.seed(1000 + self.params['seed'], gym, test_env)
        self.env = env
        self.test_env = test_env

        self.check_params_validity(self.params)
        if self.params['oracle'] and not self.params['dqn_train_pin_other_q_values']:
            raise RuntimeError('dqn_train_pin_other_q_values must be set to true when performing oracle action generalization')

        self.determine_device()

        if self.params['agent'] == 'random':
            self.agent = RandomAgent(env.observation_space, env.action_space)
        elif self.params['agent'] == 'dqn':
            self.agent = DQNAgent(env.observation_space, env.action_space, env.get_action_similarity_scores, self.params)
        elif self.params['agent'] == 'action_dqn':
            self.agent = ActionDQNAgent(env.observation_space, env.action_space, env.get_action_similarity_scores, self.params)
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

    def update_agent(self):
        loss = []
        for count in range(self.params['updates_per_env_step']):
            temp = self.agent.update()
            loss.append(temp)
        loss = sum(loss) / len(loss)
        return loss

    def evaluate(self, step):
        ep_scores = []
        episode_range = range(self.params['n_eval_episodes'])
        if self.params['atari']:
            # evaluations take a while on atari, so add a progress bar
            episode_range = tqdm(episode_range)
        for ep in episode_range:
            s, G, done, t = self.test_env.reset(), 0, False, 0
            while not done:
                a = self.agent.act(s, testing=True).cpu()
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
        # save the rewards
        self.save_rewards(step, avg_episode_score)

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
                             device=self.params['device'],
                             lr=self.params['learning_rate'],
                             agent_type=self.params['agent'],
                             optim=self.params['gscore_optimizer'],
                             pin_other_q_values=self.params['pin_other_q_values']).to(self.params['device'])

        # perform directed q-update for each action, across multiple states
        actions = list(range(self.test_env.action_space.n))
        states = []
        s, done = self.test_env.reset(), False
        for _ in range(self.params['n_gscore_states']):
            # figure out what the nest state is
            action_taken = self.agent.act(s).cpu()
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
        self.save_gscores(step, plus_g, minus_g)

    def save_gscores(self, step, plus_g, minus_g):
        mode = 'w' if step == 0 else 'a'
        with open(self.experiment_dir + self.file_name + "_training_gscore.csv", mode) as f:
            csv_writer = csv.writer(f)
            if step == 0:  # write header
                csv_writer.writerow(['training step', 'plus_g', 'minus_g'])
            csv_writer.writerow([step, plus_g, minus_g])

    def save_rewards(self, step, r):
        mode = 'w' if step == 0 else 'a'
        with open(self.experiment_dir + self.file_name + "_training_reward.csv", mode) as f:
            csv_writer = csv.writer(f)
            if step == 0:  # write header
                csv_writer.writerow(['training step', 'reward during evaluation callback'])
            csv_writer.writerow([step, r])

    def save_batch_loss(self, step, loss):
        mode = 'w' if step == 0 else 'a'
        with open(self.experiment_dir + self.file_name + "_training_loss.csv", mode) as f:
            csv_writer = csv.writer(f)
            if step == 0:  # write header
                csv_writer.writerow(['training step', 'batch loss'])
            csv_writer.writerow([step, loss])

    def run(self):
        s, done, t = self.env.reset(), False, 0
        for step in tqdm(range(self.params['max_env_steps'])):
            a = self.agent.act(s).cpu()
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
            utils.every_n_times(self.params['save_loss_every_n_steps'], step, self.save_batch_loss, step, loss)
            if self.params['gscore']:
                utils.every_n_times(self.params['gscore_every_n_steps'], step, self.gscore_callback, step)
        self.teardown()


def main(test=False):
    trial = TrainTrial(test=test)
    trial.run()


if __name__ == "__main__":
    main()
