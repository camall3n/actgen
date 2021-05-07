import argparse
import csv
import logging
import os
import random

import gym
from matplotlib import pyplot as plt
import numpy as np
import seeding
import torch
from tqdm import tqdm

from .utils import Trial
from .agents import DQNAgent, DirectedQNet, ActionDQNAgent
from .gscore import plot_confusion_matrix, calc_g_score, build_confusion_matrix

logging.basicConfig(level=logging.INFO)


class ManipulationTrial(Trial):
    def __init__(self, test=True):
        args = self.parse_args()
        self.params = self.load_hyperparams(args)
        self.params['regularization'] = 'None'
        if self.params['test'] or test:
            self.params['test'] = True
        self.setup()
    
    def parse_args(self):
        manipulation_parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            parents=[self.parse_common_args()]
        )
        # manipulation args
        manipulation_parser.add_argument('--load', type=str, default='results/default_exp/dqn_seed0_none_best.pytorch',
                            help='Path to a saved model that"s fully trained')
        manipulation_parser.add_argument('--out_file', type=str, default='change_q_metric.csv',
                            help='Path to a output file to write to that will contain the computed metrics')
        args = self.parse_unknown_args(manipulation_parser)
        return args

    def setup(self):
        seeding.seed(0, random, torch, np)
        test_env = self.make_gym_env()
        seeding.seed(1000 + self.params['seed'], gym, test_env)
        self.test_env = test_env
         
        self.check_params_validity(self.params)
        self.determine_device()

        if self.params['agent'] == 'dqn':
            self.agent = DQNAgent(test_env.observation_space, test_env.action_space, test_env.get_duplicate_actions, self.params)
        elif self.params['agent'] == 'action_dqn':
            self.agent = ActionDQNAgent(test_env.observation_space, test_env.action_space, test_env.get_duplicate_actions, self.params)

        # load saved model
        if not self.params['test']:
            print("loading model from ", self.params['load'])
            self.agent.q.load(self.params['load'])

        self.experiment_dir = self.params['results_dir'] + self.params['tag'] + '/'
        os.makedirs(self.experiment_dir, exist_ok=True)

    def run(self):
        # number of actions
        num_total_actions = int(self.test_env.action_space.n)
        num_different_actions = int(num_total_actions / self.params['duplicate'])

        # figure out what are the possible actions
        actions = list(range(num_total_actions))

        # get all states we care about
        states = []
        s, done = self.test_env.reset(), False
        for _ in tqdm(range(self.params['n_gscore_states'])):
            # figure out what the nest state is
            action_taken = self.agent.act(s).cpu()
            sp, r, done, _ = self.test_env.step(action_taken)
            s = sp if not done else self.test_env.reset()
            states.append(s)

        # perform directed update for all states and actions
        if self.params['agent'] == 'dqn':
            n_inputs = self.test_env.observation_space.shape[0]
            n_outputs = self.test_env.action_space.n
        if self.params['agent'] == 'action_dqn':
            n_inputs = self.test_env.observation_space.shape[0] + self.test_env.action_space.n
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
        q_deltas = q_net.directed_update(states, actions, self.params['delta_update'], self.params['num_update'], self.agent.q)

        # write to csv of direction of change (both for an average state)
        q_deltas_avg = np.mean(q_deltas, axis=0)  # average q_deltas over state
        assert q_deltas_avg.shape == (len(actions), num_total_actions)

        with open(self.experiment_dir + self.params['out_file'], 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['number of similar actions that are updated in the same direction',
                                 'number of similar actions that are updated in the different direction',
                                 'number of different actions that are updated in the same direction',
                                 'number of different actions that are updated in the different direction'])
            for a in actions:
                max_change = np.max(q_deltas_avg[a])
                similar_act_same_dir, similar_act_diff_dir, diff_act_same_dir, diff_act_diff_dir = \
                    direction_of_change(q_deltas_avg[a], a, self.params['change_percentage_thresh'] * max_change,
                                        num_total_actions, self.params['duplicate'])
                csv_writer.writerow([similar_act_same_dir, similar_act_diff_dir, diff_act_same_dir, diff_act_diff_dir])

        # plot the q_delta (for an average state)
        if not self.params['test']:
            q_deltas_first = q_deltas[0]
            plt.figure()
            for a in actions:  # for each action updated
                plt.subplot(int(num_total_actions / num_different_actions), num_different_actions, a + 1)
                plot_q_delta(q_deltas_first[a], a)
            plt.show()

        # construct and plot the confusion matrix
        cfn_mat_all_states = np.zeros((len(states),
                                       self.test_env.action_space.n,
                                       self.test_env.action_space.n))
        for i, _ in enumerate(states):
            cfn_mat_all_states[i, :, :] = build_confusion_matrix(q_deltas[i, :, :], self.params['duplicate'])
        avg_cfn_mat = np.mean(cfn_mat_all_states, axis=0)
        plus_g, minus_g = calc_g_score(avg_cfn_mat, self.params['duplicate'])
        if not self.params['test']:
            print(f"+g score: {plus_g} \n-g score: {minus_g}")
            plot_confusion_matrix(avg_cfn_mat, self.params['duplicate'], len(states))


def direction_of_change(q_deltas, a, thresh, num_total_actions, num_dup_actions):
    """
    given q_delta, see if similar/different actions were updated in
    in the correct direction.

    @:param
        q_deltas: q_deltas for one state, np arary of shape (env.action_space.n, )
        a: the action (an int) whose q value was repeatedly updated
        thresh: the threshold for counting some q value as changed

    @:return
        number of similar actions that are updated in the same direction
        number of similar actions that are updated in the different direction
        number of different actions that are updated in the same direction
        number of different actions that are updated in the different direction
    """
    # identify similar & different actions
    num_different_actions = int(num_total_actions / num_dup_actions)
    similar_actions = range(int(a % num_different_actions), num_total_actions, num_different_actions)
    diff_actions = np.delete(range(num_total_actions), similar_actions)

    # get q values for similar & different actions
    similar_actions_values = q_deltas[similar_actions]
    diff_actions_values = q_deltas[diff_actions]
    assert similar_actions_values.shape == (int(num_total_actions / num_different_actions),)
    assert (diff_actions_values.shape ==
            (num_total_actions - int(num_total_actions / num_different_actions),))

    # apply the thresh hold, ignore small changes
    similar_actions_values = np.where(abs(similar_actions_values) > thresh, similar_actions_values, 0)
    diff_actions_values = np.where(abs(diff_actions_values) > thresh, diff_actions_values, 0)

    # check if in right direction
    num_same_dir_similar = sum(np.array(similar_actions_values) > 0) - 1
    num_diff_dir_similar = sum(np.array(similar_actions_values) < 0)
    num_same_dir_diff = sum(np.array(diff_actions_values) > 0)
    num_diff_dir_diff = sum(np.array(diff_actions_values) < 0)
    return num_same_dir_similar, num_diff_dir_similar, num_same_dir_diff, num_diff_dir_diff


def plot_q_delta(q_deltas, action_idx):
    """
    given q_delta for one state for all actions, plot them in a graph to compare
    this function plots the change in q_values in one plot for one action

    q_deltas.shape = (env.action_space.n,)
    """
    n = len(q_deltas)
    possible_actions = np.arange(n)

    # plot reference line at 0
    plt.plot(possible_actions, np.zeros_like(possible_actions), '--k')

    # plot only the latest new value
    plt.plot(possible_actions, q_deltas)

    plt.title('action {}'.format(action_idx))
    plt.xlabel('{} different actions'.format(n))
    plt.ylabel(r'$\Delta$q(s,a)')


def main(test=False):
    trial = ManipulationTrial(test=test)
    trial.run()


if __name__ == "__main__":
    main()
