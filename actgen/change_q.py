import argparse
import logging
import random
import csv

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import gym
import seeding
import torch

from . import utils
from . import wrappers as wrap
from .agents import DQNAgent, DirectedQNet, ActionDQNAgent
from .gscore import plot_confusion_matrix, calc_g_score, build_confusion_matrix

logging.basicConfig(level=logging.INFO)


class ManipulationTrial:
    def __init__(self, test=True):
        args = self.parse_args()
        self.params = self.load_hyperparams(args)
        if self.params['test'] or test:
            self.params['test'] = True
        self.setup()

    def parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # yapf: disable
        parser.add_argument('--env_name', type=str, default='LunarLander-v2',
                            help='Which gym environment to use')
        parser.add_argument('--agent', type=str, default='dqn',
                            choices=['dqn', 'random', 'action_dqn'],
                            help='Which agent to use')
        parser.add_argument('--duplicate', '-d', type=int, default=5,
                            help='Number of times to duplicate actions')
        parser.add_argument('--seed', '-s', type=int, default=0,
                            help='Random seed')
        parser.add_argument('--hyperparams', type=str, default='hyperparams/lunar_lander.csv',
                            help='Path to hyperparameters csv file')
        parser.add_argument('--load', type=str, default='results/qnet_seed0_best.pytorch',
                            help='Path to a saved model that"s fully trained')
        parser.add_argument('--out_file', type=str, default='results/change_q_metric.csv',
                            help='Path to a output file to write to that will contain the computed metrics')
        parser.add_argument('--test', default=False, action='store_true',
                            help='Enable test mode for quickly checking configuration works')
        parser.add_argument('--num_update', '-n', type=int, default=1,
                            help='Number of times to update a particular action q value')
        parser.add_argument('--delta_update', '-u', type=float, default=10.0,
                            help='increase the q value by this much for every update applied')
        parser.add_argument('--change_percentage_thresh', '-p', type=float, default=0.10,
                            help='only changes past this percentage are considered when computing the metrics')
        parser.add_argument('--optimizer', type=str, default='sgd',
                            help='Which optimizer to use when manipulating q values')
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
        test_env = gym.make(self.params['env_name'])
        test_env = wrap.FixedDurationHack(test_env)
        if isinstance(test_env.action_space, gym.spaces.Box):
            test_env = wrap.DiscreteBox(test_env)
        test_env = wrap.DuplicateActions(test_env, self.params['duplicate'])
        test_env = wrap.TorchInterface(test_env)
        seeding.seed(1000 + self.params['seed'], gym, test_env)
        self.test_env = test_env

        if self.params['agent'] == 'dqn':
            self.agent = DQNAgent(test_env.observation_space, test_env.action_space, self.params)
        elif self.params['agent'] == 'action_dqn':
            self.agent = ActionDQNAgent(test_env.observation_space, test_env.action_space, self.params)
        # load saved model
        if not self.params['test']:
            print("loading model from ", self.params['load'])
            self.agent.q.load(self.params['load'])

    def teardown(self):
        pass

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
            action_taken = self.agent.act(s)
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
                             lr=self.params['learning_rate'],
                             agent_type=self.params['agent'],
                             optim=self.params['optimizer'])
        q_deltas = q_net.directed_update(states, actions, self.params['delta_update'], self.params['num_update'], self.agent.q)

        # write to csv of direction of change (both for an average state)
        q_deltas_avg = np.mean(q_deltas, axis=0)  # average q_deltas over state
        assert q_deltas_avg.shape == (len(actions), num_total_actions)

        with open(self.params['out_file'], 'w') as f:
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
            print(f"+g score: {plus_g} \n -g score: {minus_g}")
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
