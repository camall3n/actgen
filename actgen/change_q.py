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
from .agents import DQNAgent, DirectedQNet

logging.basicConfig(level=logging.INFO)


class Trial:
    def __init__(self, test=True):
        args = self.parse_args()
        self.params = self.load_hyperparams(args)
        self.params['max_env_steps'] = 100
        if self.params['test'] or test:
            self.params['test'] = True
        self.setup()

    def parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # yapf: disable
        parser.add_argument('--env_name', type=str, default='LunarLander-v2',
                            help='Which gym environment to use')
        parser.add_argument('--agent', type=str, default='dqn',
                            choices=['dqn', 'random'],
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
        parser.add_argument('--num_update', '-n', type=int, default=5,
                            help='Number of times to update a particular action q value')
        parser.add_argument('--delta_update', '-u', type=float, default=10.0,
                            help='increase the q value by this much for every update applied')
        parser.add_argument('--change_percentage_thresh', '-p', type=float, default=0.10,
                            help='only changes past this percentage are considered when computing the metrics')
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

        assert self.params['agent'] == 'dqn'
        self.agent = DQNAgent(test_env.observation_space, test_env.action_space, self.params)
        # load saved model
        if not self.params['test']:
            self.agent.q.load(self.params['load'])

    def teardown(self):
        pass


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


def build_confusion_matrix(q_deltas, num_duplicate):
    """
    builds a confusion matrix for the direction of the bumps in q_delta for one state

    :param
        q_deltas (np.array): each row corresponds to the latest q_delta from one action
                            shape = (num_actions, len(q_delta_for_one_action)) = (num_actions, num_actions)
        num_duplicate: number of set of duplicate actions
    :return
           a confusion matrix, where row corresponds to the action being updated, and col in that row correspond to
           the delta for that intervention.
           The index is arranged so that similar actions are adjacent: (a1, a2, .., an, b1, b2, .., bn, ..)
    """
    # arrange index so similar actions are adjacent
    num_actions = len(q_deltas)
    num_original_actions = int(num_actions / num_duplicate)
    new_idx = []
    for i in range(num_original_actions):
        new_idx += list(range(i, num_actions, num_original_actions))
    assert len(new_idx) == num_actions
    q_deltas = q_deltas[new_idx]

    mat = np.zeros((num_actions, num_actions))
    for action, q_delta in enumerate(q_deltas):
        # normalize q_delta by min-max scaling to [0, 1]
        q_delta = (q_delta - np.min(q_delta)) / (np.max(q_delta) - np.min(q_delta))
        # put row in matrix
        q_delta = q_delta[new_idx]
        mat[action] = q_delta
    return mat


def calc_g_score(avg_confusion_mat, num_duplicate):
    """
    calculate the two g scores based on an average confusion matrix

    :param
        avg_confusion_mat: a confusion matrix that's the average of multiple matrix over different states
        num_duplicate: number of set of duplicate actions
    :return: +g, -g
    """
    num_actions = len(avg_confusion_mat)
    num_original_actions = int(num_actions / num_duplicate)
    # matrix diagonal should be all 1's
    for i in range(num_actions):
        assert avg_confusion_mat[i, i] == 1

    # get plus_g score
    plus_g_sum = 0
    for i in range(num_original_actions):
        similar_square_block = avg_confusion_mat[i * num_duplicate: (i + 1) * num_duplicate,
                               i * num_duplicate: (i + 1) * num_duplicate]
        plus_g_sum += np.sum(similar_square_block)
    plus_g = plus_g_sum - 1 * len(avg_confusion_mat)  # subtract 1's from diagonal
    num_plus = len(avg_confusion_mat) * num_duplicate - len(
        avg_confusion_mat)  # number of entries used for the +g score
    plus_g = plus_g / num_plus  # average the sum

    # get minus_g score
    minus_g_sum = np.sum(avg_confusion_mat) - plus_g_sum
    num_minus = len(avg_confusion_mat) ** 2 - num_plus - len(
        avg_confusion_mat)  # number of entries used for the -g score
    minus_g = minus_g_sum / num_minus  # average the sum

    return plus_g, minus_g


def plot_confusion_matrix(mat, num_duplicate, num_states):
    plt.figure()
    plt.title(f"average confusion matrix: {num_duplicate} sets of duplicate action, {num_states} states")
    plt.xlabel(r'normalized $\Delta$q(s,a)')
    plt.ylabel("action being updated")
    plt.imshow(mat)
    plt.colorbar()
    plt.show()


def main(test=False):
    # hyper parameters for plotting
    bogy_trial = Trial()
    num_updates = bogy_trial.params['num_update']  # number of consecutive updates
    delta_update = bogy_trial.params['delta_update']  # add this to the original q value each iteration of the update
    # only changes past this percentage threshold are considered a change.
    # this percentage is the percentage of the maximum increase the q value experienced.
    change_percentage_thresh = bogy_trial.params['change_percentage_thresh']
    num_total_actions = int(bogy_trial.test_env.action_space.n)
    num_different_actions = int(num_total_actions / bogy_trial.params['duplicate'])
    out_file_path = bogy_trial.params['out_file']

    # figure out what are the possible actions
    actions = list(range(num_total_actions))

    # get all states we care about
    states = []
    s, done = bogy_trial.test_env.reset(), False
    for _ in tqdm(range(bogy_trial.params['max_env_steps'])):
        # figure out what the nest state is
        action_taken = bogy_trial.agent.act(s)
        sp, r, done, _ = bogy_trial.test_env.step(action_taken)
        s = sp if not done else bogy_trial.test_env.reset()
        states.append(s)

    # perform directed update for all states and actions
    q_net = DirectedQNet(n_features=bogy_trial.test_env.observation_space.shape[0],
                         n_actions=bogy_trial.test_env.action_space.n,
                         n_hidden_layers=bogy_trial.params['n_hidden_layers'],
                         n_units_per_layer=bogy_trial.params['n_units_per_layer'],
                         lr=bogy_trial.params['learning_rate'])
    q_deltas = q_net.directed_update(states, actions, delta_update, num_updates, bogy_trial.agent.q)

    # write to csv of direction of change (both for an average state)
    q_deltas_avg = np.mean(q_deltas, axis=0)  # average q_deltas over state
    assert q_deltas_avg.shape == (len(actions), num_total_actions)

    with open(out_file_path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['number of similar actions that are updated in the same direction',
                             'number of similar actions that are updated in the different direction',
                             'number of different actions that are updated in the same direction',
                             'number of different actions that are updated in the different direction'])
        for a in actions:
            max_change = np.max(q_deltas_avg[a])
            similar_act_same_dir, similar_act_diff_dir, diff_act_same_dir, diff_act_diff_dir = \
                direction_of_change(q_deltas_avg[a], a, change_percentage_thresh * max_change,
                                    num_total_actions, bogy_trial.params['duplicate'])
            csv_writer.writerow([similar_act_same_dir, similar_act_diff_dir, diff_act_same_dir, diff_act_diff_dir])

    # plot the q_delta (for an average state)
    if not test:
        q_deltas_first = q_deltas[0]
        plt.figure()
        for a in actions:  # for each action updated
            plt.subplot(int(num_total_actions / num_different_actions), num_different_actions, a + 1)
            plot_q_delta(q_deltas_first[a], a)
        plt.show()

    # construct and plot the confusion matrix
    cfn_mat_all_states = np.zeros((len(states),
                                   bogy_trial.test_env.action_space.n,
                                   bogy_trial.test_env.action_space.n))
    for i, _ in enumerate(states):
        cfn_mat_all_states[i, :, :] = build_confusion_matrix(q_deltas[i, :, :], bogy_trial.params['duplicate'])
    avg_cfn_mat = np.mean(cfn_mat_all_states, axis=0)
    plus_g, minus_g = calc_g_score(avg_cfn_mat, bogy_trial.params['duplicate'])
    if not test:
        print(f"+g score: {plus_g} \n -g score: {minus_g}")
        plot_confusion_matrix(avg_cfn_mat, bogy_trial.params['duplicate'], bogy_trial.params['max_env_steps'])


if __name__ == "__main__":
    main()
