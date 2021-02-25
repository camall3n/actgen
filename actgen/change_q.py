import argparse
import logging
import random
import csv

from matplotlib import pyplot as plt
import numpy as np
import gym
import seeding
import torch

from . import utils
from . import wrappers as wrap
from .agents import DQNAgent

logging.basicConfig(level=logging.INFO)


class Trial:
    def __init__(self, test=True):
        args = self.parse_args()
        self.params = self.load_hyperparams(args)
        if self.params['test'] or test:
            self.params['test'] = True
        self.setup()

    def parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # yapf: disable
        parser.add_argument('--env_name', type=str, default='CartPole-v0',
                            help='Which gym environment to use')
        parser.add_argument('--agent', type=str, default='dqn',
                            choices=['dqn', 'random'],
                            help='Which agent to use')
        parser.add_argument('--duplicate', '-d', type=int, default=5,
                            help='Number of times to duplicate actions')
        parser.add_argument('--seed', '-s', type=int, default=0,
                            help='Random seed')
        parser.add_argument('--hyperparams', type=str, default='hyperparams/defaults.csv',
                            help='Path to hyperparameters csv file')
        parser.add_argument('--saved_model', type=str, default='results/qnet_best.pytorch',
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
        test_env = wrap.DuplicateActions(test_env, self.params['duplicate'])
        test_env = wrap.TorchInterface(test_env)
        seeding.seed(1000 + self.params['seed'], gym, test_env)
        self.test_env = test_env

        assert self.params['agent'] == 'dqn'
        self.agent = DQNAgent(test_env.observation_space, test_env.action_space, self.params)
        # load saved model
        if not self.params['test']:
            self.agent.q.load(self.params['saved_model'])

    def teardown(self):
        pass

    def original_q_values(self, state):
        """
        get the q values for all actions for a particular state
        """
        self.agent.q.eval()
        q_values = self.agent.q(state.float())
        return q_values.detach().numpy()[0]

    def dqn_directed_update(self, state, action, toward_target):
        """
        update the dqn agent towards a target for one state.
        specifically, update update q(state, action) towards toward_target.
        returns the old q(state,a) and the updated q(state, a) for all a.
        """
        self.agent.q.train()
        self.agent.optimizer.zero_grad()

        old_q_values = self.agent.q(state.float())  # q(state, a) for all a
        q_target = old_q_values.clone()
        q_target[0][action] = toward_target

        loss = torch.nn.functional.smooth_l1_loss(input=old_q_values, target=q_target)
        loss.backward()
        self.agent.optimizer.step()

        self.agent.q.eval()
        new_q_values = self.agent.q(state.float())

        return old_q_values.detach().numpy()[0], new_q_values.detach().numpy()[0]

    def direction_of_change(self, old_values, new_values, action, thresh):
        """
        given a set of old and new q values, compare and see if similar/different actions were updated in
        in the correct direction.

        @:param
            old_values, new_values are both np.array
            old_values.shape = (env.action_space.n,)
            new_values.shape = (num_updates, env.action_space.n)
            action: the action (an int) whose q value was repeatedly updated
            thresh: the threshold for counting some q value as changed

        @:return
            number of similar actions that are updated in the same direction
            number of similar actions that are updated in the different direction
            number of different actions that are updated in the same direction
            number of different actions that are updated in the different direction
        """
        # identify similar & different actions
        num_total_actions = int(self.test_env.action_space.n)
        num_different_actions = int(num_total_actions / self.params['duplicate'])
        similar_actions = range(int(action % num_different_actions), num_total_actions, num_different_actions)
        diff_actions = np.delete(range(num_total_actions), similar_actions)

        # get q values for similar & different actions
        difference = new_values - old_values
        similar_actions_values = np.take(difference, similar_actions, axis=-1)
        diff_actions_values = np.take(difference, diff_actions, axis=-1)
        assert similar_actions_values.shape == (len(difference), int(num_total_actions / num_different_actions))
        assert diff_actions_values.shape == (len(difference), int(num_total_actions / num_different_actions))

        # apply the thresh hold, ignore small changes
        similar_actions_values = np.where(abs(similar_actions_values) > thresh, similar_actions_values, 0)
        diff_actions_values = np.where(abs(diff_actions_values) > thresh, diff_actions_values, 0)

        # check if in right direction
        num_same_dir_similar = sum(np.mean(similar_actions_values, axis=0) > 0) - 1
        num_diff_dir_similar = sum(np.mean(similar_actions_values, axis=0) < 0)
        num_same_dir_diff = sum(np.mean(diff_actions_values, axis=0) > 0)
        num_diff_dir_diff = sum(np.mean(diff_actions_values, axis=0) < 0)
        return num_same_dir_similar, num_diff_dir_similar, num_same_dir_diff, num_diff_dir_diff


def plot(old_values, new_values, action_idx):
    """
    given a set of old and new q values, plot them in a graph to compare
    this function plots the change in q_values in one plot for one action

    old_values.shape = (env.action_space.n,)
    new_values.shape = (num_updates, env.action_space.n)
    """
    n = len(old_values)
    possible_actions = np.arange(n)

    # plot reference line at 0
    plt.plot(possible_actions, np.zeros_like(possible_actions), '--k')

    # plot all new values
    for i, new_val in enumerate(new_values):
        plt.plot(possible_actions, new_val - old_values, label='iter {}'.format(i))

    plt.title('action {}'.format(action_idx))
    plt.xlabel('{} different actions'.format(n))
    plt.ylabel(r'$\Delta$q(s,a)')


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

    # prepare to plot, open csv to write to
    plt.figure()
    metrics_out_file = open(out_file_path, 'w')
    csv_writer = csv.writer(metrics_out_file)
    csv_writer.writerow(['number of similar actions that are updated in the same direction',
                         'number of similar actions that are updated in the different direction',
                         'number of different actions that are updated in the same direction',
                         'number of different actions that are updated in the different direction'])

    # figure out what are the possible actions
    actions = range(num_total_actions)

    # updates for each action
    for a in actions:
        # set up
        trial = Trial(test=test)
        s = trial.test_env.reset()  # one starting state
        original_q_values = trial.original_q_values(s)  # q(s,a) for all a in actions

        update_value = original_q_values[a] + delta_update
        new_values = []
        # update for a few consecutive times
        for i in range(num_updates):
            old_val, new_val = trial.dqn_directed_update(s, a, update_value)
            new_values.append(new_val)
            assert all(original_q_values) == all(old_val), "oops, original q values changes for some reason"

        # plot for current action
        max_change = np.max(new_values) - original_q_values[a]
        plt.subplot(int(num_total_actions / num_different_actions), num_different_actions, a + 1)
        plt.ylim(-1.2 * max_change, 1.2 * max_change)
        plot(original_q_values, new_values, a)

        # get metric for current action (precision recall)
        similar_act_same_dir, similar_act_diff_dir, diff_act_same_dir, diff_act_diff_dir = \
            trial.direction_of_change(original_q_values, new_values, a,
                                      thresh=change_percentage_thresh * max_change)
        csv_writer.writerow([similar_act_same_dir, similar_act_diff_dir, diff_act_same_dir, diff_act_diff_dir])

    # show plot, close csv
    plt.show()
    metrics_out_file.close()


if __name__ == "__main__":
    main()
