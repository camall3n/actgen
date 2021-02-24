import argparse
import logging
import random
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
        # hyper params to set up testing
        self.params['epsilon_final'] = 0
        self.params['n_eval_episodes'] = 100
        self.params['replay_warmup_steps'] = 0
        self.params['epsilon_decay_period'] = 0
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
        parser.add_argument('--test', default=False, action='store_true',
                            help='Enable test mode for quickly checking configuration works')
        args, unknown = parser.parse_known_args()
        other_args = {
            (remove_prefix(key, '--'), val)
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
        seeding.seed(1000+self.params['seed'], gym, test_env)
        self.test_env = test_env

        assert self.params['agent'] == 'dqn'
        self.agent = DQNAgent(test_env.observation_space, test_env.action_space, self.params)
        # load saved model
        if not self.params['test']:
            self.agent.q.load("results/qnet_best.pytorch")

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

    def plot(self, old_values, new_values, action_idx):
        """
        given a set of old and new q values, plot them in a graph to compare
        this function plots the change in q_values in one plot for one action

        old_values.shape = (env.action_space.n,)
        new_values.shape = (num_updates, env.action_space.n)
        """
        possible_actions = np.arange(self.test_env.action_space.n)

        # plot reference line at 0
        plt.plot(possible_actions, np.zeros_like(possible_actions), '--k')

        # plot all new values
        for i, new_val in enumerate(new_values):
            plt.plot(possible_actions, new_val - old_values, label='iter {}'.format(i))

        plt.title('action {}'.format(action_idx))
        plt.xlabel('{} different actions'.format(self.test_env.action_space.n))
        plt.ylabel('q(s,a)')
        # plt.legend()

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
        difference = new_values - old_values
        if action % 2 == 0:
            similar_actions = range(0, self.test_env.action_space.n, 2)
            diff_actions = range(1, self.test_env.action_space.n, 2)
        else:
            similar_actions = range(1, self.test_env.action_space.n, 2)
            diff_actions = range(0, self.test_env.action_space.n, 2)
        # collect q values for different/similar actions
        similar_actions_values = np.take(difference, similar_actions, axis=-1)
        diff_actions_values = np.take(difference, diff_actions, axis=-1)
        assert similar_actions_values.shape == (len(difference), int(self.test_env.action_space.n / 2))
        assert diff_actions_values.shape == (len(difference), int(self.test_env.action_space.n / 2))
        # apply the thresh hold, ignore small changes
        similar_actions_values = np.where(abs(similar_actions_values) > thresh, similar_actions_values, 0)
        diff_actions_values = np.where(abs(diff_actions_values) > thresh, diff_actions_values, 0)
        # check if in right direction
        num_same_dir_similar = sum(np.mean(similar_actions_values, axis=0) > 0) - 1
        num_diff_dir_similar = sum(np.mean(similar_actions_values, axis=0) < 0)
        num_same_dir_diff = sum(np.mean(diff_actions_values, axis=0) > 0)
        num_diff_dir_diff = sum(np.mean(diff_actions_values, axis=0) < 0)
        return num_same_dir_similar, num_diff_dir_similar, num_same_dir_diff, num_diff_dir_diff


def main(test=False):
    # hyper parameters for plotting
    num_updates = 5  # number of consecutive updates
    delta_update = 10  # add this much to the original q value each iteration of the update
    change_thresh = 0.01  # only changes past this threshold are considered a change

    # prepare to plot
    plt.figure()

    # figure out what are the possible actions
    actions = range(Trial().test_env.action_space.n)

    # updates for each action
    for a in actions:
        # set up
        trial = Trial(test=test)
        s = trial.test_env.reset()  # one starting state
        original_q_values = trial.original_q_values(s)  # q(s,a) for all a in actions

        base_update_value = original_q_values[a] + delta_update
        new_values = []
        for i in range(num_updates):
            update_value = base_update_value + delta_update * i
            old_val, new_val = trial.dqn_directed_update(s, a, update_value)
            new_values.append(new_val)
            assert all(original_q_values) == all(old_val), "oops, original q values changes for some reason"
        # plot for current action
        plt.subplot(5, 2, a+1)
        plt.ylim(-0.1, 0.1)
        trial.plot(original_q_values, new_values, a)
        # get metric for current action (precision recall)
        similar_act_same_dir, similar_act_diff_dir, diff_act_same_dir, diff_act_diff_dir = \
            trial.direction_of_change(original_q_values, new_values, a, thresh=change_thresh)
        print(f"for action {a} metrics are: {similar_act_same_dir}, {similar_act_diff_dir},"
              f" {diff_act_same_dir}, {diff_act_diff_dir}")

    # show plot
    plt.show()


if __name__ == "__main__":
    main()
