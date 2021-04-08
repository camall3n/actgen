import argparse
import logging
import random

import numpy as np
import gym
import seeding
import torch

from . import utils
from . import wrappers as wrap
from .agents import DQNAgent, ActionDQNAgent

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
                            choices=['CartPole-v0', 'Pendulum-v0', 'LunarLander-v2'],
                            help='Which gym environment to use')
        parser.add_argument('--agent', type=str, default='dqn',
                            choices=['dqn', 'action_dqn'],
                            help='Which agent to use')
        parser.add_argument('--duplicate', '-d', type=int, default=5,
                            help='Number of times to duplicate actions')
        parser.add_argument('--seed', '-s', type=int, default=0,
                            help='Random seed')
        parser.add_argument('--hyperparams', type=str, default='hyperparams/lunar_lander.csv',
                            help='Path to hyperparameters csv file')
        parser.add_argument('--test', default=False, action='store_true',
                            help='Enable test mode for quickly checking configuration works')
        parser.add_argument('--load', type=str, default='results/default_exp/dqn_seed0_none_best.pytorch',
                            help='Path to the saved model file')
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
        seeding.seed(1000+self.params['seed'], gym, test_env)
        self.test_env = test_env

        if self.params['agent'] == 'dqn':
            self.agent = DQNAgent(test_env.observation_space, test_env.action_space, self.params)
        elif self.params['agent'] == 'action_dqn':
            self.agent = ActionDQNAgent(test_env.observation_space, test_env.action_space, self.params)
        # load saved model
        if not self.params['test']:
            self.agent.q.load(self.params['load'])

    def teardown(self):
        pass

    def evaluate(self, step):
        ep_scores = []
        for ep in range(self.params['n_eval_episodes']):
            s, G, done, t = self.test_env.reset(), 0, False, 0
            while not done:
                a = self.agent.act(s, testing=True)
                sp, r, done, _ = self.test_env.step(a)
                s, G, t = sp, G + r, t + 1
            ep_scores.append(G.detach())
        avg_episode_score = (sum(ep_scores)/len(ep_scores)).item()
        logging.info("Evaluation: step {}, average score {}".format(
            step, avg_episode_score))


def main(test=False):
    trial = Trial(test=test)
    trial.evaluate(0)


if __name__ == "__main__":
    main()
