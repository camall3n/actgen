import argparse
import copy
import logging
import random

import numpy as np
import gym
import seeding
import torch
from tqdm import tqdm

from . import utils
from . import wrappers as wrap
from .agents import RandomAgent, DQNAgent
from .utils import Experience

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
        parser.add_argument('--test', default=False, action='store_true',
                            help='Enable test mode for quickly checking configuration works')
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
        seeding.seed(1000+self.params['seed'], gym, test_env)
        self.env = env
        self.test_env = test_env

        if self.params['agent'] == 'random':
            self.agent = RandomAgent(env.observation_space, env.action_space)
        elif self.params['agent'] == 'dqn':
            self.agent = DQNAgent(env.observation_space, env.action_space, self.params)
        self.best_score = -np.inf

    def teardown(self):
        pass

    def update_agent(self):
        loss = []
        for count in range(self.params['updates_per_env_step']):
            temp = self.agent.update()
            loss.append(temp)
        loss = sum(loss)/len(loss)
        return loss

    def evaluate(self, step):
        ep_scores = []
        for ep in range(self.params['n_eval_episodes']):
            s, G, done, t = self.test_env.reset(), 0, False, 0
            while not done:
                a = self.agent.act(s)
                sp, r, done, _ = self.test_env.step(a)
                s, G, t = sp, G + r, t + 1
            ep_scores.append(G.detach())
        avg_episode_score = (sum(ep_scores)/len(ep_scores)).item()
        logging.info("Evaluation: step {}, average score {}".format(
            step, avg_episode_score))
        if avg_episode_score > self.best_score:
            self.best_score = avg_episode_score
            is_best = True
        else:
            is_best = False
        self.agent.save(is_best, self.params['seed'])

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
        self.teardown()


def main(test=False):
    trial = Trial(test=test)
    trial.run()

if __name__ == "__main__":
    main()
