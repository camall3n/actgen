import argparse
import logging
import random
import os
import csv

import numpy as np
import gym
import seeding
import torch

from .utils import Trial
from .agents import DQNAgent, ActionDQNAgent


logging.basicConfig(level=logging.INFO)


class EvalTrial(Trial):
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

    def setup(self):
        seeding.seed(0, random, torch, np)
        test_env = self.make_gym_env()
        seeding.seed(1000+self.params['seed'], gym, test_env)
        self.test_env = test_env
       
        self.determine_device()

        if self.params['agent'] == 'dqn':
            self.agent = DQNAgent(test_env.observation_space, test_env.action_space, test_env.get_duplicate_actions, self.params)
        elif self.params['agent'] == 'action_dqn':
            self.agent = ActionDQNAgent(test_env.observation_space, test_env.action_space, test_env.get_duplicate_actions, self.params)
        
        self.all_rewards = []

    def evaluate(self):
        # iterate over the directory
        directory = self.params['results_dir'] + self.params['tag']
        for file_name in os.listdir(directory):
            # find best models
            if file_name.endswith("best.pytorch"):
                file_path = os.path.join(directory, file_name)
                # load saved model
                if not self.params['test']:
                    self.agent.q.load(file_path)
                # get reward for current model
                ep_scores = []
                for _ in range(self.params['n_eval_episodes']):
                    s, G, done, t = self.test_env.reset(), 0, False, 0
                    while not done:
                        a = self.agent.act(s, testing=True).cpu()
                        sp, r, done, _ = self.test_env.step(a)
                        s, G, t = sp, G + r, t + 1
                    ep_scores.append(G.detach())
                avg_episode_score = (sum(ep_scores)/len(ep_scores)).item()
                logging.info("Evaluation: average score {}".format(avg_episode_score))
                self.all_rewards.append((file_path, avg_episode_score))
    
    def save_rewards(self):
        reward_file_path = os.path.join(self.params['results_dir'] + self.params['tag'], "evaluation_reward.csv")
        with open(reward_file_path, 'w') as f:
            csv_writer = csv.writer(f)
            avg_reward = sum([i for _, i in self.all_rewards]) / len(self.all_rewards)
            csv_writer.writerow(['average reward', avg_reward])
            for r in self.all_rewards:
                csv_writer.writerow([r[0], r[1]])
        logging.info("saved all evaluation reward to {}".format(reward_file_path))
        logging.info("found {} saved best models".format(len(self.all_rewards)))


def main(test=False):
    trial = EvalTrial(test=test)
    trial.evaluate()
    trial.save_rewards()


if __name__ == "__main__":
    main()
