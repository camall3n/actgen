import argparse
import logging

import gym
from tqdm import tqdm

from .agents import RandomAgent, DQNAgent, save_model, load_model
from . import utils
from .utils import Experience
from . import wrappers as wrap

logging.basicConfig(level=logging.INFO)


class Trial:
    def __init__(self):
        args = self.parse_args()
        self.params = self.load_hyperparams(args)
        self.setup()

    def parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # yapf: disable
        parser.add_argument('--env', type=str, default='CartPole-v0',
                            help='Which gym environment to use')
        parser.add_argument('--duplicate', '-d', type=int, default=5,
                            help='Number of times to duplicate actions')
        parser.add_argument('--seed', '-s', type=int, default=0,
                            help='Random seed')
        parser.add_argument('--hyperparams', type=str, default='hyperparams/defaults.csv',
                            help='Path to hyperparameters csv file')
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
        params['env_name'] = args.env
        params['seed'] = args.seed
        params['duplicate'] = args.duplicate
        for arg_name, arg_value in args.other_args:
            utils.update_param(params, arg_name, arg_value)
        return params

    def setup(self):
        env = gym.make(self.params['env_name'])
        env = wrap.DuplicateActions(env, self.params['duplicate'])
        env = wrap.FixedDurationHack(env)
        env = wrap.TorchInterface(env)
        self.env = env
        self.agent = DQNAgent(self.env.observation_space.n, self.env.action_space.n, seed=self.params['seed'])

    def teardown(self):
        pass

    def pre_episode(self, episode):
        logging.info("Episode {}".format(episode))

    def run_episode(self, episode):
        s, done, t = self.env.reset(), False, 0
        while not done:
            a = self.agent.act(s)
            sp, r, done, _ = self.env.step(a)
            t = t + 1
            terminal = False if t == self.env.unwrapped._max_episode_steps else done
            self.agent.store(Experience(s, a, r, sp, terminal))
            s = sp

    def post_episode(self, episode):
        logging.info('Episode complete')
        loss = []
        for count in tqdm(range(self.params['updates_per_episode'])):
            temp = self.agent.update()
            loss.append(temp)
        loss = sum(loss)/len(loss)

        utils.every_n_times(self.params['eval_period'], episode, self.evaluate, episode)

    def evaluate(self, episode):
        ep_scores = []
        for ep in range(self.params['n_eval_episodes']):
            s, G, done, t = self.env.reset(), 0, False, 0
            while not done:
                a = self.agent.act(s, testing=True)
                sp, r, done, _ = self.env.step(a)
                s, G, t = sp, G + r, t + 1
            ep_scores.append(G.detach())
        avg_episode_score = (sum(ep_scores)/len(ep_scores)).item()
        logging.info("Evaluation: episode {}, average score {}".format(
            episode, avg_episode_score))

    def run(self):
        for episode in range(self.params['max_episode']):
            self.pre_episode(episode)
            self.run_episode(episode)
            self.post_episode(episode)
        self.teardown()


if __name__ == "__main__":
    trial = Trial()
    trial.run()
