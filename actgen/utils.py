from collections import namedtuple, defaultdict
import csv
from distutils.util import strtobool
import logging
from pydoc import locate

import gym
import argparse
import torch

from . import wrappers as wrap

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class Trial:
    """
    a base trial class for running experiments
    """
    def __init__(self):
        pass

    def parse_common_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False
        )
        # common args
        parser.add_argument('--env_name', type=str, default='CartPole-v0',
                            help='Which gym environment to use (abbreviate Atari envs: e.g. "MsPacman")')
        parser.add_argument('--agent', type=str, default='dqn',
                            choices=['dqn', 'random', 'action_dqn'],
                            help='Which agent to use')
        parser.add_argument('--duplicate', '-d', type=int, default=5,
                            help='Number of times to duplicate actions')
        parser.add_argument('--action_effect_multiplier', '-e', type=float, default=1,
                            help='the action effect multiplier (use none-one values to enable semi-duplicate actions)')
        parser.add_argument('--random_actions', default=False, action='store_true',
                            help='Make the duplicate actions all random actions')
        parser.add_argument('--seed', '-s', type=int, default=0,
                            help='Random seed')
        parser.add_argument('--hyperparams', type=str, default='hyperparams/defaults.csv',
                            help='Path to hyperparameters csv file')
        parser.add_argument('--test', default=False, action='store_true',
                            help='Enable test mode for quickly checking configuration works')
        parser.add_argument('--results_dir', type=str, default='./results/',
                            help='Path to the result directory to save model files')
        parser.add_argument('--tag', type=str, default='default_exp',
                            help='A tag for the current experiment, used as a subdirectory name for saving models')
        parser.add_argument('--disable_gpu', default=False, action='store_true',
                            help='enforce training on CPU')
        return parser
    
    def check_params_validity(self, params):
        assert 0 <= params['action_effect_multiplier'] <= 1, "action_effect_multiplier must be a float between [0, 1]"
        logging.info('action_effect_multiplier is {}'.format(params['action_effect_multiplier']))
        assert params['duplicate'] >= 1, "number of sets fo duplicate actions can't be less than 1"

    def parse_unknown_args(self, parser):
        args, unknown = parser.parse_known_args()
        other_args = {
            (remove_prefix(key, '--'), val)
            for (key, val) in zip(unknown[::2], unknown[1::2])
        }
        args.other_args = other_args
        return args
    
    def load_hyperparams(self, args):
        params = load_hyperparams(args.hyperparams)
        for arg_name, arg_value in vars(args).items():
            if arg_name == 'hyperparams':
                continue
            params[arg_name] = arg_value
        for arg_name, arg_value in args.other_args:
            update_param(params, arg_name, arg_value)
        return params
    
    def make_gym_env(self):
        if self.params['atari']:
            env = wrap.make_deepmind_atari(
                    self.params['env_name'],
                    max_episode_steps=(
                    self.params['max_episode_steps'] if self.params['max_episode_steps'] > 0 else None
                    ),
                    episode_life=self.params['episode_life'],
                    clip_rewards=self.params['clip_rewards'],
                    frame_stack=self.params['frame_stack'],
                    scale=self.params['scale_pixel_values'])
        else:
            env = gym.make(self.params['env_name'])
            env = wrap.FixedDurationHack(env)
        if self.params['random_actions']:
            logging.info('making all duplicate actions random actions')
            env = wrap.RandomActions(env, self.params['duplicate'])
        else:
            logging.info('making {} sets of exactly same duplicate actions'.format(self.params['duplicate']))
            env = wrap.DuplicateActions(env, self.params['duplicate'], self.params['action_effect_multiplier'])
        if isinstance(env.action_space, gym.spaces.Box):
            env = wrap.DiscreteBox(env)
        env = wrap.TorchInterface(env)
        return env
    
    def determine_device(self):
        self.params['device'] = determine_device(self.params['disable_gpu'])
        logging.info('training on device {}'.format(self.params['device']))

    def teardown(self):
        pass


def load_hyperparams(filepath):
    params = dict()
    with open(filepath, newline='') as file:
        reader = csv.reader(file, delimiter=',', quotechar='|')
        for name, value, dtype in reader:
            if dtype == 'bool':
                params[name] = bool(strtobool(value))
            else:
                params[name] = locate(dtype)(value)
    return params


def save_hyperparams(filepath, params):
    with open(filepath, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for name, value in sorted(params.items()):
            type_str = defaultdict(lambda: None, {
                bool: 'bool',
                int: 'int',
                str: 'str',
                float: 'float',
            })[type(value)] # yapf: disable
            if type_str is not None:
                writer.writerow((name, value, type_str))


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def update_param(params, name, value):
    if name not in params:
        raise KeyError(
            "Parameter '{}' specified, but not found in hyperparams file.".format(name))
    else:
        logging.info("Updating parameter '{}' to {}".format(name, value))
    if type(params[name]) == bool:
        params[name] = bool(strtobool(value))
    else:
        params[name] = type(params[name])(value)


def every_n_times(n, count, callback, *args, final_count=None):
    if (count % n == 0) or (final_count is not None and (count == final_count)):
        callback(*args)


def determine_device(disable_gpu):
    if disable_gpu or not torch.cuda.is_available():
        return torch.device('cpu')
    else:
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda')
