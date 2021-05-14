import argparse
import logging

import torch
from tqdm import tqdm

from . import utils
from .agents import DQNAgent
from .utils import Experience
from .train import TrainTrial

logging.basicConfig(level=logging.INFO)


class RegressionTrial(TrainTrial):
    def __init__(self, test=True):
        super().__init__()
        args = self.parse_args()
        self.params = self.load_hyperparams(args)
        if self.params['test'] or test:
            self.params['test'] = True
            self.params['max_env_steps'] = 1000
            self.params['eval_every_n_steps'] = 100
            self.params['epsilon_decay_period'] = 250
            self.params['n_eval_episodes'] = 2
            self.params['replay_warmup_steps'] = 50
            self.params['gscore'] = True
        self.setup()

    def parse_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # yapf: disable
        parser.add_argument('--env_name', type=str, default='CartPole-v0',
                            help='Which gym environment to use')
        parser.add_argument('--agent', type=str, default='action_dqn',
                            choices=['dqn', 'random', 'action_dqn'],
                            help='Which agent to use')
        parser.add_argument('--duplicate', '-d', type=int, default=5,
                            help='Number of times to duplicate actions')
        parser.add_argument('--seed', '-s', type=int, default=0,
                            help='Random seed')
        parser.add_argument('--hyperparams', type=str, default='hyperparams/defaults.csv',
                            help='Path to hyperparameters csv file')
        parser.add_argument('--test', default=False, action='store_true',
                            help='Enable test mode for quickly checking configuration works')
        parser.add_argument('--load', type=str, default='results/cartpole_seed0_dup5_best.pytorch',
                            help='Path to the saved model file')
        args = self.parse_common_args(parser)
        # yapf: enable
        return args

    def setup(self):
        super().setup()
        self.target_agent = DQNAgent(self.env.observation_space, self.env.action_space, self.params)
        if not self.params['test']:
            print("loading from " + self.params['load'])
            self.target_agent.q.load(self.params['load'])
    
    def update_agent(self):
        batch_loss = []
        for count in range(self.params['updates_per_env_step']):
            if len(self.agent.replay) < self.params['replay_warmup_steps']:
                batch_loss.append(0)
                continue
                
            # take the first n items from replay buffer
            # single batch
            batch = self.agent.replay.memory[:self.params['batch_size']]
            batch = list(zip(*batch))
            batch = utils.Experience(*batch)
            # multi batch
            batch = self.agent.replay.sample(self.params['batch_size'])
            states = torch.stack(batch.state)

            self.agent.q.train()
            self.agent.optimizer.zero_grad()

            # values for the flipped network
            q_values = torch.as_tensor([])
            for s in states:
                q_val = self.agent._get_q_values_for_state(s)
                q_values = torch.cat([q_values, q_val.squeeze()], axis=0)

            # targets from the normal DQN
            with torch.no_grad():
                q_targets = torch.as_tensor([])
                for s in states:
                    q_tar = self.target_agent._get_q_values_for_state(s)
                    q_targets = torch.cat([q_targets, q_tar.squeeze()], axis=0)

            # update
            loss = torch.nn.functional.smooth_l1_loss(input=q_values, target=q_targets)
            loss.backward()
            self.agent.optimizer.step()

            self.agent.n_training_steps += 1
            self.agent.q_target.soft_copy_from(self.agent.q, self.params['target_copy_tau'])

            temp = loss.detach().item()
            batch_loss.append(temp)
        l = sum(batch_loss) / len(batch_loss)
        return l

    def run(self):
        s, done, t = self.env.reset(), False, 0
        for step in tqdm(range(self.params['max_env_steps'])):
            a = self.target_agent.act(s)
            sp, r, done, _ = self.env.step(a)
            t = t + 1
            terminal = torch.as_tensor(False) if t == self.env.unwrapped._max_episode_steps else done
            self.agent.store(Experience(s, a, r, sp, terminal))
            loss = self.update_agent()
            # print(loss)
            if loss == 0 and step > 10000:
                print("loss has reached 0, hooray")
                exit(0)
            if done:
                s = self.env.reset()
            else:
                s = sp
            utils.every_n_times(self.params['eval_every_n_steps'], step, self.evaluate, step)
        self.teardown()


def main(test=False):
    trial = RegressionTrial(test=False)
    trial.run()


if __name__ == "__main__":
    main()
