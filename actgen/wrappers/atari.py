import warnings
from abc import ABC
from collections import deque

import cv2
import gym
import numpy as np
from gym import Wrapper

from .fixed_duration_hack import FixedDurationHack


def get_box_shape_like(other_box, new_shape=None, dtype=None):
    """
        other_box: gym.spaces.Box
        new_shape: Optional[Tuple[int]]
        Gets newgym.Box like the one provided
    """
    # the Box space in gym stores the low and high as the shape of the space,
    # (see: https://github.com/openai/gym/blob/master/gym/spaces/box.py#L43)
    # as a result, we are ensuring that they are all the same and grabbing the unique value

    if new_shape is None:
        new_shape = other_box.shape

    if dtype is None:
        dtype = other_box.dtype

    low = np.unique(other_box.low)
    high = np.unique(other_box.high)
    assert len(low) == 1
    assert len(high) == 1

    return gym.spaces.Box(low=low[0], high=high[0], shape=new_shape, dtype=dtype)


# From https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/wrappers.py
class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


# From https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class DiscountWrapper(Wrapper):
    def __init__(self, env: gym.Env, discount: float):
        assert 0 <= discount <= 1
        super().__init__(env)
        self.discount = discount

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if 'rewards' in info:
            info['undiscounted_reward'] = reward
            reward = 0
            for i, r in info['rewards']:
                reward += (self.discount**i) * r
        else:
            warnings.warn('Expected to find list of rewards in the info dict')

        return observation, reward, done, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((shp[0] * k, ) + shp[1:]), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ModifyImageWrapper(Wrapper, ABC):
    def __init__(self, env, func):
        """
        Wrapper that modifies an image using the provided function. This will also modify the intermediate images
        of the option execution if present.
        """
        super().__init__(env)
        self._func = func

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        return self._func(ob)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)

        if 'frames' in info:
            # don't forget to modify the intermediate frames too!
            info['frames'] = [self._func(frame) for frame in info['frames']]

        return self._func(ob), reward, done, info


class WarpFrame(ModifyImageWrapper):
    def __init__(self, env, shape=(84, 84)):
        """Warp frames to 84x84 as done in the Nature paper and later work.
        Expects inputs to be of shape height x width x num_channels
        """
        super().__init__(env, self._warp)
        self.height, self.width = shape
        self.observation_space = get_box_shape_like(env.observation_space, (
            self.height,
            self.width,
            1,
        ))

    def _warp(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class ScaledFloatFrame(ModifyImageWrapper):
    def __init__(self, env):
        super().__init__(env, self._scale)
        self.observation_space = get_box_shape_like(env.observation_space, dtype=np.float32)

    def _scale(self, frame):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(frame).astype(np.float32) / 255.0


class PyTorchFrame(ModifyImageWrapper):
    """Image shape from (height x width x channels) to (C x H x W)"""
    def __init__(self, env):
        super().__init__(env, self._to_pytorch_format)
        shape = env.observation_space.shape
        self.observation_space = get_box_shape_like(env.observation_space,
                                                    (shape[-1], shape[0], shape[1]))

    def _to_pytorch_format(self, observation):
        return np.rollaxis(np.asarray(observation), 2)


class LazyFrames:
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]


def make_deepmind_atari(
        env_name='MsPacman',
        max_episode_steps=None,
        episode_life=False,
        clip_rewards=True,
        frame_stack=True,
        scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    import gym_minipacman
    if 'MiniPacman' in env_name:
        env = gym.make('{}NoFrameskip-v0'.format(env_name))  # hack to be compatible with minipacman naming
    else:
        env = gym.make('{}NoFrameskip-v4'.format(env_name))
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = FixedDurationHack(env)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    if scale:
        env = ScaledFloatFrame(env)
    return env