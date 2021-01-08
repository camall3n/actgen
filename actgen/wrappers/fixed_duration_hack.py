import gym


class FixedDurationHack(gym.Wrapper):
    """
    Description:
        Given a TimeLimit-wrapped environment, copy the _max_episode_steps
        variable to the unwrapped environment for easier access, in case
        *this* environment gets wrapped by additional wrappers.
    """
    def __init__(self, env):
        super().__init__(env)
        self.unwrapped._max_episode_steps = env._max_episode_steps
