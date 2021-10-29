import gym
import numpy as np

import math

from .. import utils


class DuplicateActions(gym.Wrapper):
    """
    duplicate the original actions in a gym environment
    """
    def __init__(self, env, n_dup, similarity_score=1):
        """
        when creating a duplicate-action env, there is an option to create semi-duplicate actions by
        specifying a similarity_score argument. 
        The resulting space from this wrapper will be Discrete if original env is Discrete, and Box if 
        original env is Box, and all other env types will trigger an UnImplementedError
        If the space needs to discretized, use the DiscreteBox wrapper after this wrapper.
        :param env: an unwrapped gym environment that has discrete actions
        :param n_dup: the number of times to duplicate the original actions
                        e.g. n_dup = 3 in a cartpole environment will result in a total of 6 actions
                            in the order of (L1, L2, L3, R1, R2, R3)
        :param similarity_score: a float between [0, 1], indicating the similarity
                between original actions and duplicated actions.
                0 indicates no similarity at all, and 1 inidcated the same duplicated actions
        """
        super(DuplicateActions, self).__init__(env)
        self.n_dup = n_dup
        self.similarity_score = similarity_score
        # create new action space
        if type(env.action_space) == gym.spaces.Discrete:
            # catch illegal semi-duplicate actions
            if similarity_score < 1:
                raise NotImplementedError('only Box action spaces can have semi-duplicate actions')
            else:
                self.action_space = gym.spaces.Discrete(n_dup * env.action_space.n)
        elif type(env.action_space) == gym.spaces.Box:
            # turn a 1D box into a n_dup-D box
            # e.g. if original is Box([-1,], [1,], shape=(2,)), n_dup = 3
            #       then new is Box([[-1],[-1],[-1]], [[1],[1],[1]], shape=(2,2,2))
            # stats for old box
            shape = env.action_space.shape
            assert len(shape) == 1  # only implemented for 1D Box
            n_dims = shape[0]
            high = np.expand_dims(env.action_space.high, 0)  # np array
            low = np.expand_dims(env.action_space.low, 0)
            # stats for new box
            semi_high = high * similarity_score
            semi_low = low * similarity_score
            new_high = np.concatenate([high] + [semi_high] * (n_dup - 1))
            new_low = np.concatenate([low] + [semi_low] * (n_dup - 1))
            # create new Box
            self.action_space = gym.spaces.Box(new_low, new_high, dtype=env.action_space.dtype)
        else:
            raise NotImplementedError('duplicate actions only implemented for Discrete and Box action spaces')

    def step(self, action):
        """
        :param action: a number, in range (0, n_dup * num_original_actions)
        :return: (next_state, reward, done, info)
        """
        if type(self.action_space) == gym.spaces.Discrete:
            if action not in self.action_space:
                raise RuntimeError("trying to take action not in action space")
            # the corresponding original action of the original env
            original_a = math.floor(action / self.n_dup)
            return self.env.step(original_a)
        elif type(self.action_space) == gym.spaces.Box:
            return self.env.step(action)
    
    def get_action_similarity_scores(self, a):
        """
        return a list of similarity scores of 'a' with all the actions in the action_space
        """
        if type(self.action_space) == gym.spaces.Discrete:
            return utils.get_action_similarity_for_discrete_action_space(a, self.n_dup, self.action_space.n, self.similarity_score)
        elif type(self.action_space) == gym.spaces.Box:
            raise RuntimeError('Box action space should be wrapped in discrete_box wrapper to get action similarity')
        else:
            raise NotImplementedError('duplicate actions only implemented for Discrete and Box action spaces')


def test_duplicate_action_env():
    env = DuplicateActions(gym.make("CartPole-v0"), 2)  # env with 4 actions

    assert 5 not in env.action_space
    assert 1 in env.action_space

    assert env.action_space == gym.spaces.Discrete(4)

    assert env.action_space.n == 4

    assert env.observation_space == gym.make("CartPole-v0").observation_space

    env = DuplicateActions(gym.make('Pendulum-v0'), 2)  # Box action space
    assert env.action_space == gym.spaces.Box(env.action_space.low, env.action_space.high)
    env.reset()
    for i in range(10):
        action = np.array([np.random.rand()])
        env.step(action)


if __name__ == '__main__':
    test_duplicate_action_env()
