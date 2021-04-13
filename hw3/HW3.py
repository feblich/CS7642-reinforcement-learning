# final submission
import gym
import numpy as np
from math import sqrt
from gym.envs import toy_text


class FrozenLakeAgent(object):
    def __init__(self):
        self.whoami = '903465890'

    def amap_to_gym(self, amap='FFGG'):
        """Maps the `amap` string to a gym env"""
        amap = np.asarray(amap, dtype='c')
        side = int(sqrt(amap.shape[0]))
        amap = amap.reshape((side, side))
        return gym.make('FrozenLake-v0', desc=amap).unwrapped

    def solve(self, amap, gamma, alpha, epsilon, n_episodes, seed):
        """Implement the agent"""
        env = self.amap_to_gym(amap)
        np.random.seed(seed)
        env.seed(seed)

        Qsa = np.zeros((len(amap), 4))
        for i_episode in range(n_episodes):
            observation = env.reset()
            p = np.random.random()
            if p >= epsilon:
                action = np.argmax(Qsa[observation])
            else:
                action = np.random.randint(4)
            done = False
            while not done:
                prev_obs = observation
                prev_action = action
                observation, reward, done, info = env.step(prev_action)
                p = np.random.random()
                if p >= epsilon:
                    action = np.argmax(Qsa[observation])
                else:
                    action = np.random.randint(4)

                # do the update
                if done:
                    Qsa[prev_obs, prev_action] = Qsa[prev_obs, prev_action] + alpha * (
                                reward - Qsa[prev_obs, prev_action])
                    break
                else:
                    Qsa[prev_obs, prev_action] = Qsa[prev_obs, prev_action] + alpha * (
                                reward + gamma * Qsa[observation, action] - Qsa[prev_obs, prev_action])

        # compute policy
        # first create a dict between action and directions
        action_dict = {0: '<', 1: 'v', 2: '>', 3: '^'}
        policy = []
        for state in Qsa:
            policy.append(action_dict[np.argmax(state)])

        policy = ''.join(policy)

        return policy





# import gym
# import numpy as np
# from math import sqrt
#
# def amap_to_gym(amap='FFGG'):
#     """Maps the `amap` string to a gym env"""
#     amap = np.asarray(amap, dtype='c')
#     side = int(sqrt(amap.shape[0]))
#     amap = amap.reshape((side, side))
#     return gym.make('FrozenLake-v0', desc=amap).unwrapped
#
#
#
# amap='SFFFHFFFFFFFFFFG'
# gamma=1.0
# alpha=0.25
# epsilon=0.29
# n_episodes=14697
# seed=741684
#
# env = amap_to_gym(amap)
#
# # for i_episode in range(2):
# #     observation = env.reset()
# #     for t in range(100):
# #         env.render()
# #         print(observation)
# #         action = env.action_space.sample()
# #         observation, reward, done, info = env.step(action)
# #         if done:
# #             print("Episode finished after {} timesteps".format(t+1))
# #             break
# # env.close()
#
# Qsa = np.zeros((len(amap), 4))
# # holes = [i for i,x in enumerate(list(amap)) if x == 'H']
# for i_episode in range(n_episodes):
#     observation = env.reset()
#     p = np.random.random()
#     if p >= epsilon:
#         action = np.argmax(Qsa[observation])
#     else:
#         action = np.random.randint(4)
#     done = False
#     while not done:
#         prev_obs = observation
#         prev_action = action
#         observation, reward, done, info = env.step(prev_action)
#         p = np.random.random()
#         if p >= epsilon:
#             action = np.argmax(Qsa[observation])
#         else:
#             action = np.random.randint(4)
#
#         # do the update
#         if done:
#             Qsa[prev_obs, prev_action] = Qsa[prev_obs, prev_action] + alpha * (
#                         reward - Qsa[prev_obs, prev_action])
#             break
#         else:
#             Qsa[prev_obs, prev_action] = Qsa[prev_obs, prev_action] + alpha * (
#                         reward + gamma * Qsa[observation, action] - Qsa[prev_obs, prev_action])
#
# # compute policy
# # first create a dict between action and directions
# action_dict = {0: '<', 1: 'v', 2: '>', 3: '^'}
# policy = []
# for state in Qsa:
#     policy.append(action_dict[np.argmax(state)])
#
# policy = ''.join(policy)
#
