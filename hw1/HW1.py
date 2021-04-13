# import pandas as pd
# import numpy as np
# a = 4
# def goodSides(mask):
#     """
#     Get the list of values of good sides from input mask.
#
#     Args:
#         mask (list): mask of die, 0 for good side, 1 for bad side, values are 1 indexed.
#
#     Returns:
#         list: list of good side values
#     """
#     return [i+1 for (i, v) in enumerate(mask) if v==0]
#
#
# def getStates(sides, max_rolls):
#     """
#     Get all possible states given "maximum rolls" (not really how many times agent can roll the dice,
#     but for getting the largest value that is possible from rolling that many times).
#
#     Args:
#         sides (list): values of good sides
#         max_rolls (int): maximux rolls that is possible when rolling the max side value continuously
#
#     Returns:
#         list: all possible states in ascending order, appended by the 'E' for end state and 'B' for bankrupt state
#     """
#
#     states = [0] + [x for x in sides]
#     r = 0
#     max_num = (max_rolls) * sides[-1]
#     counts = 0
#     while (True):
#         counts = len(states)
#         for i in range(len(states)):
#             for j in range(len(sides)):
#                 s = states[i] + sides[j]
#                 if s not in states and s <= max_num:
#                     states.append(s)
#         if counts == len(states): break
#     states = sorted(states)
#     states.append('E')
#     states.append('B')
#     return states
#
#
# def getTransitions(states, sides, N, max_rolls):
#     """
#     Generate transition matrices given states, good side and total side number.
#
#     Args:
#         states (list): all possible states
#         sides (list): good side values (rewards)
#         N (int): total number of sides
#         max_rolls (int): see doc for getStatus
#
#     Returns:
#         np.array: transition matrices of size (A, S, S), A is #actions, S is #states
#     """
#     trans0 = pd.DataFrame(0, index=states, columns=states)
#     trans1 = pd.DataFrame(0, index=states, columns=states)
#     n = len(sides)                                     # number of good side
#     b_rate = 1 - n/N                                   # probability of rolling a bad side
#     trans0.iloc[len(sides)+1:, -1] = 1                 # preset all transitions to bankrupt state as true for action 'roll'
#     for i in range(len(states)-2):                     # loop through all numerical states
#         if states[i] <= sides[-1]*(max_rolls-1):       # check if the current row (state) is from less than max_rolls rolls
#             for j in range(len(states)-2):
#                 if states[j] - states[i] in sides:
#                     trans0.iloc[i, j] = 1/N
#             trans0.iloc[i, -1] = b_rate                # set probability of transition to bankrupt state
#     trans1.iloc[:-1, -2] = 1
#     trans1.iloc[-1, -1] = 1
#     return np.stack((trans0.to_numpy(), trans1.to_numpy()), axis=0)
#
#
# def getRewards(states):
#     """
#     Generate reward function.
#
#     Args:
#         states (list): all possible states
#
#     Returns:
#         reward (np.array): reward matrices of size (S, A)
#     """
#     rewards = pd.DataFrame(0, index=states, columns=["roll", "end"])
#     for s in states:
#         if s not in ['B', 'E']:
#             rewards.loc[s, "end"] = s
#     return rewards.to_numpy()
#
# mask = [0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0]
# sides = goodSides(mask)
#
#
# max_rolls = 3
# states = getStates(sides, max_rolls)
#
# trans = getTransitions(states, sides, len(mask), max_rolls)
#
# import numpy as np
# import math
# # is_bad_side = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
# is_bad_side=[1, 1, 1, 0, 0, 0]
# N = len(is_bad_side)
# dieSides = np.arange(1,N+1)
# states = np.arange(0, N**2/2)
# badSides = []
# for i in range(N):
#     if is_bad_side[i] == 1:
#         badSides.append(dieSides[i])
#
# def getTransProb(N, badSides, states):
#     transP = np.zeros((len(states),len(states)))
#     rewards = np.zeros((len(states),len(states)))
#     for i in range(len(states)):
#         for j in range(len(states)):
#             if states[i] != states[j] and i <= j:
#                 if j - i not in badSides:
#                     if j <= N:
#                         transP[i, j] = 1/N
#                         rewards[i, j] = j - i
#                     else:
#                         transP[i, j] = (1/N)**(math.trunc(j/N) + 1)
#                         rewards[i, j] = j - i
#                 else:
#                     transP[i, j] = 1-(len(badSides)/N)
#                     rewards[i, j] = -i
#
#     return transP, rewards
#
#
#
# transP, rewards = getTransProb(N, badSides, states)
#
#
# # value iteration
# values = np.zeros(len(states))
# deltas = np.zeros(len(states))
# theta = 1e-3
#
# deltas = np.zeros(len(states))
# for i in range(len(values)):
#     oldVs = values[i]
#     vs = 0
#     for j in range(len(states)):
#         if i != j and i <= j:
#             vs += transP[i, j]*(rewards[i, j] + values[j])
#     values[i] = vs
#     deltas[i] = max(deltas[i], abs(oldVs - vs))
#



import numpy as np
import math
# is_bad_side = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0]
is_bad_side=[1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
# is_bad_side=[1, 1, 1, 0, 0, 0]
# is_bad_side = [0,0,1,1,1,0,1]
N = len(is_bad_side)
dieSides = np.arange(1,N+1)
states = np.arange(0, N**2/2)
badSides = []
for i in range(N):
    if is_bad_side[i] == 1:
        badSides.append(dieSides[i])

def getTransProb(N, badSides, states):
    transPRoll = np.zeros((len(states)+1,len(states)+1))
    rewardsRoll = np.zeros((len(states)+1,len(states)+1))
    transPQuit = np.zeros((len(states) + 1, len(states) + 1))
    rewardsQuit = np.zeros((len(states) + 1, len(states) + 1))
    for i in range(len(states)+1):
        for j in range(len(states)+1):
            if i != j and i < j and abs(i - j) <= max(dieSides):
                if j - i not in badSides:
                    transPRoll[i, j] = 1/N
                    rewardsRoll[i, j] = j - i
                else:
                    transPRoll[i, -1] = len(badSides)/N
                    rewardsRoll[i, -1] = -i

    rewardsRoll[:, -1] = np.arange(0,len(states)+1)*-1

    transPQuit[:, -1] = np.ones(len(states)+1)
    rewardsQuit[:, -1] = np.arange(0, len(states)+1)

    return transPRoll, rewardsRoll, transPQuit, rewardsQuit

transPRoll, rewardsRoll, transPQuit, rewardsQuit = getTransProb(N, badSides, states)


# value iteration
values = np.zeros(len(states)+1)
deltas = np.zeros(len(states)+1)
theta = 1e-3

delta = 0

for k in range(50):
    for i in range(len(values)):
        oldVs = values[i].copy()
        vsRoll = 0
        vsQuit = 0
        for j in range(len(values)):
            if j != len(values)-1:
                vsRoll += transPRoll[i, j]*(rewardsRoll[i, j] + values[j])
                vsQuit += transPQuit[i, j]*(rewardsQuit[i, j])
            else:
                vsRoll += transPRoll[i, j] * (rewardsRoll[i, j])
                # vsQuit += transPQuit[i, j] * (rewardsQuit[i, j])
                vsQuit = 0

        vs = np.maximum(vsRoll, vsQuit)
        values[i] = vs.copy()
        # values[i] = vsRoll
        delta = np.maximum(delta, abs(oldVs - values[i]))





import numpy as np
import math
class MDPAgent(object):
    def __init__(self):
        pass

    def solve(self, is_bad_side):
        """Implement the agent"""

        N = len(is_bad_side)
        dieSides = np.arange(1,N+1)
        states = np.arange(0, N**2/2)
        badSides = []
        for i in range(N):
            if is_bad_side[i] == 1:
                badSides.append(dieSides[i])

        def getTransProb(N, badSides, states):
            transPRoll = np.zeros((len(states)+1,len(states)+1))
            rewardsRoll = np.zeros((len(states)+1,len(states)+1))
            transPQuit = np.zeros((len(states) + 1, len(states) + 1))
            rewardsQuit = np.zeros((len(states) + 1, len(states) + 1))
            for i in range(len(states)+1):
                for j in range(len(states)+1):
                    if i != j and i < j and abs(i - j) <= max(dieSides):
                        if j - i not in badSides:
                            transPRoll[i, j] = 1/N
                            rewardsRoll[i, j] = j - i
                        else:
                            transPRoll[i, -1] = len(badSides)/N
                            rewardsRoll[i, -1] = -i

            rewardsRoll[:, -1] = np.arange(0,len(states)+1)*-1

            transPQuit[:, -1] = np.ones(len(states)+1)
            rewardsQuit[:, -1] = np.arange(0, len(states)+1)

            return transPRoll, rewardsRoll, transPQuit, rewardsQuit

        transPRoll, rewardsRoll, transPQuit, rewardsQuit = getTransProb(N, badSides, states)


        # value iteration
        values = np.zeros(len(states)+1)
        deltas = np.zeros(len(states)+1)
        theta = 1e-3

        delta = 0

        for k in range(50):
            for i in range(len(values)):
                oldVs = values[i].copy()
                vsRoll = 0
                vsQuit = 0
                for j in range(len(values)):
                    if j != len(values)-1:
                        vsRoll += transPRoll[i, j]*(rewardsRoll[i, j] + values[j])
                        vsQuit += transPQuit[i, j]*(rewardsQuit[i, j])
                    else:
                        vsRoll += transPRoll[i, j] * (rewardsRoll[i, j])
                        # vsQuit += transPQuit[i, j] * (rewardsQuit[i, j])
                        vsQuit = 0

                vs = np.maximum(vsRoll, vsQuit)
                values[i] = vs.copy()
                # values[i] = vsRoll
                delta = np.maximum(delta, abs(oldVs - values[i]))
        return round(values[0],3)
