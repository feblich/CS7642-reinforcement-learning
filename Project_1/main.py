import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random

# In both experiment 1 & 2, there are 100 training sets, each containing 10
# sequences where each sequence is a bounded random walk with an arbitrary
# number of steps until either state A or G reached

def getTrueValues(A, b):
    return np.linalg.inv(A).dot(b)

def getSingleRandomWalk(states, startingState):

    startIdx = states.index(startingState)
    sequence = [states[startIdx]]
    leftTerminal = states[0]
    rightTerminal = states[-1]
    state = states[startIdx]
    while state not in [leftTerminal, rightTerminal]:
        stateIdx = states.index(state)
        newIdx = stateIdx + random.choice([-1, 1])
        sequence.append(states[newIdx])
        state = states[newIdx]
    return sequence


def getStateVector(states, state):
    stateVec = np.zeros((len(states) - 2, 1))
    idx = states.index(state) - 1
    stateVec[idx] = 1
    return stateVec


def getTrainingSets(nTrainingSets, nSeqs, states):
    trainingSets = []
    statesMatrix = []
    returns = []
    for i in range(nTrainingSets):
        thisTrainingSet = []
        thisStatesMatrixSet = []
        thisReturnSet = []
        for j in range(nSeqs):
            thisSeq = getSingleRandomWalk(states, "D")
            thisTrainingSet.append(thisSeq)
            if thisSeq[-1] == "A":
                thisReturnSet.append(0)
            else:
                thisReturnSet.append(1)
            # create statesMatrix
            thisStatesMatrix = getStateVector(states, thisSeq[0])
            for i in range(1, len(thisSeq)):
                if thisSeq[i] not in [states[0], states[-1]]:
                    thisStatesMatrix = np.concatenate((thisStatesMatrix, getStateVector(states, thisSeq[i])), axis=1)
            thisStatesMatrixSet.append(thisStatesMatrix)

        trainingSets.append(thisTrainingSet)
        statesMatrix.append(thisStatesMatrixSet)
        returns.append(thisReturnSet)

    return trainingSets, statesMatrix, returns


def weightUpdate(stateSequence, returnVal, alpha, lbd, dw, w):
    for k in range(stateSequence.shape[1]):
        e = sum([stateSequence[:, [l - 1]] * lbd ** (k + 1 - l) for l in range(1, k + 2)])
        if k == stateSequence.shape[1] - 1:
            dw += alpha * (returnVal - np.dot(w.T, stateSequence[:, k])) * e
        else:
            dw += alpha * (np.dot(w.T, stateSequence[:, k + 1]) - np.dot(w.T, stateSequence[:, k])) * e
    return dw



if __name__ == "__main__":

    # First off, compute the true value (which in here equal weights) of each state (B, C, D, E, F)
    # using Bellman equation, then solve it using matrix method A*trueValue = b:
    A = np.array([[1, -.5, 0, 0, 0],
                   [-.5, 1, -.5, 0, 0],
                   [0, -.5, 1, -.5, 0],
                   [0, 0, -.5, 1, -.5],
                   [0, 0, 0, -.5, 1]])
    b = np.array([[0], [0], [0], [0], [.5]])
    trueValues = getTrueValues(A, b)

    lambdas = [0, .1, .3, .5, .7, .9, 1]
    alpha = .01
    nTrainingSets = 100
    nSeqs = 10

    # get sequence dataset
    trainingSets, statesMatrix, returns = getTrainingSets(nTrainingSets, nSeqs, ["A", "B", "C", "D", "E", "F", "G"])

    # fig 3 (experiment 1)
    rmse = defaultdict(int)
    for lbd in lambdas:
        for i in range(nTrainingSets):
            w = np.array([[.5, .5, .5, .5, .5]]).T
            wPrev = w.copy()
            diff = .001
            while diff >= .001:
                dw = 0
                for j in range(nSeqs):
                    thisSeq = statesMatrix[i][j]
                    thisReturn = returns[i][j]
                    dw = weightUpdate(thisSeq, thisReturn, alpha, lbd, dw, w)

                w += dw
                diff = np.linalg.norm(wPrev - w)
                wPrev = w.copy()

            rmse[lbd] += np.sqrt(np.mean((w - trueValues)**2))/nTrainingSets # divided by 100 (nTrainingSets) since we're averaging over all lambdas

    # convert from defaultdict to dict
    rmse = dict(rmse)
    xFig3, yFig3 = zip(*rmse.items())
    plt.figure()
    plt.plot(xFig3, yFig3, marker='o')
    plt.xlabel("lambda")
    plt.ylabel("ERROR")


    # fig 4 (experiment 2, single presentation)
    rmse = defaultdict(int)
    lambdas = [0, .1, .3, .5, .7, .8, .9, 1]
    alphas = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6]
    for lbd in lambdas:
        for i in range(nTrainingSets):
            w = .5*np.ones((5, len(alphas)))
            for j in range(nSeqs):
                thisSeq = statesMatrix[i][j]
                thisReturn = returns[i][j]
                for alpha in alphas:
                    dw = 0
                    idx = alphas.index(alpha)
                    dw = weightUpdate(thisSeq, thisReturn, alpha, lbd, dw, w[:, [idx]])
                    w[:, [idx]] += dw

            rmse[lbd] += np.sqrt(np.mean((w - np.repeat(trueValues, w.shape[1], axis=1)) ** 2, axis=0))/nTrainingSets


    rmse = dict(rmse)

    plt.figure()
    plt.plot(alphas, rmse[0], marker='o')
    plt.plot(alphas, rmse[.3], marker='o')
    plt.plot(alphas, rmse[.8], marker='o')
    plt.plot(alphas, rmse[1], marker='o')
    plt.legend(["lambda=0", "lambda=0.3", "lambda=0.8", "lambda=1"])
    plt.xlabel("alpha")
    plt.ylabel("ERROR")


    # fig 5, seems alpha=.2 has the lowest error
    # get best (lowest rmse) alpha for each lambda
    fig5Dict = defaultdict(int)
    for lbdVals in list(rmse.items()):
        fig5Dict[lbdVals[0]] = min(lbdVals[1])

    fig5Dict = dict(fig5Dict)
    plt.figure()
    plt.plot(list(fig5Dict.keys()), list(fig5Dict.values()), marker='o')
    plt.xlabel("lambda")
    plt.ylabel("ERROR using the best alpha")
    plt.show()
