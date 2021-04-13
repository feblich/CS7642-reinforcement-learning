################
# DO NOT REMOVE
# Versions
# numpy==1.18.0
################
a = 2
b = 4
import numpy as np
r=np.random.rand(4,5)
import numpy as np
from math import trunc
class TDAgent(object):
    def __init__(self):
        pass

    def solve(self, p, V, rewards):
        # compute expected conventional (MC) return
        eG0 = p*(rewards[0]+rewards[2]+sum(rewards[4:])) + (1-p)*(rewards[1]+sum(rewards[3:]))

        # compute n-step expected returns at t = 0
        G01 = p*(rewards[0] + V[1]) + (1-p)*(rewards[1] + V[2])
        G02 = p*(rewards[0] + rewards[2] + V[3]) + (1-p)*(rewards[1]+rewards[3]+V[3])
        G03 = p*(rewards[0] + rewards[2] + rewards[4] + V[4]) + (1-p)*(rewards[1]+rewards[3] + rewards[4]+V[4])
        G04 = p*(rewards[0] + rewards[2] + rewards[4] +rewards[5] + V[5]) + (1-p)*(rewards[1] + rewards[3] + rewards[4] + rewards[5] + V[5])
        G05 = p*(rewards[0] + rewards[2] + rewards[4] + rewards[5] + rewards[6] + V[6]) + (1-p)*(rewards[1] + rewards[3] + rewards[4] + rewards[5] + rewards[6] + V[6])

        # construct (lambda return - eG0 = 0) at t = 0 and solve with np.roots:
        coeffsOfLambdaReturn = [(G05-G04), (G04-G03), (G03-G02), (G02-G01), (G01-eG0)]
        lambdas = np.roots(coeffsOfLambdaReturn)

        # return only 0 < lambdas < 1
        return trunc((10.0**3)*min(np.real(lambdas[np.logical_and(lambdas >= 0, lambdas < 0.99)]))) / 10.0**3
