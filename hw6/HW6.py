import cvxpy as cp
import numpy as np

class RPSAgent(object):
    def __init__(self):
        pass

    def solve(self, R):
        m = 5
        n = 4
        R = np.array(R)
        R = R * -1
        R = np.append(R, np.ones(3).reshape(1, 3), axis=0)
        R = np.append(R, -1 * np.ones(3).reshape(1, 3), axis=0)
        R = np.append(np.ones(5).reshape(5, 1), R, 1)
        R[3, 0] = 0
        R[4, 0] = 0
        b = np.array([0, 0, 0, 1, -1])
        c = np.array([1, 0, 0, 0])

        x = cp.Variable(n)
        prob = cp.Problem(cp.Maximize(c.T @ x),
                          [R @ x <= b])
        prob.solve()

        return x.value[1:]


import unittest


class TestRPS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = RPSAgent()

    def test_case_1(self):
        R = [
            [0, 1, -1], [-1, 0, 1], [1, -1, 0]
        ]

        np.testing.assert_almost_equal(
            self.agent.solve(R),
            np.array([0.333, 0.333, 0.333]),
            decimal=3
        )

    def test_case_2(self):
        R = [[0, 2, -1],
             [-2, 0, 1],
             [1, -1, 0]]

        np.testing.assert_almost_equal(
            self.agent.solve(R),
            np.array([0.250, 0.250, 0.500]),
            decimal=3
        )


unittest.main(argv=[''], verbosity=2, exit=False)