import unittest
import numpy as np
import autograd.numpy as npa
from numpy.random import random

import sys
sys.path.append("..")

from utils import J_bin


class TestUtils(unittest.TestCase):

    """ Tests the util functions"""

    def setUp(self):
        pass

    def test_J_bin(self):

        # create a fake permittivity
        eps_m = 6
        N = 1000
        eps = random((N, N))*(eps_m - 1) + 1.0
        design_region = np.zeros((N, N))
        design_region[N//4:3*N//4, N//4:3*N//4] = 1

        # compute the binarization penalty
        penalty = J_bin(eps, eps_m, design_region)

        # ensure it's normalized
        assert (0 < penalty and penalty < 1)


if __name__ == '__main__':
    unittest.main()