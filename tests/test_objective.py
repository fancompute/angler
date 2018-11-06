import unittest
import numpy as np

import sys
sys.path.append('..')

from angler.objective import Objective, obj_arg
from angler.gradients import *

class TestGradient(unittest.TestCase):

    """ Tests the flexible objective function specifier """

    def setUp(self):

        # define the arguments to the objective function
        self.f1 = obj_arg('Ez_lin', component='Ez', nl=False)
        self.f2 = obj_arg('Hz_lin', component='Hz', nl=False)
        self.f3 = obj_arg('Ez_lin2', component='Ez', nl=False)
        self.f3_nl = obj_arg('Ez_nl', component='Ez', nl=True)

        arg_list_lin = [self.f1, self.f2, self.f3]
        arg_list_nl = [self.f1, self.f2, self.f3_nl]

        # define the objective function
        import autograd.numpy as npa
        def J(f1, f2, f3):

            A = npa.sum(npa.abs(f1*f2))
            B = npa.sum(npa.abs(f2*f3))
            C = npa.abs(A - B)
            return -C

        self.objective_lin = Objective(J, arg_list_lin)
        self.objective_nl = Objective(J, arg_list_nl)

        Nx = 40
        Ny = 4
        self.F1 = np.random.random((Nx, Ny))
        self.F2 = np.ones((Nx, Ny))
        self.F3 = np.random.random((Nx, Ny))

    def test_J(self):
        # test to see of the objective function returns something
        
        self.assertLess(self.objective_lin.J(self.F1, self.F2, self.F3), 0)

    def test_is_linear(self):
        # test to see if it correctly classified the objectives

        self.assertTrue(self.objective_lin.is_linear())
        self.assertFalse(self.objective_nl.is_linear())

    def test_autograd_J(self):
        # tests whether the partials were computed

        for dj in self.objective_lin.dJ_list:
            self.assertIsNot(dj, None)
        for dj in self.objective_nl.dJ_list:
            self.assertIsNot(dj, None)

    def test_gradient_load(self):
        # test to see of the the gradients are loaded in correctly

        for grad_fn in self.objective_lin.grad_fn_list:
            self.assertIsNot(grad_fn, None)
        for grad_fn in self.objective_nl.grad_fn_list:
            self.assertIsNot(grad_fn, None)

    def test_gradient_selector(self):
        # test whether correct gradients were selected

        self.assertEqual(self.f1.gradient, grad_linear_Ez)
        self.assertEqual(self.f2.gradient, grad_linear_Ez)
        self.assertEqual(self.f3.gradient, grad_linear_Ez)
        self.assertEqual(self.f3_nl.gradient, grad_kerr_Ez)

if __name__ == '__main__':
    unittest.main()



