import unittest
import numpy as np
import matplotlib.pylab as plt

from angler import Simulation

class Test_Simulation(unittest.TestCase):
    """ Tests the simulation object for various functionalities """

    # this function gets run automatically at beginning of testing.
    def setUp(self):

        # the 'good' inputs
        Nx = 100
        Ny = 50
        self.omega = 2*np.pi*200e12
        self.eps_r = np.ones((Nx, Ny))
        self.dl = 0.01
        self.NPML = [10, 10]
        self.pol = 'Ez'
        self.L0 = 1e-6

    """ all of the functions below get run by the unittest module as long as the 
    function names start with 'test' """


    def test_1D(self):
        Nx = self.eps_r.shape[0]
        eps_1d = np.ones((Nx,))

        S = Simulation(self.omega, eps_1d, self.dl, [10, 0], self.pol)

        S.src = np.zeros(S.eps_r.shape, dtype=np.complex64)
        S.src[Nx//2, 0] = 1j
        (Hx, Hy, Ez) = S.solve_fields()


    """ These functions ensuring that an error is thrown
    when passing certain arguments to Simulation """

    def test_freq(self):

        # negative frequency
        with self.assertRaises(AssertionError):
            Simulation(-self.omega, self.eps_r, self.dl, self.NPML, self.pol)

    def test_NPML(self):

        # NPML a number
        with self.assertRaises(AssertionError):
            Simulation(self.omega, self.eps_r, self.dl, 10, self.pol)

        # NPML too many elements
        with self.assertRaises(AssertionError):
            Simulation(self.omega, self.eps_r, self.dl, [10, 10, 10], self.pol)

    def test_pol(self):

        # polarization not a string
        with self.assertRaises(AssertionError):
            Simulation(self.omega, self.eps_r, self.dl, self.NPML, 5)

        # polarization not the right string
        with self.assertRaises(AssertionError):
            Simulation(self.omega, self.eps_r, self.dl, self.NPML, 'WrongPolarization')


if __name__ == '__main__':
    unittest.main()
