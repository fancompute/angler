import unittest
import numpy as np

from fdfdpy import Simulation

class Test_Simulation(unittest.TestCase):
    """ Tests the simulation object for various functionalities """

    # this function gets run automatically at beginning of testing.
    def setUp(self):

        # the 'good' inputs
        Nx = 100
        Ny = 50
        self.omega = 100
        self.eps_r = np.ones((Nx, Ny))
        self.dl = 0.001
        self.NPML = [10, 10]
        self.pol = 'Hz'
        self.L0 = 1e-4

    """ all of the functions below get run by the unittest module as long as the 
    function names start with 'test' """


    """ These functions ensuring that an error is thrown
    when passing certain arguments to Simulation """

    def test_freq(self):

        # negative frequency
        with self.assertRaises(ValueError):
            Simulation(-self.omega, self.eps_r, self.dl, self.NPML, self.pol)

        # list of frequencies
        with self.assertRaises(ValueError):
            Simulation([100, 200, 300], self.eps_r, self.dl, self.NPML, self.pol)

    def test_eps(self):

        # negative epsilon
        with self.assertRaises(ValueError):
            Simulation(self.omega, -self.eps_r, self.dl, self.NPML, self.pol)

        # list epsilon instead of numpy array
        with self.assertRaises(ValueError):
            Simulation(self.omega, list(self.eps_r), self.dl, self.NPML, self.pol)

    def test_dl(self):

        # negative dl
        with self.assertRaises(ValueError):
            Simulation(self.omega, self.eps_r, -self.dl, self.NPML, self.pol)

        # list of dl
        with self.assertRaises(ValueError):
            Simulation(self.omega, self.eps_r, [1e-4, 1e-5], self.NPML, self.pol)

    def test_NPML(self):

        # NPML a number
        with self.assertRaises(ValueError):
            Simulation(self.omega, self.eps_r, self.dl, 10, self.pol)

        # NPML too many elements
        with self.assertRaises(ValueError):
            Simulation(self.omega, self.eps_r, self.dl, [10, 10, 10], self.pol)

        # NPML larger than domain
        with self.assertRaises(ValueError):
            Simulation(self.omega, self.eps_r, self.dl, [200, 200], self.pol)

    def test_pol(self):

        # polarization not a string
        with self.assertRaises(ValueError):
            Simulation(self.omega, self.eps_r, self.dl, self.NPML, 5)

        # polarization not the right string
        with self.assertRaises(ValueError):
            Simulation(self.omega, self.eps_r, self.dl, self.NPML, 'WrongPolarization')


if __name__ == '__main__':
    main()
