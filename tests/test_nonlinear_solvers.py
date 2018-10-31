import unittest

import numpy as np
import matplotlib.pylab as plt
from numpy.testing import assert_allclose

from fdfdpy import Simulation


class Test_NLSolve(unittest.TestCase):

    def test_born_newton(self):
        """Tests whether born and newton methods get the same result"""

        n0 = 3.4
        omega = 2*np.pi*200e12
        dl = 0.01
        chi3 = 2.8E-18

        width = 1
        L = 5
        L_chi3 = 4

        width_voxels = int(width/dl)
        L_chi3_voxels = int(L_chi3/dl)

        Nx = int(L/dl)
        Ny = int(3.5*width/dl)

        eps_r = np.ones((Nx, Ny))
        eps_r[:, int(Ny/2-width_voxels/2):int(Ny/2+width_voxels/2)] = np.square(n0)

        nl_region = np.zeros(eps_r.shape)
        nl_region[int(Nx/2-L_chi3_voxels/2):int(Nx/2+L_chi3_voxels/2), int(Ny/2-width_voxels/2):int(Ny/2+width_voxels/2)] = 1

        simulation = Simulation(omega, eps_r, dl, [15, 15], 'Ez')
        simulation.add_mode(n0, 'x', [17, int(Ny/2)], width_voxels*3)
        simulation.setup_modes()
        simulation.add_nl(chi3, nl_region, eps_scale=True, eps_max=np.max(eps_r))

        srcval_vec = np.logspace(1, 3, 3)
        pwr_vec = np.array([])
        T_vec = np.array([])
        for srcval in srcval_vec:
            simulation.setup_modes()
            simulation.src *= srcval

            # Newton
            simulation.solve_fields_nl(solver_nl='newton')
            E_newton = simulation.fields["Ez"]

            # Born
            simulation.solve_fields_nl(solver_nl='born')
            E_born = simulation.fields["Ez"]

            # More solvers (if any) should be added here with corresponding calls to assert_allclose() below

            assert_allclose(E_newton, E_born, rtol=1e-3)

if __name__ == '__main__':
    unittest.main()
