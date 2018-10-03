import unittest
import numpy as np
from numpy.testing import assert_allclose

import sys
sys.path.append("..")

from fdfdpy import Simulation
from nonlinear_avm.structures import three_port
from nonlinear_avm.optimization_scipy import Optimization_Scipy as Optimization
from nonlinear_avm.adjoint import dJdeps_linear, dJdeps_nonlinear
import copy

class TestScipy(unittest.TestCase):

    def setUp(self):
        # create a simulation to test just like in notebook

        lambda0 = 2e-6              # free space wavelength (m)
        c0 = 3e8                    # speed of light in vacuum (m/s)
        omega = 2*np.pi*c0/lambda0  # angular frequency (2pi/s)
        dl = 1.1e-1                 # grid size (L0)
        NPML = [15, 15]             # number of pml grid points on x and y borders
        pol = 'Ez'                  # polarization (either 'Hz' or 'Ez')
        source_amp = 50             # amplitude of modal source (A/L0^2?)

        # material constants
        n_index = 2.44              # refractive index
        eps_m = n_index**2          # relative permittivity
        chi3 = 4.1*1e-19            # Al2S3 from Boyd (m^2/V^2)
        max_ind_shift = 5.8e-2      # maximum allowed nonlinear index shift

        # geometric parameters
        L = 4         # length of box (L0)
        H = 4         # height of box (L0)
        w = .2        # width of waveguides (L0)
        d = H/2.44    # distance between waveguides (L0)
        l = 3         # length of waveguide from PML to box (L0)
        spc = 2       # space between box and PML (L0)

        # define permittivity of three port system
        eps_r = three_port(L, H, w, d, dl, l, spc, NPML, eps_start=eps_m)
        (Nx, Ny) = eps_r.shape
        nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points

        # set the modal source and probes
        self.simulation = Simulation(omega, eps_r, dl, NPML, 'Ez')
        self.simulation.add_mode(np.sqrt(eps_m), 'x', [NPML[0]+int(l/2/dl), ny], int(H/2/dl), scale=source_amp)
        self.simulation.setup_modes()

        # top modal profile
        top = Simulation(omega, eps_r, dl, NPML, 'Ez')
        top.add_mode(np.sqrt(eps_m), 'x', [-NPML[0]-int(l/2/dl), ny+int(d/2/dl)], int(H/2/dl))
        top.setup_modes()
        J_top = np.abs(top.src)

        # bottom modal profile
        bot = Simulation(omega, eps_r, dl, NPML, 'Ez')
        bot.add_mode(np.sqrt(eps_m), 'x', [-NPML[0]-int(l/2/dl), ny-int(d/2/dl)], int(d/dl))
        bot.setup_modes()
        J_bot = np.abs(bot.src)

        # define the design
        self.design_region = np.array(eps_r > 1).astype(int)
        self.design_region[:nx-int(L/2/dl),:] = 0
        self.design_region[nx+int(L/2/dl):,:] = 0
        self.simulation.eps_r[self.design_region == 1] = eps_m/2 + 1/2

        # add nonlinearity
        nl_region = copy.deepcopy(self.design_region)
        self.simulation.nonlinearity = []  # This is needed in case you re-run this cell, for example (or you can re-initialize simulation every time)
        self.simulation.add_nl(chi3, nl_region, eps_scale=True, eps_max=eps_m)

        # define linear and nonlinear parts of objective function + the total objective function form
        import autograd.numpy as npa
        J = lambda e, e_nl, eps: npa.sum(npa.square(npa.abs(e))*J_top) + npa.sum(npa.square(npa.abs(e_nl))*J_bot)

        self.optimization = Optimization(Nsteps=100, eps_max=5, J=J, field_start='linear', solver='newton')

    def test_scipy(self):

        self.optimization.run_LBGFS(self.simulation, self.design_region)


if __name__ == '__main__':
    unittest.main()


