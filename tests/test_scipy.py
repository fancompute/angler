import unittest
import numpy as np
from numpy.testing import assert_allclose

import sys
sys.path.append("..")

from fdfdpy import Simulation
from nonlinear_avm.structures import two_port
from nonlinear_avm.optimization import Optimization
from nonlinear_avm.adjoint import dJdeps_linear, dJdeps_nonlinear
import copy

class TestScipy(unittest.TestCase):

    def setUp(self):
        # create a simulation to test just like in notebook

        # fundamental constants and simulation parameters
        lambda0 = 2e-6              # free space wavelength (m)
        c0 = 3e8                    # speed of light in vacuum (m/s)
        omega = 2*np.pi*c0/lambda0  # angular frequency (2pi/s)
        dl = 1e-1                   # grid size (L0)
        NPML = [15, 15]             # number of pml grid points on x and y borders
        pol = 'Ez'                  # polarization (either 'Hz' or 'Ez')
        source_amp = 20              # amplitude of modal source (A/L0^2?)

        # material constants
        n_index = 2.44              # refractive index
        eps_m = n_index**2          # relative permittivity
        chi3 = 4.1*1e-19            # Al2S3 from Boyd (m^2/V^2)
        max_ind_shift = 7e-3      # maximum allowed nonlinear refractive index shift (computed from damage threshold)

        # geometric parameters
        L = 3         # length of design region (L0)
        H = 2         # height of design region (L0)
        w = 0.5        # width of waveguides (L0)
        l = 4         # length of waveguide from PML to design region (L0)
        spc = 2     # space between box and PML (L0)

        # define permittivity of three port system
        eps_r, design_region = two_port(L, H, w, l, spc, dl, NPML, eps_start=eps_m)
        (Nx, Ny) = eps_r.shape
        nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points

        # set the modal source and probes
        simulation = Simulation(omega, eps_r, dl, NPML, 'Ez')
        simulation.add_mode(np.sqrt(eps_m), 'x', [NPML[0]+int(l/2/dl), ny], int(Ny/2), scale=source_amp)
        simulation.setup_modes()

        # out modal profile
        sim_out = Simulation(omega, eps_r, dl, NPML, 'Ez')
        sim_out.add_mode(np.sqrt(eps_m), 'x', [-NPML[0]-int(l/2/dl), ny], int(Ny/2))
        sim_out.setup_modes()
        J_out = np.abs(sim_out.src)

        # in modal profile
        sim_in = Simulation(omega, eps_r, dl, NPML, 'Ez')
        sim_in.add_mode(np.sqrt(eps_m), 'x', [NPML[0]+int(l*2/3/dl), ny], int(Ny/2))
        sim_in.setup_modes()
        J_in = np.abs(sim_in.src)

        chi3 = 2.8*1e-18                           # Silcion in m^2/V^2 from Boyd's book
        nl_region = copy.deepcopy(design_region)

        simulation.nonlinearity = []  # This is needed in case you re-run this cell, for example (or you can re-initialize simulation every time)
        simulation.add_nl(chi3, nl_region, eps_scale=True, eps_max=eps_m)

        # define linear and nonlinear parts of objective function + the total objective function form
        J = {}
        J['linear']    = lambda e, eps_r:  np.sum(np.square(np.abs(e))*J_out)
        J['nonlinear'] = lambda e, eps_r: -np.sum(np.square(np.abs(e))*J_out)
        J['total']     = lambda J_lin, J_nonlin: J_lin + J_nonlin

        # define linear and nonlinear parts of dJdE + the total derivative form
        dJ = {}
        dJ['dE_linear']    = lambda e, eps_r:  np.conj(e)*J_out
        dJ['dE_nonlinear'] = lambda e, eps_r: -np.conj(e)*J_out
        dJ['total']        = lambda J_lin, J_nonlin, dJ_lin, dJ_nonlin: dJ_lin + dJ_nonlin

        # optimization parameters
        Nsteps =  5
        step_size = 1e-3
        solver = 'newton'
        opt_method = 'adam'

        # initialize an optimization object with the above parameters and objective function information
        simulation.eps_r = eps_r
        optimization = Optimization(Nsteps=Nsteps, J=J, dJ=dJ, eps_max=eps_m, step_size=step_size,
                                    solver=solver, opt_method=opt_method, max_ind_shift=max_ind_shift)

        self.simulation = simulation
        self.optimization = optimization
        self.design_region = design_region

    def test_scipy(self):

        self.optimization.run_scipy(self.simulation, self.design_region)


if __name__ == '__main__':
    unittest.main()


