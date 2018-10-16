import unittest
import numpy as np
import autograd.numpy as npa
from numpy.random import random
import copy

import sys
sys.path.append("..")
sys.path.append("../fdfdpy")

from structures import three_port, two_port
from fdfdpy import Simulation
from utils import Binarizer
from optimization import Optimization

class TestUtils(unittest.TestCase):

    """ Tests the util functions"""

    def setUp(self):
        # create a simulation to test just like in notebook

        lambda0 = 2e-6              # free space wavelength (m)
        c0 = 3e8                    # speed of light in vacuum (m/s)
        omega = 2*np.pi*c0/lambda0  # angular frequency (2pi/s)
        dl = 0.5e-1                 # grid size (L0)
        NPML = [15, 15]             # number of pml grid points on x and y borders
        pol = 'Ez'                  # polarization (either 'Hz' or 'Ez')
        source_amp = 10             # amplitude of modal source (A/L0^2?)

        # material constants
        n_index = 2.44              # refractive index
        eps_m = n_index**2          # relative permittivity
        self.eps_m = eps_m
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
        eps_r, design_region = three_port(L, H, w, d, l, spc, dl, NPML, eps_start=eps_m)
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
        self.J_top = np.abs(top.src)

        # bottom modal profile
        bot = Simulation(omega, eps_r, dl, NPML, 'Ez')
        bot.add_mode(np.sqrt(eps_m), 'x', [-NPML[0]-int(l/2/dl), ny-int(d/2/dl)], int(H/2/dl))
        bot.setup_modes()
        self.J_bot = np.abs(bot.src)

        # compute straight line simulation
        eps_r_wg, _ = two_port(L, H, w, l, spc, dl, NPML, eps_start=eps_m)
        (Nx_wg, Ny_wg) = eps_r_wg.shape
        nx_wg, ny_wg = int(Nx_wg/2), int(Ny_wg/2)            # halfway grid points     
        simulation_wg = Simulation(omega, eps_r_wg, dl, NPML, 'Ez')
        simulation_wg.add_mode(np.sqrt(eps_m), 'x', [NPML[0]+int(l/2/dl), ny_wg], int(H/2/dl), scale=source_amp)
        simulation_wg.setup_modes()

        # compute normalization
        sim_out = Simulation(omega, eps_r_wg, dl, NPML, 'Ez')
        sim_out.add_mode(np.sqrt(eps_m), 'x', [-NPML[0]-int(l/2/dl), ny], int(H/2/dl))
        sim_out.setup_modes()
        J_out = np.abs(sim_out.src)
        (_, _, Ez_wg) = simulation_wg.solve_fields()
        SCALE = np.sum(np.square(np.abs(Ez_wg))*J_out)

        # define the design region
        self.design_region = design_region
        self.simulation.init_design_region(design_region, eps_m, style='random')

        # add nonlinearity
        nl_region = copy.deepcopy(self.design_region)
        self.simulation.nonlinearity = []  # This is needed in case you re-run this cell, for example (or you can re-initialize simulation every time)
        self.simulation.add_nl(chi3, nl_region, eps_scale=True, eps_max=eps_m)

        # define linear and nonlinear parts of objective function + the total objective function form
        def J(e, e_nl, eps):
            linear_top = npa.sum(npa.square(npa.abs(e))*self.J_top)
            linear_bot = npa.sum(npa.square(npa.abs(e))*self.J_bot)
            nonlinear_top = npa.sum(npa.square(npa.abs(e_nl))*self.J_top)
            nonlinear_bot = npa.sum(npa.square(npa.abs(e_nl))*self.J_bot)
            objfn = linear_top - nonlinear_top + nonlinear_bot - linear_top
            return objfn

        self.optimization = Optimization(J=J, Nsteps=4, eps_max=eps_m, field_start='linear', nl_solver='newton')

    def test_binarize(self):

        binarizer = Binarizer(self.design_region, self.eps_m)

        def J(e, e_nl, eps):
            linear_top = npa.sum(npa.square(npa.abs(e))*self.J_top)
            linear_bot = npa.sum(npa.square(npa.abs(e))*self.J_bot)
            nonlinear_top = npa.sum(npa.square(npa.abs(e_nl))*self.J_top)
            nonlinear_bot = npa.sum(npa.square(npa.abs(e_nl))*self.J_bot)
            objfn = linear_top + nonlinear_bot
            return objfn

        self.simulation.init_design_region(self.design_region, self.eps_m, style='full')

        # Test Binarizer.smoothness

        J_bin = binarizer.smoothness(J)

        @binarizer.smoothness
        def J_bin_decorator(e, e_nl, eps):
            linear_top = npa.sum(npa.square(npa.abs(e))*self.J_top)
            linear_bot = npa.sum(npa.square(npa.abs(e))*self.J_bot)
            nonlinear_top = npa.sum(npa.square(npa.abs(e_nl))*self.J_top)
            nonlinear_bot = npa.sum(npa.square(npa.abs(e_nl))*self.J_bot)
            objfn = linear_top + nonlinear_bot       
            return objfn

        (_, _, Ez) = self.simulation.solve_fields()
        (_, _, Ez_nl, _) = self.simulation.solve_fields_nl()
        eps = self.simulation.eps_r

        J1 = J(Ez, Ez_nl, eps)
        J2 = J_bin(Ez, Ez_nl, eps)
        J3 = J_bin_decorator(Ez, Ez_nl, eps)
        print('for smoothness binarizer with full cells:\n\tJ1 = {}\n\tJ2 = {}\n\tJ3 = {}'.format(J1, J2, J3))


        self.simulation.init_design_region(self.design_region, self.eps_m, style='random')

        # Test Binarizer.smoothness
        print('\n\ntesting smoothness binarizer')

        J_bin = binarizer.smoothness(J)

        @binarizer.smoothness
        def J_bin_decorator(e, e_nl, eps):
            linear_top = npa.sum(npa.square(npa.abs(e))*self.J_top)
            linear_bot = npa.sum(npa.square(npa.abs(e))*self.J_bot)
            nonlinear_top = npa.sum(npa.square(npa.abs(e_nl))*self.J_top)
            nonlinear_bot = npa.sum(npa.square(npa.abs(e_nl))*self.J_bot)
            objfn = linear_top + nonlinear_bot        
            return objfn

        (_, _, Ez) = self.simulation.solve_fields()
        (_, _, Ez_nl, _) = self.simulation.solve_fields_nl()
        eps = self.simulation.eps_r

        J1 = J(Ez, Ez_nl, eps)
        J2 = J_bin(Ez, Ez_nl, eps)
        J3 = J_bin_decorator(Ez, Ez_nl, eps)
        print('for smoothness binarizer with random cells:\n\tJ1 = {}\n\tJ2 = {}\n\tJ3 = {}'.format(J1, J2, J3))

        # Test Binarizer.density
        print('\n\ntesting density binarizer')
        J_bin = binarizer.density(J)

        @binarizer.density
        def J_bin_decorator(e, e_nl, eps):
            linear_top = npa.sum(npa.square(npa.abs(e))*self.J_top)
            linear_bot = npa.sum(npa.square(npa.abs(e))*self.J_bot)
            nonlinear_top = npa.sum(npa.square(npa.abs(e_nl))*self.J_top)
            nonlinear_bot = npa.sum(npa.square(npa.abs(e_nl))*self.J_bot)
            objfn = linear_top + nonlinear_bot      
            return objfn

        (_, _, Ez) = self.simulation.solve_fields()
        (_, _, Ez_nl, _) = self.simulation.solve_fields_nl()
        eps = self.simulation.eps_r

        J1 = J(Ez, Ez_nl, eps)
        J2 = J_bin(Ez, Ez_nl, eps)
        J3 = J_bin_decorator(Ez, Ez_nl, eps)

        print('for density binarizer with random cells:\n\tJ1 = {}\n\tJ2 = {}\n\tJ3 = {}'.format(J1, J2, J3))
        assert J1 > J2
        assert J2 == J3

        self.simulation.init_design_region(self.design_region, self.eps_m, style='empty')

        (_, _, Ez) = self.simulation.solve_fields()
        (_, _, Ez_nl, _) = self.simulation.solve_fields_nl()
        eps = self.simulation.eps_r

        J1 = J(Ez, Ez_nl, eps)
        J2 = J_bin(Ez, Ez_nl, eps)
        J3 = J_bin_decorator(Ez, Ez_nl, eps)

        print('for density binarizer with empty cells:\n\tJ1 = {}\n\tJ2 = {}\n\tJ3 = {}'.format(J1, J2, J3))
        assert J1 == J2
        assert J2 == J3

        self.simulation.init_design_region(self.design_region, self.eps_m, style='halfway')

        (_, _, Ez) = self.simulation.solve_fields()
        (_, _, Ez_nl, _) = self.simulation.solve_fields_nl()
        eps = self.simulation.eps_r

        J1 = J(Ez, Ez_nl, eps)
        J2 = J_bin(Ez, Ez_nl, eps)
        J3 = J_bin_decorator(Ez, Ez_nl, eps)

        print('for density binarizer with halfway cells:\n\tJ1 = {}\n\tJ2 = {}\n\tJ3 = {}'.format(J1, J2, J3))
        assert J2 == 0
        assert J2 == J3

        self.simulation.init_design_region(self.design_region, self.eps_m, style='full')

        (_, _, Ez) = self.simulation.solve_fields()
        (_, _, Ez_nl, _) = self.simulation.solve_fields_nl()
        eps = self.simulation.eps_r

        J1 = J(Ez, Ez_nl, eps)
        J2 = J_bin(Ez, Ez_nl, eps)
        J3 = J_bin_decorator(Ez, Ez_nl, eps)

        print('for density binarizer with full cells:\n\tJ1 = {}\n\tJ2 = {}\n\tJ3 = {}'.format(J1, J2, J3))
        assert J1 == J2
        assert J2 == J3

        # print('for smoothness binarizer:\n\tJ1 = {}\n\tJ2 = {}\n\tJ3 = {}'.format(J1, J2, J3))
        # assert J2 == J3
        # assert J1 > J2


        # Test Binarizer.density

        binarizer = Binarizer(self.design_region, self.eps_m, exp_const=10)

        print('\n\ntesting exponential density binarizer')
        J_bin2 = binarizer.density_exp(J)

        @binarizer.density_exp
        def J_bin2_decorator(e, e_nl, eps):
            linear_top = npa.sum(npa.square(npa.abs(e))*self.J_top)
            linear_bot = npa.sum(npa.square(npa.abs(e))*self.J_bot)
            nonlinear_top = npa.sum(npa.square(npa.abs(e_nl))*self.J_top)
            nonlinear_bot = npa.sum(npa.square(npa.abs(e_nl))*self.J_bot)
            objfn = linear_top + nonlinear_bot      
            return objfn

        self.simulation.init_design_region(self.design_region, self.eps_m, style='random')

        (_, _, Ez) = self.simulation.solve_fields()
        (_, _, Ez_nl, _) = self.simulation.solve_fields_nl()
        eps = self.simulation.eps_r

        J1 = J(Ez, Ez_nl, eps)
        J2 = J_bin2(Ez, Ez_nl, eps)
        J3 = J_bin2_decorator(Ez, Ez_nl, eps)

        print('for exponential density binarizer with random cells:\n\tJ1 = {}\n\tJ2 = {}\n\tJ3 = {}'.format(J1, J2, J3))
        assert J1 > J2
        assert J2 == J3

        self.simulation.init_design_region(self.design_region, self.eps_m, style='empty')

        (_, _, Ez) = self.simulation.solve_fields()
        (_, _, Ez_nl, _) = self.simulation.solve_fields_nl()
        eps = self.simulation.eps_r

        J1 = J(Ez, Ez_nl, eps)
        J2 = J_bin(Ez, Ez_nl, eps)
        J3 = J_bin_decorator(Ez, Ez_nl, eps)

        print('for exponential density binarizer with empty cells:\n\tJ1 = {}\n\tJ2 = {}\n\tJ3 = {}'.format(J1, J2, J3))
        assert J1 == J2
        assert J2 == J3

        self.simulation.init_design_region(self.design_region, self.eps_m, style='halfway')

        (_, _, Ez) = self.simulation.solve_fields()
        (_, _, Ez_nl, _) = self.simulation.solve_fields_nl()
        eps = self.simulation.eps_r

        J1 = J(Ez, Ez_nl, eps)
        J2 = J_bin(Ez, Ez_nl, eps)
        J3 = J_bin_decorator(Ez, Ez_nl, eps)

        print('for exponential density binarizer with halfway cells:\n\tJ1 = {}\n\tJ2 = {}\n\tJ3 = {}'.format(J1, J2, J3))
        assert J2 == 0
        assert J2 == J3

        self.simulation.init_design_region(self.design_region, self.eps_m, style='full')

        (_, _, Ez) = self.simulation.solve_fields()
        (_, _, Ez_nl, _) = self.simulation.solve_fields_nl()
        eps = self.simulation.eps_r

        J1 = J(Ez, Ez_nl, eps)
        J2 = J_bin(Ez, Ez_nl, eps)
        J3 = J_bin_decorator(Ez, Ez_nl, eps)

        print('for exponential density binarizer with full cells:\n\tJ1 = {}\n\tJ2 = {}\n\tJ3 = {}'.format(J1, J2, J3))
        assert J1 == J2
        assert J2 == J3

if __name__ == '__main__':
    unittest.main()
