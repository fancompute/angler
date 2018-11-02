import unittest
import numpy as np
from numpy.testing import assert_allclose
import copy
import sys
sys.path.append('..')
from angler import Simulation
from angler.structures import three_port
from angler.optimization import Optimization
import autograd.numpy as npa


class TestGradient(unittest.TestCase):

    def setUp(self):
        # create a simulation to test just like in notebook

        lambda0 = 2e-6              # free space wavelength (m)
        c0 = 3e8                    # speed of light in vacuum (m/s)
        omega = 2*np.pi*c0/lambda0  # angular frequency (2pi/s)
        dl = 1.1e-1                 # grid size (L0)
        NPML = [15, 15]             # number of pml grid points on x and y borders
        pol = 'Ez'                  # polarization (either 'Hz' or 'Ez')
        source_amp = 100             # amplitude of modal source (A/L0^2?)

        # material constants
        n_index = 2.44              # refractive index
        eps_m = n_index**2          # relative permittivity
        max_ind_shift = 5.8e-2      # maximum allowed nonlinear index shift

        # geometric parameters
        L = 4         # length of box (L0)
        H = 4         # height of box (L0)
        w = .2        # width of waveguides (L0)
        d = H/2.44    # distance between waveguides (L0)
        l = 3         # length of waveguide from PML to box (L0)
        spc = 2       # space between box and PML (L0)

        # define permittivity of three port system
        (eps_r, design_region) = three_port(L, H, w, d, dl, l, spc, NPML, eps_start=eps_m)
        (Nx, Ny) = eps_r.shape
        nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points

        # set the modal source and probes
        self.simulation = Simulation(omega, eps_r, dl, NPML, 'Ez')
        self.simulation.add_mode(np.sqrt(eps_m), 'x', [NPML[0]+int(l/2/dl), ny], int(H/2/dl), scale=source_amp)
        self.simulation.setup_modes()
        self.simulation.init_design_region(design_region, eps_m)

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

        # define linear and nonlinear parts of objective function + the total objective function form
        J = lambda e, e_nl: npa.sum(npa.square(npa.abs(e))*J_top) + npa.sum(npa.square(npa.abs(e_nl))*J_bot)
        import autograd.numpy as npa
        def J(e, e_nl):
            linear_top    =  1*npa.sum(npa.square(npa.abs(e))*J_top)
            linear_bot    = -1*npa.sum(npa.square(npa.abs(e))*J_bot)
            nonlinear_top = -1*npa.sum(npa.square(npa.abs(e_nl))*J_top)
            nonlinear_bot =  1*npa.sum(npa.square(npa.abs(e_nl))*J_bot)
            objfn = linear_top + nonlinear_top + nonlinear_bot + linear_top
            return objfn

        self.design_region = design_region
        self.optimization = Optimization(J=J, simulation=self.simulation, design_region=self.design_region, eps_m=eps_m)

    def test_linear_gradient(self):

        avm_grads, num_grads = self.optimization.check_deriv(Npts=5, d_rho=1e-6)

        avm_grads = np.array(avm_grads)
        num_grads = np.array(num_grads)

        print('linear regime: \n\tanalytical: {}\n\tnumerical:  {}'.format(avm_grads, num_grads))

        assert_allclose(avm_grads, num_grads, rtol=1e-03, atol=.1)

    def test_nonlinear_gradient(self):

        chi3 = 4.1*1e-19            # Al2S3 from Boyd (m^2/V^2)

        # add nonlinearity
        nl_region = copy.deepcopy(self.design_region)
        self.simulation.nonlinearity = [] 
        self.simulation.add_nl(chi3, nl_region, eps_scale=True, eps_max=self.optimization.eps_m)

        avm_grads, num_grads = self.optimization.check_deriv(Npts=5, d_rho=1e-6)

        avm_grads = np.array(avm_grads)
        num_grads = np.array(num_grads)

        print('nonlinear regime: \n\tanalytical: {}\n\tnumerical:  {}'.format(avm_grads, num_grads))

        assert_allclose(avm_grads, num_grads, rtol=1e-03, atol=.1)

if __name__ == '__main__':
    unittest.main()
