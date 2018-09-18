import unittest
import numpy as np

from fdfdpy import Simulation
from structures import three_port
from optimization import Optimization
from adjoint import dJdeps_linear, dJdeps_nonlinear


class TestGradient(unittest.TestCase):

    def setUp(self):
        # create a simulation to test just like in notebook

        lambda0 = 2e-6              # free space wavelength (m)
        c0 = 3e8                    # speed of light in vacuum (m/s)
        omega = 2*np.pi*c0/lambda0  # angular frequency (2pi/s)
        dl = 1.1e-1                 # grid size (L0)
        NPML = [15, 15]             # number of pml grid points on x and y borders
        pol = 'Ez'                  # polarization (either 'Hz' or 'Ez')
        source_amp = 40             # amplitude of modal source (A/L0^2?)

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
        print('Calculate an input power of {} Watts/L0'.format(self.simulation.W_in))

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
        self.J = {}
        self.J['linear']    = lambda e: np.sum(np.square(np.abs(e))*J_top)
        self.J['nonlinear'] = lambda e: np.sum(np.square(np.abs(e))*J_bot)
        self.J['total']     = lambda J_lin, J_nonlin: J_lin + J_nonlin

        # define linear and nonlinear parts of dJdE + the total derivative form
        self.dJdE = {}
        self.dJdE['linear']    = lambda e: np.conj(e)*J_top
        self.dJdE['nonlinear'] = lambda e: np.conj(e)*J_bot
        self.dJdE['total']     = lambda dJdE_lin, dJdE_nonlin: dJdE_lin + dJdE_nonlin

        # define the design and nonlinear regions
        self.design_region = np.array(eps_r > 1).astype(int)
        self.design_region[:nx-int(L/2/dl),:] = 0
        self.design_region[nx+int(L/2/dl):,:] = 0
        self.regions = {}
        self.regions['design'] = self.design_region
        self.regions['nonlin'] = self.design_region

        # define the nonlinearity
        chi3_fdfd = chi3/self.simulation.L0**2          # In the units of the FDFD solver such that eps_nl = eps_r + 3*chi3_fdfd*|e|^2
        kerr_nonlinearity = lambda e: 3*chi3_fdfd*np.square(np.abs(e))
        kerr_nl_de = lambda e: 3*chi3_fdfd*np.conj(e)
        self.nonlin_fns = {}
        self.nonlin_fns['eps_nl'] = kerr_nonlinearity
        self.nonlin_fns['dnl_de'] = kerr_nl_de

        self.optimization = Optimization(Nsteps=10, J=self.J, dJdE=self.dJdE, eps_max=eps_m, step_size=0.001,
                                    solver='newton', opt_method='adam', max_ind_shift=None)

    def test_linear_gradient(self):

        # solve for the linear fields and gradient of the linear objective function
        (Hx, Hy, Ez) = self.simulation.solve_fields()
        grad_lin = dJdeps_linear(self.simulation, self.design_region, self.J[
                                 'linear'], self.dJdE['linear'], averaging=False)

        avm_grads, num_grads = self.optimization.check_deriv(self.simulation, self.design_region)

        print(avm_grads)
        print(num_grads)

        import pdb; pdb.set_trace()


if __name__ == '__main__':
    unittest.main()