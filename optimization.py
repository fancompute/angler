import sys
sys.path.append(".")

from adjoint import adjoint_linear, adjoint_nonlinear

import numpy as np
import copy
import progressbar
import matplotlib.pylab as plt
from scipy.optimize import minimize, fmin_l_bfgs_b
from fdfdpy.constants import *
from autograd import grad
from filter import (eps2rho, rho2eps, get_W, deps_drhob, drhob_drhot,
                    drhot_drho, rho2rhot, drhot_drho, rhot2rhob)


class Optimization():

    def __init__(self, J=None, simulation=None, design_region=None, eps_m=5,
                 R=5, eta=0.5, beta=100,
                 field_start='linear', nl_solver='newton', max_ind_shift=None):

        self._J = J
        self.simulation = simulation
        self.design_region = design_region

        self.eps_m = eps_m
        self.R = R

        (Nx, Ny) = self.simulation.eps_r.shape
        self.W = get_W(Nx, Ny, self.design_region, self.R)
        self.eta = eta
        self.beta = beta

        self.field_start = field_start
        self.nl_solver = nl_solver
        self.max_ind_shift = max_ind_shift

        self.src_amplitudes = []
        self.objfn_list = []

        self.mopt = np.zeros((Nx, Ny))
        self.vopt = np.zeros((Nx, Ny))

        # compute the jacobians of J and store these
        self.dJ = self._autograd_dJ(J)

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, J):
        self._J = J
        self.dJ = self._autograd_dJ(J)

    def _autograd_dJ(self, J):
        """ Uses autograd to automatically compute Jacobians of J with respect to each argument"""

        # note: eventually want to check whether J has eps_nl argument, then switch between linear and nonlinear depending.
        dJ = {}
        dJ['lin'] = grad(J, 0)
        dJ['nl'] = grad(J, 1)
        return dJ

    def compute_J(self, simulation):
        """ Returns the current objective function of a simulation"""

        if simulation.fields['Ez'] is None:
            (_, _, Ez) = simulation.solve_fields()
        else:
            Ez = simulation.fields['Ez']

        if simulation.fields_nl['Ez'] is None:
            (_, _, Ez_nl, _) = simulation.solve_fields_nl()
        else:
            Ez_nl = simulation.fields_nl['Ez']

        return self.J(Ez, Ez_nl)

    def compute_dJ(self, simulation, design_region):
        """ Returns the current grad of a simulation"""

        if simulation.fields['Ez'] is None:
            (_, _, Ez) = simulation.solve_fields()
        else:
            Ez = simulation.fields['Ez']

        if simulation.fields_nl['Ez'] is None:
            (_, _, Ez_nl, _) = simulation.solve_fields_nl()
        else:
            Ez_nl = simulation.fields_nl['Ez']

        return self._grad_linear(Ez, Ez_nl)# + self._grad_nonlinear(Ez, Ez_nl)

    def _grad_linear(self, Ez, Ez_nl):
        """gives the linear field gradient: partial J/ partial * E_lin dE_lin / deps"""

        b_aj = -self.dJ['lin'](Ez, Ez_nl)
        Ez_aj = adjoint_linear(self.simulation, b_aj)

        EPSILON_0_ = EPSILON_0*self.simulation.L0
        omega = self.simulation.omega
        dAdeps = self.design_region*omega**2*EPSILON_0_

        rho = self.simulation.rho
        rho_t = rho2rhot(rho, self.W)
        rho_b = rhot2rhob(rho_t, eta=self.eta, beta=self.beta)
        eps_mat = (self.eps_m - 1)

        filt_mat = drhot_drho(self.W)
        proj_mat = drhob_drhot(rho_t, eta=self.eta, beta=self.beta)

        Ez_vec = np.reshape(Ez, (-1,))
        Ez_proj_vec = eps_mat * proj_mat * filt_mat.dot(Ez_vec)
        Ez_proj = np.reshape(Ez_proj_vec, Ez.shape)

        return 1*np.real(Ez_aj * dAdeps * Ez_proj)


    def _grad_nonlinear(self, Ez, Ez_nl):
        """gives the linear field gradient: partial J/ partial * E_lin dE_lin / deps"""

        b_aj = -self.dJ['nl'](Ez, Ez_nl)
        Ez_aj = adjoint_nonlinear(self.simulation, b_aj)

        EPSILON_0_ = EPSILON_0*self.simulation.L0
        omega = self.simulation.omega
        dAdeps = self.design_region*omega**2*EPSILON_0_
        dAnldeps = dAdeps + self.design_region*omega**2*EPSILON_0_*self.simulation.dnl_deps

        rho = self.simulation.rho
        rho_t = rho2rhot(rho, self.W)
        rho_b = rhot2rhob(rho_t, eta=self.eta, beta=self.beta)
        eps_mat = (self.eps_m - 1)

        filt_mat = drhot_drho(self.W)
        proj_mat = drhob_drhot(rho_t, eta=self.eta, beta=self.beta)

        Ez_vec = np.reshape(Ez, (-1,))
        Ez_proj_vec = eps_mat * proj_mat * filt_mat.dot(Ez_vec)
        Ez_proj = np.reshape(Ez_proj_vec, Ez.shape)

        return 1*np.real(Ez_aj * dAnldeps * eps_mat * Ez_proj)


    def check_deriv(self, Npts=5, d_rho=1e-3):
        """ Returns a list of analytical and numerical derivatives to check grad accuracy"""

        self.simulation.eps_r = self.simulation.eps_r
        self.simulation.solve_fields()
        self.simulation.solve_fields_nl()

        # solve for the linear fields and grad of the linear objective function
        grad_avm = self.compute_dJ(self.simulation, self.design_region)
        J_orig = self.compute_J(self.simulation)

        avm_grads = []
        num_grads = []

        # for a number of points
        for _ in range(Npts):

            # pick a random point within the design region
            x, y = np.where(self.design_region == 1)
            i = np.random.randint(len(x))
            pt = [x[i], y[i]]

            # create a new, perturbed permittivity
            rho_new = copy.deepcopy(self.simulation.rho)
            rho_new[pt[0], pt[1]] += d_rho

            # make a copy of the current simulation
            sim_new = copy.deepcopy(self.simulation)
            eps_new = rho2eps(rho=rho_new, eps_m=self.eps_m, W=self.W,
                              eta=self.eta, beta=self.beta)

            sim_new.rho = rho_new
            sim_new.eps_r = eps_new

            sim_new.solve_fields()
            sim_new.solve_fields_nl()

            # plt.imshow(np.abs(sim_new.fields['Ez']))
            # plt.imshow(eps_new)
            # plt.show()

            # solve for the fields with this new permittivity
            J_new = self.compute_J(sim_new)

            # compute the numerical grad
            grad_num = (J_new - J_orig) / d_rho

            # append both grads to lists
            avm_grads.append(grad_avm[pt[0], pt[1]])
            num_grads.append(grad_num)

        return avm_grads, num_grads


    def _set_design_region(self, x, simulation, design_region):
        """ Inserts a vector x into the design_region of simulation.eps_r"""

        eps_vec = copy.deepcopy(np.ndarray.flatten(simulation.eps_r))
        des_vec = np.ndarray.flatten(design_region)

        # Only update the permittivity if it actually differs from the current one 
        # If it doesn't, we don't want to erase the stored fields

        if np.linalg.norm(x - eps_vec[des_vec == 1])/np.linalg.norm(x) > 1e-10:
            eps_vec[des_vec == 1] = x
            eps_new = np.reshape(eps_vec, simulation.eps_r.shape)
            simulation.eps_r = eps_new

    def _get_design_region(self, spatial_array, design_region):
        """ Returns a vector of the elements of spatial_array that are in design_region"""

        spatial_vec = copy.deepcopy(np.ndarray.flatten(spatial_array))
        des_vec = np.ndarray.flatten(design_region)
        x = spatial_vec[des_vec == 1]
        return x

    def _make_progressbar(self, N):
        """ Returns a progressbar to use during optimization"""

        if self.max_ind_shift is not None:

            bar = progressbar.ProgressBar(widgets=[
                ' ', progressbar.DynamicMessage('ObjectiveFn'),
                ' ', progressbar.DynamicMessage('ObjectiveFn_Normalized'),
                ' Iteration: ',
                ' ', progressbar.Counter(), '/%d' % N,
                ' ', progressbar.AdaptiveETA(),
            ], max_value=N)

        else:

            bar = progressbar.ProgressBar(widgets=[
                ' ', progressbar.DynamicMessage('ObjectiveFn'),
                ' Iteration: ',
                ' ', progressbar.Counter(), '/%d' % N,
                ' ', progressbar.AdaptiveETA(),
            ], max_value=N)

        return bar

    def _update_progressbar(self, pbar, iteration, J):

        if self.max_ind_shift is not None:
            objfn_norm = J/np.max(np.square(np.abs(self.simulation.src)))
            pbar.update(iteration, ObjectiveFn=J, ObjectiveFn_Normalized=objfn_norm)
        else:
            pbar.update(iteration, ObjectiveFn=J)

    def run(self, method='LBFGS', Nsteps=100, step_size=0.1,
            beta1=0.9, beta2=0.999, eta=0.5, beta=100, R=5, verbose=True):
        """ Runs an optimization."""

        self.Nsteps = Nsteps
        self.verbose = verbose

        # get the material density from the simulation if only the first time being run
        if self.simulation.rho is None:
            eps = copy.deepcopy(self.simulation.eps_r)
            self.simulation.rho = eps2rho(eps)

        allowed = ['LBFGS', 'GD', 'ADAM']

        if method.lower() in ['lbfgs']:
            self._run_LBFGS()

        elif method.lower() == 'gd':
            self._run_GD(step_size=step_size)

        elif method.lower() == 'adam':
            self._run_ADAM(step_size=step_size, beta1=beta1, beta2=beta2)

        else:
            raise ValueError("'method' must be in {}".format(allowed))

    def _run_GD(self, step_size):
        """ Performs simple grad descent optimization"""

        pbar = self._make_progressbar(self.Nsteps)

        for iteration in range(self.Nsteps):

            J = self.compute_J(self.simulation)
            self.objfn_list.append(J)

            self._update_progressbar(pbar, iteration, J)

            self._set_source_amplitude()

            grad = self.compute_dJ(self.simulation, self.design_region)

            self._update_rho(grad, step_size)

    def _run_ADAM(self, step_size, beta1, beta2):
        """ Performs simple grad descent optimization"""

        pbar = self._make_progressbar(self.Nsteps)

        for iteration in range(self.Nsteps):

            J = self.compute_J(self.simulation)
            self.objfn_list.append(J)

            self._update_progressbar(pbar, iteration, J)

            self._set_source_amplitude()

            grad = self.compute_dJ(self.simulation, self.design_region)

            grad_adam = self._step_adam(grad, iteration, beta1, beta2)

            self._update_rho(grad_adam, step_size)

    def _run_LBFGS(self):
        """Performs L-BFGS Optimization of objective function w.r.t. eps_r"""

        pbar = self._make_progressbar(self.Nsteps)

        def _objfn(x, *argv):
            """ Returns objective function given some permittivity distribution"""

            rho_bar = rho2rho_bar(x)
            eps = rho_bar2eps(rho_bar, self.eps_max)

            self._set_design_region(eps, self.simulation, self.design_region)
            J = self.compute_J(self.simulation)

            # return minus J because we technically will minimize
            return -J

        def _grad(x,  *argv):
            """ Returns full grad given some permittivity distribution"""
            # make a simulation copy

            rho_bar = rho2rho_bar(x)
            eps = rho_bar2eps(rho_bar, self.eps_max)

            self._set_design_region(eps, self.simulation, self.design_region)

            # compute grad, extract design region, turn into vector, return
            grad = self.compute_dJ(self.simulation, self.design_region)
            grad_vec = self._get_design_region(grad, self.design_region)

            return -grad_vec

        # this simple callback function gets run each iteration
        # keeps track of the current iteration step for the progressbar
        # also resets eps on the simulation
        iter_list = [0]

        def _update_iter_count(x_current):
            J = self.compute_J(self.simulation)
            # pbar.update(iter_list[0], ObjectiveFn=J)
            self._update_progressbar(pbar, iter_list[0], J)

            iter_list[0] += 1
            self.objfn_list.append(J)
            self._set_design_region(x_current, self.simulation, self.design_region)
            self._set_source_amplitude()

        # set up bounds on epsilon ((1, eps_m), (1, eps_m), ... ) for each grid in design region
        eps_bounds = tuple([(1, self.eps_max) for _ in range(np.sum(self.design_region == 1))])
        rho_bounds = tuple([(0, 1) for _ in range(np.sum(self.design_region == 1))])

        # start eps off with the one currently within design region
        eps0 = self._get_design_region(self.simulation.eps_r, self.design_region)
        rho_bar0 = eps2rho_bar(eps0, eps_m)
        eps0 = rho_bar2eps(rho_bar, self.eps_max)

        # minimize
        (x, _, _) = fmin_l_bfgs_b(_objfn, x0, fprime=_grad, args=(), approx_grad=0,
                            bounds=eps_bounds, m=10, factr=100,
                            pgtol=1e-08, epsilon=1e-08, iprint=-1,
                            maxfun=15000, maxiter=self.Nsteps, disp=self.verbose,
                            callback=_update_iter_count, maxls=20)

        # finally, set the simulation permittivity to that found via optimization
        self._set_design_region(x, self.simulation, self.design_region)

    def _set_source_amplitude(self, epsilon=1e-2, N=1):
        """ If max_index_shift specified, sets the self.simulation.src amplitude
            low enough so that this is satisfied.
            'epsilon' is the amount to subtract from source to get it under.
        """

        # keep a running list of the source amplitudes
        self.src_amplitudes.append(np.max(np.abs(self.simulation.src)))

        # if a max index shift is specified
        if self.max_ind_shift is not None:

            # for a number of iterations
            for _ in range(N):

                # compute the index shift and update the source according to the ratio
                dn = self.simulation.compute_index_shift()
                max_dn = np.max(dn)
                ratio = self.max_ind_shift / max_dn

                self.simulation.src = self.simulation.src * (np.sqrt(ratio) - epsilon)

    def _update_permittivity(self, grad, step_size):
        """ Manually updates the permittivity with the grad info """

        # deep copy original permittivity (deep for safety)
        eps_old = copy.deepcopy(self.simulation.eps_r)

        # update the old eps to get a new eps with the grad
        eps_new = eps_old + self.design_region * step_size * grad

        # push back inside bounds
        eps_new[eps_new < 1] = 1
        eps_new[eps_new > self.eps_max] = self.eps_max

        # reset the epsilon of the simulation
        self.simulation.eps_r = eps_new

    def _update_rho(self, grad, step_size):
        """ Manually updates the permittivity with the grad info """

        self.simulation.rho = self.simulation.rho + self.design_region * step_size * grad
        self.simulation.rho[self.simulation.rho < 0] = 0
        self.simulation.rho[self.simulation.rho > 1] = 1

        self.simulation.eps_r = rho2eps(self.simulation.rho, self.eps_m, self.W,
                                        eta=self.eta, beta=self.beta)


    def _step_adam(self, grad, iteration, beta1, beta2, epsilon=1e-8):
        """ Performs one step of adam optimization"""

        mopt = beta1 * self.mopt + (1 - beta1) * grad
        mopt_t = mopt / (1 - beta1**(iteration + 1))
        vopt = beta2 * self.vopt + (1 - beta2) * (np.square(grad))
        vopt_t = vopt / (1 - beta2**(iteration + 1))
        grad_adam = mopt_t / (np.sqrt(vopt_t) + epsilon)

        self.mopt = mopt
        self.vopt = vopt

        return grad_adam

    def plt_objs(self, norm=None, ax=None):
        """ Plots objective function vs. iteration"""

        iter_range = range(1, len(self.objfn_list) + 1)
        if norm == 'field':
            obj_scaled = [o/a for o, a in zip(self.objfn_list, self.src_amplitudes)]
            ax.set_ylabel('objective function / field')
        elif norm == 'power':
            obj_scaled = [o/a**2 for o, a in zip(self.objfn_list, self.src_amplitudes)]
            ax.set_ylabel('objective function / power')
        else:
            obj_scaled = self.objfn_list
            ax.set_ylabel('objective function')


        ax.plot(iter_range,  obj_scaled)
        ax.set_xlabel('iteration number')
        ax.set_title('optimization results')
        return ax

    def scan_frequency(self, Nf=50, df=1/20):
        """ Scans the objective function vs. frequency """

        # create frequencies (in Hz)
        delta_f = self.simulation.omega*df
        freqs = 1/2/np.pi*np.linspace(self.simulation.omega - delta_f/2,
                                      self.simulation.omega + delta_f/2,  Nf)

        bar = progressbar.ProgressBar(max_value=Nf)

        # loop through frequencies
        objs = []
        for i, f in enumerate(freqs):

            bar.update(i + 1)

            # make a new simulation object
            sim_new = copy.deepcopy(self.simulation)

            # reset the simulation to compute new A (hacky way of doing it)
            sim_new.omega = 2*np.pi*f
            sim_new.eps_r = self.simulation.eps_r

            # # compute the fields
            # (_, _, Ez) = sim_new.solve_fields()
            # (_, _, Ez_nl, _) = sim_new.solve_fields_nl()

            # compute objective function and append to list
            obj_fn = self.compute_J(sim_new)
            objs.append(obj_fn)

        # compute HM
        objs_array = np.array(objs)
        HM = np.max(objs_array)/2
        above_HM = objs_array > HM

        # does a scan up and down from the midpoint and counts number above HM in this peak
        num_above_HM = 0
        for i in range(int(Nf/2), Nf):
            if not above_HM[i]:
                break
            num_above_HM += 1
        for i in range(int(Nf/2)-1, -1, -1):
            if not above_HM[i]:
                break
            num_above_HM += 1

        # compute FWHM (Hz) using the number above HM and the freq difference
        FWHM = num_above_HM*(freqs[1] - freqs[0])

        return freqs, objs, FWHM

    def scan_power(self, probes=None, Ns=50, s_max=10):
        """ Scans the source amplitude and computes the objective function
            probes is a list of functions for computing the power, for example:
            [lambda simulation: simulation.flux_probe('x', [-NPML[0]-int(l/2/dl), ny + int(d/2/dl)], int(H/2/dl))]
        """

        if probes is None:
            raise ValueError("need to specify 'probes' kwarg as a list of functions for computing the power in each port.")
        num_probes = len(probes)

        # create src_amplitudes
        s_list = np.logspace(-6, np.log10(s_max), Ns)        

        bar = progressbar.ProgressBar(max_value=Ns)

        # transmission
        transmissions = [[] for _ in num_probes]

        for i, s in enumerate(s_list):

            bar.update(i + 1)

            # make a new simulation object
            sim_new = copy.deepcopy(self.simulation)

            # reset the simulation to compute new A (hacky way of doing it)
            sim_new.src = s * sim_new.src / np.max(np.abs(sim_new.src))

            # compute the fields
            _ = sim_new.solve_fields_nl()

            # compute power transmission using each probe
            for probe_index, probe in probes:
                W_out = probe(sim_new)
                sim_new.modes[0].compute_normalization(sim_new)                
                transmissions[probe_index][i] = W_out / W_in
        return transmissions

    def plot_transmissions(self, transmissions, legend=None):
        """ Plots the results of the power scan """

        for p in transmissions:
            plt.plot(p)
        plt.xscale('log')
        plt.xlabel('input source amplitude')
        plt.ylabel('power transmission')
        if legend is not None:
            plt.legend(legend)
        plt.show()
