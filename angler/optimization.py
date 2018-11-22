import numpy as np
import scipy.sparse as sp
import copy
import progressbar
import matplotlib.pylab as plt
from scipy.optimize import minimize, fmin_l_bfgs_b
from autograd import grad

from angler.constants import *
from angler.filter import (eps2rho, rho2eps, get_W, deps_drhob, drhob_drhot,
                    drhot_drho, rho2rhot, drhot_drho, rhot2rhob)


class Optimization():

    def __init__(self, objective, simulation, design_region, eps_m=5,
                 R=None, eta=0.5, beta=1e-9,
                 field_start='linear', nl_solver='newton', max_ind_shift=None):

        # store essential objects
        self.objective = objective
        self.simulation = simulation
        self.design_region = design_region

        # store and compute filter and projection objects
        self.R = R
        (Nx, Ny) = self.simulation.eps_r.shape
        if self.R is not None:
            self.W = get_W(Nx, Ny, self.design_region,  NPML=self.simulation.NPML, R=self.R)
        else:
            self.W = sp.eye(Nx*Ny, dtype=np.complex64)
        self.eta = eta
        self.beta = beta

        # stre other optional parameters
        self.eps_m = eps_m
        self.field_start = field_start
        self.nl_solver = nl_solver
        self.max_ind_shift = max_ind_shift

        # store things that will be used later
        self.src_amplitudes = []
        self.objfn_list = []
        self.fields_current = False     # are the field_args current?  if not, will need to recompute

    def _solve_objfn_arg_fields(self, simulation):
        """ 
            solves for all of the fields needed in the objective function.
            also checks if it's linear (and if so, doesnt solve_nl)
        """

        # do something only if the fields are not current, or there are no stored fields
        if not self.fields_current or not self.field_arg_list:

            _ = simulation.solve_fields()
            if not self.objective.is_linear():
                _ = simulation.solve_fields_nl()

            # prepare a list of arguments that correspond to obj_fn arguments
            field_arg_list = []
            for arg in self.objective.arg_list:
                if not arg.nl:
                    field = simulation.fields[arg.component]
                else:
                    field = simulation.fields_nl[arg.component]
                if field is None:
                    raise ValueError("couldn't find a field defined for component '{}'.  Could be the wrong polarization (simulation is '{}' polarization).".format(arg.component, self.simulation.pol))
                field_arg_list.append(field)

            # store these arguments (so that they dont have to be recomputed)
            self.field_arg_list = field_arg_list

            # lets other functions know the field_arg_list is up to date
            self.fields_current = True


    def compute_J(self, simulation):
        """ Returns the current objective function of a simulation"""

        # stores all of the fields for the objective function in self.field_arg_list
        self._solve_objfn_arg_fields(simulation)

        # pass these arguments to the objective function
        return self.objective.J(*self.field_arg_list)

    def compute_dJ(self, simulation, design_region):
        """ Returns the current grad of a simulation"""

        # stores all of the fields for the objective function in self.field_arg_list
        self._solve_objfn_arg_fields(simulation)

        # sum up gradient contributions from each argument in J
        gradient_sum = 0
        for gradient_fn, dJ, arg in zip(self.objective.grad_fn_list, self.objective.dJ_list, self.objective.arg_list):
            if not arg.nl:
                Fz = simulation.fields[simulation.pol]
            else:
                Fz = simulation.fields_nl[simulation.pol]
            gradient = gradient_fn(self, dJ, Fz, self.field_arg_list)
            gradient_sum += gradient

        return gradient_sum

    def check_deriv(self, Npts=5, d_rho=1e-3):
        """ Returns a list of analytical and numerical derivatives to check grad accuracy"""

        self.fields_current = False
        self.simulation.eps_r = rho2eps(rho=self.simulation.rho, eps_m=self.eps_m, W=self.W,
                                        eta=self.eta, beta=self.beta)

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

            self.fields_current = False

            # solve for the fields with this new permittivity
            J_new = self.compute_J(sim_new)

            # compute the numerical grad
            grad_num = (J_new - J_orig) / d_rho

            # append both grads to lists
            avm_grads.append(grad_avm[pt[0], pt[1]])
            num_grads.append(grad_num)

        return avm_grads, num_grads

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
            beta1=0.9, beta2=0.999, verbose=True, temp_plt=None):
        """ Runs an optimization."""

        self.Nsteps = Nsteps
        self.verbose = verbose
        self.temp_plt = temp_plt

        self._check_temp_plt()

        # get the material density from the simulation if only the first time being run
        if self.simulation.rho is None:
            eps = copy.deepcopy(self.simulation.eps_r)
            self.simulation.rho = eps2rho(eps)
        
        self.fields_current = False
        
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

            if self.temp_plt is not None:
                if np.mod(iteration, self.temp_plt.it_plot) == 0:
                    self.plot_it(iteration)

            self._update_rho(grad, step_size)

    def _run_ADAM(self, step_size, beta1, beta2):
        """ Performs simple grad descent optimization"""
        pbar = self._make_progressbar(self.Nsteps)

        for iteration in range(self.Nsteps):

            J = self.compute_J(self.simulation)
            self.objfn_list.append(J)
            # pbar.update(iteration, ObjectiveFn=J)
            self._update_progressbar(pbar, iteration, J)

            self._set_source_amplitude()

            grad = self.compute_dJ(self.simulation, self.design_region)

            if iteration == 0:
                mopt = np.zeros(grad.shape)
                vopt = np.zeros(grad.shape)

            (grad_adam, mopt, vopt) = self._step_adam(grad, mopt, vopt, iteration, beta1, beta2,)

            if self.temp_plt is not None:
                if np.mod(iteration, self.temp_plt.it_plot) == 0:
                    self.plot_it(iteration)

            self._update_rho(grad_adam, step_size)

    def _run_LBFGS(self):
        """Performs L-BFGS Optimization of objective function w.r.t. eps_r"""

        pbar = self._make_progressbar(self.Nsteps)

        def _objfn(rho, *argv):
            """ Returns objective function given some permittivity distribution"""

            self._set_design_region(rho)
            J = self.compute_J(self.simulation)

            # return minus J because we technically will minimize
            return -J

        def _grad(rho,  *argv):
            """ Returns full grad given some permittivity distribution"""

            self._set_design_region(rho)

            # compute grad, extract design region, turn into vector, return
            grad = self.compute_dJ(self.simulation, self.design_region)
            grad_vec = self._get_design_region(grad)

            return -grad_vec

        # this simple callback function gets run each iteration
        # keeps track of the current iteration step for the progressbar
        # also resets eps on the simulation
        iter_list = [0]

        def _update_iter_count(x_current):
            J = self.compute_J(self.simulation)
            self._update_progressbar(pbar, iter_list[0], J)
            self.objfn_list.append(J)
            self._set_design_region(x_current)
            self._set_source_amplitude()
            if self.temp_plt is not None:
                if np.mod(iter_list[0], self.temp_plt.it_plot) == 0:
                    self.plot_it(iter_list[0])

            iter_list[0] += 1

        N_des = np.sum(self.design_region == 1)              # num points in design region
        rho_bounds = tuple([(0, 1) for _ in range(N_des)])   # bounds on rho {0, 1}

        # start eps off with the one currently within design region
        rho = copy.deepcopy(self.simulation.rho)
        rho0 = self._get_design_region(rho)
        rho0 = np.reshape(rho0, (-1,))

        # minimize
        (rho_final, _, _) = fmin_l_bfgs_b(_objfn, rho0, fprime=_grad, args=(), approx_grad=0,
                            bounds=rho_bounds, m=10, factr=10,
                            pgtol=1e-15, epsilon=1e-08, iprint=-1,
                            maxfun=15000, maxiter=self.Nsteps, disp=self.verbose,
                            callback=_update_iter_count, maxls=20)

        # finally, set the simulation permittivity to that found via optimization
        self._set_design_region(rho_final)

    def plot_it(self, iteration):
        Nplots = len(self.temp_plt.plot_what)
        plt.close('all')

        Ez_lin = self.simulation.fields['Ez']

        # This is what was used for the T-port in the paper    
        vmin = 4 * np.sqrt(self.simulation.W_in)
        vmax = np.abs(Ez_lin).max()/1.5

        # # This is more general but less flexible
        # if self.temp_plt.vlims[0] == None:
        #     vmin = np.abs(Ez_lin).min()
        # else:
        #     vmin = self.temp_plt.vlims[0]
        # if self.temp_plt.vlims[1] == None:
        #     vmax = np.abs(Ez_lin).max()
        # else:
        #     vmax = self.temp_plt.vlims[1]

        if Nplots == 4:
            f, faxs = plt.subplots(2, 2, figsize=self.temp_plt.figsize)
            axs = faxs.ravel()
        else:
            f, axs = plt.subplots(1, Nplots, figsize=self.temp_plt.figsize)

        for n, plots in enumerate(self.temp_plt.plot_what):
            if plots == 'eps':
                ax = axs[n]
                self.simulation.plt_eps(outline=False, cbar=False, ax=ax)
                ax.set_title('Permittivity')
                ax.annotate('Iteration:%4d' % np.int(iteration), (0.2, 0.92),
                    xycoords='axes fraction',
                    size='medium',
                    color='k',
                    horizontalalignment='center',
                    verticalalignment='center')
            if plots == 'of':
                ax = axs[n]
                self.plt_objs(ax=ax)
                ax.set_title('Objective')
            if plots == 'elin':
                ax = axs[n]
                self.simulation.plt_abs(outline=True, cbar=False, ax=ax, logscale=True, vmin=vmin, vmax=vmax)
                ax.set_title('Linear field')
            if plots == 'enl':
                ax = axs[n]
                self.simulation.plt_abs(outline=True, cbar=False, ax=ax, nl=True, logscale=True, vmin=vmin, vmax=vmax)
                ax.set_title('Nonlinear field')
        
        fname = self.temp_plt.folder + ('it%06d.png' % np.int(iteration))
        plt.savefig(fname, dpi=self.temp_plt.dpi)

    def _set_design_region(self, x):
        """ Inserts a vector x into design region of simulation.rho """

        rho_vec = np.reshape(copy.deepcopy(self.simulation.rho), (-1,))
        des_vec = np.reshape(self.design_region, (-1,))

        # Only update the rho if it actually differs from the current one
        # If it doesn't, we don't want to erase the stored fields

        if np.linalg.norm(x - rho_vec[des_vec == 1])/np.linalg.norm(x) > 1e-10:
            rho_vec[des_vec == 1] = x
            rho_new = np.reshape(rho_vec, self.simulation.rho.shape)
            self.simulation.rho = rho_new
            eps_new = rho2eps(rho=rho_new, eps_m=self.eps_m, W=self.W,
                              eta=self.eta, beta=self.beta)
            self.simulation.eps_r = eps_new

        self.fields_current = False

    def _get_design_region(self, spatial_array):
        """ Returns a vector of the elements of spatial_array that are in design_region"""

        spatial_vec = copy.deepcopy(np.ndarray.flatten(spatial_array))
        des_vec = np.ndarray.flatten(self.design_region)
        x = spatial_vec[des_vec == 1]
        return x

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
        self.fields_current = False

    def _update_rho(self, grad, step_size):
        """ Manually updates the permittivity with the grad info """

        self.simulation.rho = self.simulation.rho + self.design_region * step_size * grad
        self.simulation.rho[self.simulation.rho < 0] = 0
        self.simulation.rho[self.simulation.rho > 1] = 1

        self.simulation.eps_r = rho2eps(self.simulation.rho, self.eps_m, self.W,
                                        eta=self.eta, beta=self.beta)
        self.fields_current = False

    def _step_adam(self, gradient, mopt_old, vopt_old, iteration, beta1, beta2, epsilon=1e-8):
        """ Performs one step of adam optimization"""

        mopt = beta1 * mopt_old + (1 - beta1) * gradient
        mopt_t = mopt / (1 - beta1**(iteration + 1))
        vopt = beta2 * vopt_old + (1 - beta2) * (np.square(gradient))
        vopt_t = vopt / (1 - beta2**(iteration + 1))
        grad_adam = mopt_t / (np.sqrt(vopt_t) + epsilon)

        return (grad_adam, mopt, vopt)

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

    def scan_frequency(self, Nf=50, df=1/20, pbar=True):
        """ Scans the objective function vs. frequency """

        # create frequencies (in Hz)
        delta_f = self.simulation.omega*df
        freqs = 1/2/np.pi*np.linspace(self.simulation.omega - delta_f/2,
                                      self.simulation.omega + delta_f/2,  Nf)

        if pbar:
            bar = progressbar.ProgressBar(max_value=Nf)

        # loop through frequencies
        objs = []
        for i, f in enumerate(freqs):

            if pbar:
                bar.update(i + 1)

            # make a new simulation object
            sim_new = copy.deepcopy(self.simulation)

            # reset the simulation to compute new A (hacky way of doing it)
            sim_new.omega = 2*np.pi*f
            sim_new.eps_r = self.simulation.eps_r

            self.fields_current = False

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

    def scan_power(self, probes=None, Ns=50, s_min=1e-2, s_max=1e2, solver='newton'):
        """ Scans the source amplitude and computes the objective function
            probes is a list of functions for computing the power, for example:
            [lambda simulation: simulation.flux_probe('x', [-NPML[0]-int(l/2/dl), ny + int(d/2/dl)], int(H/2/dl))]
        """

        if probes is None:
            raise ValueError("need to specify 'probes' kwarg as a list of functions for computing the power in each port.")
        num_probes = len(probes)

        # create src_amplitudes
        s_list = np.logspace(np.log10(s_min), np.log10(s_max), Ns)        

        bar = progressbar.ProgressBar(max_value=Ns)

        # transmission
        transmissions = [[] for _ in range(num_probes)]
        powers = []

        for i, s in enumerate(s_list):

            bar.update(i + 1)

            # make a new simulation object
            sim_new = copy.deepcopy(self.simulation)

            sim_new.modes[0].scale = s
            sim_new.modes[0].setup_src(sim_new)
            W_in = sim_new.W_in

            powers.append(W_in)

            if solver == 'hybrid':
                # NOTE: sometimes a mix of born and newtons method (hybrid) works best.  dont know why yet

                # compute the fields
                (_,_,_,c) = sim_new.solve_fields_nl(timing=False, averaging=False,
                            Estart=None, solver_nl='born', conv_threshold=1e-10,
                            max_num_iter=100)

                if c[-1] > 1e-10:
                    # compute the fields
                    (_,_,_,c) = sim_new.solve_fields_nl(timing=False, averaging=False,
                                Estart=None, solver_nl='newton', conv_threshold=1e-10,
                                max_num_iter=100)
            else:
                (_,_,_,c) = sim_new.solve_fields_nl(timing=False, averaging=False,
                            Estart=None, solver_nl=solver, conv_threshold=1e-10,
                            max_num_iter=100)

            # compute power transmission using each probe
            for probe_index, probe in enumerate(probes):
                W_out = probe(sim_new)
                transmissions[probe_index].append(W_out / W_in)

            # plt.show()

        return powers, transmissions

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

    def _check_temp_plt(self):
        if self.temp_plt is not None:
        # Clear the temp_plt folder from previous runs
            import os, shutil
            folder = self.temp_plt.folder
            if not os.path.exists(folder):
                os.makedirs(folder)
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)