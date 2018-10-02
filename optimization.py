from adjoint import dJdeps_linear, dJdeps_nonlinear

import numpy as np
import copy
import progressbar
import matplotlib.pylab as plt


class Optimization():

    def __init__(self, Nsteps=100, eps_max=5, step_size=0.01, J=None, dJ=None,
                 field_start='linear', solver='born', opt_method='adam',
                 max_ind_shift=None, end_scale=None):

        if J is None:
            J = {}
        if dJ is None:
            dJ = {}

        # store all of the parameters associated with the optimization

        self.Nsteps = Nsteps
        self.eps_max = eps_max
        self.step_size = step_size
        self.field_start = field_start
        self.solver = solver
        self.opt_method = opt_method
        self.J = J
        self.dJ = dJ
        self.objs_tot = []
        self.objs_lin = []
        self.objs_nl = []
        self.W_in = []
        self.E2_in = []
        self.convergences = []
        self.max_ind_shift = max_ind_shift
        self.end_scale = end_scale

        # determine problem state ('linear', 'nonlinear', or 'both') from J and dJ dictionaries
        self._check_J_state()    # sets self.state

    def run(self, simulation, design_region):

        # store the parameters specific to this simulation
        self.simulation = simulation
        self.design_region = design_region

        # make progressbar
        bar = progressbar.ProgressBar(max_value=self.Nsteps)

        for i in range(self.Nsteps):

            # display progressbar
            bar.update(i + 1)

            # Store the starting linear permittivity 
            eps_lin = copy.deepcopy(self.simulation.eps_r)

            # perform src amplitude adjustment for index shift capping
            if self.state == 'both' and self.max_ind_shift is not None:
                ratio = np.inf     # ratio of actual index shift to allowed
                epsilon = 5e-2     # extra bit to subtract from src
                max_count = 30     # maximum amount to try
                count = 0
                while ratio > 1:
                    dn = self.compute_index_shift(simulation)
                    max_shift = np.max(dn)
                    ratio = max_shift / self.max_ind_shift
                    if count <= max_count:
                        simulation.src = simulation.src*(np.sqrt(1/ratio) - epsilon)
                        count += 1
                    # if you've gone over the max count, we've lost our patience.  Just manually decrease it.
                    else:
                        simulation.src = simulation.src*0.98

            # perform source scaling such that the final scale is end_scale times the initial scale
            if self.state == 'both' and self.end_scale is not None:
                if i==0:
                    scale_fact = np.power(self.end_scale, 1/(self.Nsteps-1))
                else:
                    for modei in simulation.modes:
                        modei.scale = modei.scale*scale_fact
                        simulation.setup_modes()

            # if the problem has a linear component
            if self.state == 'linear' or self.state == 'both':

                # solve for the linear fields and gradient of the linear objective function
                (Hx, Hy, Ez) = self.simulation.solve_fields()
                grad_lin = dJdeps_linear(self.simulation, design_region,
                                         self.dJ['dE_linear'], self.dJ['deps_linear'], averaging=False)

            # if the problem is purely nonlinear
            else:

                # just set the fields and gradients to zeros so they don't affect the nonlinear part
                Ez = np.zeros(self.simulation.eps_r.shape)
                grad_lin = np.zeros(self.simulation.eps_r.shape)

            # if the problem has a nonlinear component
            if self.state == 'nonlinear' or self.state == 'both':

                # error checking on the field_start parameter
                if self.field_start not in ['linear', 'previous']:
                    raise AssertionError(
                        "field_start must be one of {'linear', 'previous'}")

                # construct the starting field for the linear solver based on field_start and the iteration
                if self.field_start == 'linear' or i == 0:
                    Estart = None
                else:
                    Estart = Ez

                # solve for the nonlinear fields
                (Hx_nl, Hy_nl, Ez_nl, conv) = self.simulation.solve_fields_nl(timing=False,
                                                                              averaging=False, Estart=None,
                                                                              solver_nl=self.solver, conv_threshold=1e-10,
                                                                              max_num_iter=50)
                # add final convergence to the optimization list
                self.convergences.append(float(conv[-1]))

                # compute the gradient of the nonlinear objective function
                grad_nonlin = dJdeps_nonlinear(simulation, design_region, self.dJ['dE_nonlinear'], self.dJ['deps_nonlinear'], averaging=False)

                # Restore just the linear permittivity
                self.simulation.eps_r = eps_lin

            # if the problem is purely linear
            else:

                # just set the fields and gradients to zero so they don't affect linear part.
                Ez_nl = np.zeros(self.simulation.eps_r.shape)
                grad_nonlin = np.zeros(self.simulation.eps_r.shape)

            # add the gradients together depending on problem

            # compute the objective function depending on what was supplied
            (J_lin, J_nonlin, J_tot) = self._compute_objectivefn(Ez, Ez_nl, eps_lin)

            # compute the total gradient
            grad = self.dJ['total'](J_lin, J_nonlin, grad_lin, grad_nonlin)

            # update permittivity based on gradient
            if self.opt_method == 'descent':
                # gradient descent update
                new_eps = self._update_permittivity(grad, design_region)

            elif self.opt_method == 'adam':
                # adam update                
                if i == 0:
                    mopt = np.zeros((grad.shape))
                    vopt = np.zeros((grad.shape))

                (grad_adam, mopt, vopt) = self._step_adam(grad, mopt, vopt, i,
                                                          epsilon=1e-8,
                                                          beta1=0.9,
                                                          beta2=0.999)
                new_eps = self._update_permittivity(grad_adam, design_region)
            else:
                raise AssertionError(
                    "opt_method must be one of {'descent', 'adam'}")


            # want: some way to print the obj function in the progressbar
            # without adding new lines

        return new_eps

    def plt_objs(self, ax=None, scaled=None):

        iters = range(1, len(self.objs_tot) + 1)
        objs_tot = np.asarray(self.objs_tot)
        if self.state == 'both':
            objs_lin = np.asarray(self.objs_lin)
            objs_nl = np.asarray(self.objs_nl)

        if scaled=='W_in':
            objs_tot = objs_tot/np.asarray(self.W_in)
            if self.state == 'both':
                objs_lin = objs_lin/np.asarray(self.W_in)
                objs_nl = objs_nl/np.asarray(self.W_in)
        elif scaled=='E2_in':
            objs_tot = objs_tot/np.asarray(self.E2_in)
            if self.state == 'both':
                objs_lin = objs_lin/np.asarray(self.E2_in)
                objs_nl = objs_nl/np.asarray(self.E2_in)

        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout=True)

        ax.plot(iters, objs_tot)
        ax.set_xlabel('iteration number')
        ax.set_ylabel('objective function')
        ax.set_title('optimization results')

        if self.state == 'both':
            ax.plot(iters, objs_lin)
            ax.plot(iters, objs_nl)
            ax.legend(('total', 'linear', 'nonlinear'))

        return ax

    def check_deriv_lin(self, simulation, design_region, Npts=5):
        # checks the numerical derivative matches analytical.

        # how much to perturb eps for numerical gradient
        d_eps = 1e-4

        # make copy of original epsilon
        eps_orig = copy.deepcopy(simulation.eps_r)

        # solve for the linear fields and gradient of the linear objective function
        (_, _, Ez) = simulation.solve_fields()
        grad_avm = dJdeps_linear(simulation, design_region, self.dJ['dE_linear'], self.dJ['deps_linear'], averaging=False)
        J_orig = self.J['linear'](Ez, eps_orig)

        avm_grads = []
        num_grads = []

        # for a number of points
        for _ in range(Npts):

            # pick a random point within the design region
            x, y = np.where(design_region == 1)
            i = np.random.randint(len(x))
            pt = [x[i], y[i]]

            # create a new, perturbed permittivity
            eps_new = copy.deepcopy(simulation.eps_r)
            eps_new[pt[0], pt[1]] += d_eps

            # make a copy of the current simulation
            sim_new = copy.deepcopy(simulation)
            sim_new.eps_r = eps_new

            # solve for the fields with this new permittivity
            (_, _, Ez_new) = sim_new.solve_fields()
            J_new = self.J['linear'](Ez_new, eps_new)

            # compute the numerical gradient
            grad_num = (J_new - J_orig)/d_eps

            # append both gradients to lists
            avm_grads.append(grad_avm[pt[0], pt[1]])
            num_grads.append(grad_num)

        return avm_grads, num_grads

    def check_deriv_nonlin(self, simulation, design_region, Npts=5):
        """ checks whether the numerical derivative matches analytical """

        # how much to perturb epsilon for numerical gradient
        d_eps = 1e-4

        # make copy of original epsilon
        eps_orig = copy.deepcopy(simulation.eps_r)

        (_, _, Ez, _) = simulation.solve_fields_nl(timing=False,
                                                   averaging=False, Estart=None,
                                                   solver_nl='newton', conv_threshold=1e-10,
                                                   max_num_iter=50)

        # compute the gradient of the nonlinear objective function with AVM
        grad_avm = dJdeps_nonlinear(simulation, design_region, self.dJ['dE_nonlinear'], self.dJ['deps_nonlinear'],
                                    averaging=False)

        # compute original objective function (to compare with numerical)
        J_orig = self.J['nonlinear'](Ez, eps_orig)

        avm_grads = []
        num_grads = []

        # for a number of points
        for _ in range(Npts):

            # pick a random point within design region
            x, y = np.where(design_region == 1)
            i = np.random.randint(len(x))
            pt = [x[i], y[i]]

            # perturb the permittivity and store in a new array
            eps_new = copy.deepcopy(simulation.eps_r)
            eps_new[pt[0], pt[1]] += d_eps

            # create a deep copy of the current simulation object with new eps
            sim_new = copy.deepcopy(simulation)
            sim_new.eps_r = eps_new

            # solve for the new nonlinear fields
            (_, _, Ez_new, _) = sim_new.solve_fields_nl(timing=False,
                                                        averaging=False, Estart=None,
                                                        solver_nl='newton', conv_threshold=1e-10,
                                                        max_num_iter=50)

            # compute the new objective function
            J_new = self.J['nonlinear'](Ez_new, eps_new)

            # compute the numerical gradient
            grad_num = (J_new - J_orig)/d_eps

            # append both gradients to lists
            avm_grads.append(grad_avm[pt[0], pt[1]])
            num_grads.append(grad_num)

        return avm_grads, num_grads

    def compute_index_shift(self, simulation, full_nl=False):
        """ computes the max shift of refractive index caused by nonlinearity"""

        if full_nl:
            # true index shift with nonlinear solve            
            (_, _, Ez) = self.simulation.solve_fields_nl()
        else:
            # linear approximation to index shift
            (_, _, Ez) = self.simulation.solve_fields()
            simulation.compute_nl(Ez)

        # index shift
        dn = np.sqrt(np.real(simulation.eps_nl))

        # could np.max() it here if you want.  Returning array for now
        return dn

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

            # compute the fields depending on the state
            if self.state == 'linear' or self.state == 'both':
                (_, _, Ez) = sim_new.solve_fields()

            if self.state == 'nonlinear' or self.state == 'both':

                (_, _, Ez_nl, _) = sim_new.solve_fields_nl(timing=False,
                                                           averaging=False, Estart=None,
                                                           solver_nl='newton', conv_threshold=1e-10,
                                                           max_num_iter=50)

            # create placeholders for the fields if state is not 'both'
            if self.state == 'linear':
                Ez_nl = np.zeros(Ez.shape)
            elif self.state == 'nonlinear':
                Ez = np.zeros(Ez_nl.shape)

            # compute objective function and append to list
            _, _, obj_fn = self._compute_objectivefn(Ez, Ez_nl, self.simulation.eps_r)
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

    def _update_permittivity(self, grad, design_region):
        # updates the permittivity with the gradient info

        # deep copy original permittivity (deep for safety)
        eps_old = copy.deepcopy(self.simulation.eps_r)

        # update the old eps to get a new eps with the gradient
        eps_new = eps_old + self.design_region * self.step_size * grad

        # push back inside bounds
        eps_new[eps_new < 1] = 1
        eps_new[eps_new > self.eps_max] = self.eps_max

        # reset the epsilon of the simulation
        self.simulation.eps_r = eps_new

        return eps_new

    def _check_J_state(self):
        # does error checking on the objective function dictionaries and complains if they are wrong
        # sets a flag for the run_optimization function

        keys = ['linear', 'nonlinear', 'total']

        # first, set any unspecified values = None
        for k in keys:
            if k not in self.J:
                self.J[k] = None

        keys = ['dE_linear', 'dE_nonlinear', 'total', 'deps_linear', 'deps_nonlinear']
        for k in keys:
            if k not in self.dJ:
                self.dJ[k] = None


        # next, determine if linear problem, nonlinear problem, or both
        state = 'NA'
        if self.J['linear'] is not None and self.dJ['dE_linear'] is not None:
            state = 'linear'
        if self.J['nonlinear'] is not None and self.dJ['dE_nonlinear'] is not None:
            state = 'nonlinear' if state == 'NA' else 'both'

        if state == 'both':
            if 'total' not in self.J or self.J['total'] is None or 'total' not in self.dJ or self.dJ['total'] is None:
                raise ValueError(
                    "must supply functions in J['total'] and dJ['total']")

        elif state == 'linear':
            self.J['total'] = lambda J_lin, J_nonlin: J_lin
            self.dJ['total'] = lambda J_lin, J_nonlin, dJdE_lin, dJdE_nonlin: dJdE_lin

        elif state == 'nonlinear':
            self.J['total'] = lambda J_lin, J_nonlin: J_nonlin
            self.dJ['total'] = lambda J_lin, J_nonlin, dJdE_lin, dJdE_nonlin: dJdE_nonlin

        elif state == 'NA':
            raise ValueError(
                "must supply both J and dJ with functions for 'linear', 'nonlinear' or both")

        self.state = state

    def _compute_objectivefn(self, Ez, Ez_nl, eps_r):
        # does some error checking and returns the objective function

        assert self.state in ['linear', 'nonlinear', 'both']

        # give different objective function depending on state
        if self.state == 'linear':
            J_lin = self.J['linear'](Ez, eps_r)
            J_nl  = 0
            J_tot = self.J['total'](J_lin, J_nl)

        elif self.state == 'nonlinear':
            J_lin = 0
            J_nl  = self.J['nonlinear'](Ez, eps_r)
            J_tot = self.J['total'](J_lin, J_nl)

        else:
            J_lin = self.J['linear'](Ez, eps_r)
            J_nl = self.J['nonlinear'](Ez_nl, eps_r)
            J_tot = self.J['total'](J_lin, J_nl)
            self.objs_lin.append(J_lin)
            self.objs_nl.append(J_nl)

        self.objs_tot.append(J_tot)
        self.W_in.append(self.simulation.W_in)
        self.E2_in.append(self.simulation.E2_in)
        return (J_lin, J_nl, J_tot)

    def _step_adam(self, grad, mopt_old, vopt_old, iteration_index, epsilon=1e-8, beta1=0.9, beta2=0.999):
        mopt = beta1 * mopt_old + (1 - beta1) * grad
        mopt_t = mopt / (1 - beta1**(iteration_index + 1))
        vopt = beta2 * vopt_old + (1 - beta2) * (np.square(grad))
        vopt_t = vopt / (1 - beta2**(iteration_index + 1))
        grad_adam = mopt_t / (np.sqrt(vopt_t) + epsilon)

        return (grad_adam, mopt, vopt)
