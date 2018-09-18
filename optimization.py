from adjoint import dJdeps_linear, dJdeps_nonlinear

import numpy as np
import copy
import progressbar
import matplotlib.pylab as plt


class Optimization():

    def __init__(self, Nsteps=100, eps_max=5, step_size=0.01, J={}, dJdE={},
                 field_start='linear', solver='born', opt_method='adam',
                 max_ind_shift=None):

        # store all of the parameters associated with the optimization

        self.Nsteps = Nsteps
        self.eps_max = eps_max
        self.step_size = step_size
        self.field_start = field_start
        self.solver = solver
        self.opt_method = opt_method
        self.J = J
        self.dJdE = dJdE
        self.objs_tot = []
        self.objs_lin = []
        self.objs_nl = []
        self.convergences = []
        self.max_ind_shift = max_ind_shift

    def run(self, simulation, regions={}, nonlin_fns={}):

        # store the parameters specific to this simulation
        self.simulation = simulation
        self.regions = regions
        self.nonlin_fns = nonlin_fns

        # determine problem state ('linear', 'nonlinear', or 'both') from J and dJdE dictionaries
        self._check_J_state()    # sets self.state

        # unpack design and nonlinear function dictionaries
        (design_region, nl_region, nonlinear_fn, dnl_de) = self._unpack_dicts()

        # make progressbar
        bar = progressbar.ProgressBar(max_value=self.Nsteps)

        for i in range(self.Nsteps):

            # display progressbar
            bar.update(i + 1)

            # perform src amplitude adjustment for index shift capping
            if self.state == 'both' and self.max_ind_shift is not None:
                ratio = np.inf     # ratio of actual index shift to allowed
                epsilon = 5e-2     # extra bit to subtract from src
                max_count = 30     # maximum amount to try
                count = 0
                while ratio > 1:
                    dn = self.compute_index_shift(simulation, regions, nonlin_fns)
                    max_shift = np.max(dn)
                    ratio = max_shift / self.max_ind_shift
                    if count <= max_count:
                        simulation.src = simulation.src*(np.sqrt(1/ratio) - epsilon)
                        count += 1
                    # if you've gone over the max count, we've lost our patience.  Just manually decrease it.         
                    else:
                        simulation.src = simulation.src*0.98

            # if the problem has a linear component
            if self.state == 'linear' or self.state == 'both':

                # solve for the linear fields and gradient of the linear objective function
                (Hx, Hy, Ez) = self.simulation.solve_fields()
                grad_lin = dJdeps_linear(self.simulation, design_region, self.J[
                                         'linear'], self.dJdE['linear'], averaging=False)

            # if the problem is purely nonlinear
            else:

                # just set the fields and gradients to zeros so they don't affect the nonlinear part
                Ez = np.zeros(self.simulation.eps_r.shape)
                grad_lin = np.zeros(self.simulation.eps_r.shape)

            # if the problem has a nonlinear component
            if self.state == 'nonlinear' or self.state == 'both':

                # Store the starting linear permittivity (it will be changed by the nonlinear solvers...)
                eps_lin = copy.deepcopy(self.simulation.eps_r)

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
                (Hx_nl, Hy_nl, Ez_nl, conv) = self.simulation.solve_fields_nl(nonlinear_fn, nl_region,
                                                                           dnl_de=dnl_de, timing=False,
                                                                           averaging=False, Estart=None,
                                                                           solver_nl=self.solver, conv_threshold=1e-10,
                                                                           max_num_iter=50)
                # add final convergence to the optimization list
                self.convergences.append(float(conv[-1]))

                # compute the gradient of the nonlinear objective function
                grad_nonlin = dJdeps_nonlinear(simulation, design_region, self.J['nonlinear'], self.dJdE['nonlinear'],
                                               nonlinear_fn, nl_region, dnl_de, averaging=False)

                # Restore just the linear permittivity
                self.simulation.eps_r = eps_lin

            # if the problem is purely linear
            else:

                # just set the fields and gradients to zero so they don't affect linear part.
                Ez_nl = np.zeros(self.simulation.eps_r.shape)
                grad_nonlin = np.zeros(self.simulation.eps_r.shape)

            # add the gradients together depending on problem
            grad = self.dJdE['total'](grad_lin, grad_nonlin)

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

            # compute the objective function depending on what was supplied
            obj_fn = self._compute_objectivefn(Ez, Ez_nl)

            # want: some way to print the obj function in the progressbar
            # without adding new lines

        return new_eps

    def plt_objs(self, ax=None):

        iters = range(1, len(self.objs_tot) + 1)

        if ax is None:
            fig, ax = plt.subplots(1, constrained_layout=True)

        ax.plot(iters, self.objs_tot)
        ax.set_xlabel('iteration number')
        ax.set_ylabel('objective function')
        ax.set_title('optimization results')

        if self.state == 'both':
            ax.plot(iters, self.objs_lin)
            ax.plot(iters, self.objs_nl)
            ax.legend(('total', 'linear', 'nonlinear'))

        return ax

    def check_deriv(self, simulation, design_region):
        # checks the numerical derivative matches analytical.

        # pick a few points in the design region.
        Npts = 5
        d_eps = 1e-8

        # solve for the linear fields and gradient of the linear objective function
        (_, _, Ez) = simulation.solve_fields()
        grad_avm = dJdeps_linear(simulation, design_region, self.J[
                                 'linear'], self.dJdE['linear'], averaging=False)
        J_orig = self.J['linear'](Ez)

        avm_grads = []
        num_grads = []

        for pt_index in range(Npts):
            x, y = np.where(design_region == 1)
            i = np.random.randint(len(x))
            pt = [x[i], y[i]]

            eps_new = copy.deepcopy(simulation.eps_r)
            eps_new[pt[0], pt[1]] += d_eps

            sim_new = copy.deepcopy(simulation)
            sim_new.reset_eps(eps_new)

            (_, _, Ez_new) = sim_new.solve_fields()
            J_new = self.J['linear'](Ez_new)

            avm_grads.append(grad_avm[pt[0], pt[1]])
            num_grads.append((J_new - J_orig)/d_eps)

            # import matplotlib.pylab as plt; import pdb; pdb.set_trace()

        return avm_grads, num_grads

    def compute_index_shift(self, simulation, regions, nonlin_fns):
        # computes the max shift of refractive index caused by nonlinearity

        # solve linear fields
        (Hx, Hy, Ez) = self.simulation.solve_fields()

        # get region of nonlinearity
        nl_region = regions['nonlin']

        # how eps_r changes with Ez
        eps_nl = nonlin_fns['eps_nl']

        # relative permittivity shift
        deps = eps_nl(Ez*nl_region)

        # index shift
        dn = np.sqrt(deps)

        # could np.max() it here if you want.  Returning array for now
        return dn

    def _update_permittivity(self, grad, design_region):
        # updates the permittivity with the gradient info

        # deep copy original permittivity (deep for safety)
        eps_old = copy.deepcopy(self.simulation.eps_r)

        # update the old eps to get a new eps with the gradient
        eps_new = eps_old + self.regions['design'] * self.step_size * grad

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
            if k not in self.dJdE:
                self.dJdE[k] = None

        # next, determine if linear problem, nonlinear problem, or both
        state = 'NA'
        if self.J['linear'] is not None and self.dJdE['linear'] is not None:
            state = 'linear'
        if self.J['nonlinear'] is not None and self.dJdE['nonlinear'] is not None:
            state = 'nonlinear' if state == 'NA' else 'both'

        if state == 'both':
            if 'total' not in self.J or self.J['total'] is None or 'total' not in self.dJdE or self.dJdE['total'] is None:
                raise ValueError(
                    "must supply functions in J['total'] and dJdE['total']")

        elif state == 'linear':
            self.J['total'] = lambda J_lin, J_nonlin: J_lin
            self.dJdE['total'] = lambda dJdE_lin, dJdE_nonlin: dJdE_lin

        elif state == 'nonlinear':
            self.J['total'] = lambda J_lin, J_nonlin: J_nonlin
            self.dJdE['total'] = lambda dJdE_lin, dJdE_nonlin: dJdE_nonlin

        elif state == 'NA':
            raise ValueError(
                "must supply both J and dJdE with functions for 'linear', 'nonlinear' or both")

        self.state = state

    def _compute_objectivefn(self, Ez, Ez_nl):
        # does some error checking and returns the objective function

        assert self.state in ['linear', 'nonlinear', 'both']

        # give different objective function depending on state
        if self.state == 'linear':
            J_tot = self.J['total'](self.J['linear'](Ez), 0)

        elif self.state == 'nonlinear':
            J_tot = self.J['total'](0, self.J['nonlinear'](Ez_nl))

        else:
            J_lin = self.J['linear'](Ez)
            J_nl = self.J['nonlinear'](Ez_nl)
            J_tot = self.J['total'](J_lin, J_nl)
            self.objs_lin.append(J_lin)
            self.objs_nl.append(J_nl)

        self.objs_tot.append(J_tot)
        return J_tot

    def _unpack_dicts(self):
        # does error checking on the regions and nonlin_fns dictionary and
        # returns results.

        # unpack regions
        if 'design' not in self.regions:
            raise ValueError(
                "must supply a 'design' region to regions dictionary")
        design_region = self.regions['design']

        if self.state == 'nonlinear' or self.state == 'both':
            if 'nonlin' not in self.regions:
                raise ValueError(
                    "must supply a 'nonlin' region to regions dictionary")
            nonlin_region = self.regions['nonlin']

            # unpack nonlinear functions (if state is 'nonlinear' or 'both')
            if 'eps_nl' not in self.nonlin_fns or 'dnl_de' not in self.nonlin_fns:
                raise ValueError(
                    "must supply 'eps_nl' and 'dnl_de' functions to nonlin_fns dictionary")
            eps_nl = self.nonlin_fns['eps_nl']
            dnl_de = self.nonlin_fns['dnl_de']
            if eps_nl is None or dnl_de is None:
                raise ValueError(
                    "must supply 'eps_nl' and 'dnl_de' functions to nonlin_fns dictionary")
        else:
            nonlin_region = eps_nl = dnl_de = None

        return (design_region, nonlin_region, eps_nl, dnl_de)

    def _step_adam(self, grad, mopt_old, vopt_old, iteration_index, epsilon=1e-8, beta1=0.9, beta2=0.999):
        mopt = beta1 * mopt_old + (1 - beta1) * grad
        mopt_t = mopt / (1 - beta1**(iteration_index + 1))
        vopt = beta2 * vopt_old + (1 - beta2) * (np.square(grad))
        vopt_t = vopt / (1 - beta2**(iteration_index + 1))
        grad_adam = mopt_t / (np.sqrt(vopt_t) + epsilon)

        return (grad_adam, mopt, vopt)
