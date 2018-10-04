import sys
sys.path.append(".")

from adjoint_scipy import gradient

import numpy as np
import copy
import progressbar
import matplotlib.pylab as plt
from scipy.optimize import minimize
from autograd import grad

class Optimization_Scipy():

    def __init__(self, J=None, Nsteps=100, eps_max=5, field_start='linear', nl_solver='newton'):

        # store all of the parameters associated with the optimization
        self.Nsteps = Nsteps
        self.eps_max = eps_max
        self.field_start = field_start
        self.nl_solver = nl_solver
        self._J = J

        # compute the jacobians of J and store these (note, might want to do this each time J changes)
        self.dJ = self._autograd_dJ(J)

    def __repr__(self):
        return "Optimization_Scipy(Nsteps={}, eps_max={}, J={}, field_start={}, nl_solver={})".format(
            self.Nsteps, self.eps_max, self.J, self.field_start, self.nl_solver)

    def __str__(self):
        return self.__repr__

    @property
    def J(self):
        return self._J
    
    @J.setter
    def J(self, J):
        self._J = J
        self.dJ = self._autograd_dJ(J)

    def _autograd_dJ(self, J):
        """Uses autograd to compute jacobians of J with respect to each argument"""
        dJ = {}
        dJ['lin'] = grad(J, 0)
        dJ['nl']  = grad(J, 1)
        dJ['eps'] = grad(J, 2)
        return dJ

    def compute_J(self, simulation):
        """Returns the current objective function of a simulation"""
        (_, _, Ez) = simulation.solve_fields()
        (_, _, Ez_nl, _) = simulation.solve_fields_nl()
        eps = simulation.eps_r
        return self.J(Ez, Ez_nl, eps)

    def compute_dJ(self, simulation, design_region):
        """Returns the current objective function of a simulation"""
        (_, _, Ez) = simulation.solve_fields()
        (_, _, Ez_nl, _) = simulation.solve_fields_nl()   
        arguments = (Ez, Ez_nl, simulation.eps_r) 
        return gradient(simulation, self.dJ, design_region, arguments)

    def _set_design_region(self, x, simulation, design_region):
        """Takes a vector x and inserts it into the design region of simulation's permittivity"""
        eps_vec = copy.deepcopy(np.ndarray.flatten(simulation.eps_r))
        des_vec = np.ndarray.flatten(design_region)
        eps_vec[des_vec == 1] = x
        eps_new = np.reshape(eps_vec, simulation.eps_r.shape)        
        simulation.eps_r = eps_new

    def _get_design_region(self, spatial_array, design_region):
        """returns a vector of a spatial array"""
        spatial_vec = copy.deepcopy(np.ndarray.flatten(spatial_array))
        des_vec = np.ndarray.flatten(design_region)
        x = spatial_vec[des_vec == 1]
        return x

    def run(self, simulation, design_region, method='LBFGS'):
        """Switches between different optimization methods"""
        allowed = ['LBFGS']
        if method == 'LBFGS':
             self._run_LBFGS(simulation, design_region)
        else:
            raise ValueError("'method' must be in {}".format(allowed))

    def _run_LBFGS(self, simulation, design_region):
        """Performs L-BGFS Optimization of objective function w.r.t. eps_r"""
        self.simulation = simulation
        self.design_region = design_region
        self.objfn_list = []        

        def _objfn(x, *argv):
            """Returns objetive function given some permittivity distribution"""

            # make a simulation copy
            sim = copy.deepcopy(self.simulation)
            self._set_design_region(x, sim, self.design_region)
            
            J = self.compute_J(sim)
            self.objfn_list.append(J)

            # return minus J because we technically will minimize
            return -J

        def _grad(x,  *argv):
            """Returns objetive function given some permittivity distribution"""

            # make a simulation copy
            sim = copy.deepcopy(simulation)
            self._set_design_region(x, sim, self.design_region)

            # compute gradient, extract design region, turn into vector, return
            gradient = self.compute_dJ(sim, self.design_region)                    
            gradient_vec = self._get_design_region(gradient, self.design_region)

            return -gradient_vec

        # setup bounds on epsilon
        eps_bounds = tuple([(1, self.eps_max) for _ in range(np.sum(design_region==1))])

        # starting eps within design region
        x0 = self._get_design_region(self.simulation.eps_r, self.design_region)

        res = minimize(_objfn, x0, args=None, method="L-BFGS-B", jac=_grad, 
                       bounds=eps_bounds, callback=None, options = {
                            'disp' : 1,
                            'maxiter' : self.Nsteps
                       })

        self._set_design_region(res.x, simulation, design_region)

    def check_deriv(self, simulation, design_region, Npts=5, d_eps=1e-4):
        """ returns a list of analytical and numerical derivatives to check gradient accuracy"""

        # make copy of original epsilon
        eps_orig = copy.deepcopy(simulation.eps_r)

        # solve for the linear fields and gradient of the linear objective function
        grad_avm = self.compute_dJ(simulation, design_region)
        J_orig = self.compute_J(simulation)

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
            J_new = self.compute_J(sim_new)

            # compute the numerical gradient
            grad_num = (J_new - J_orig)/d_eps

            # append both gradients to lists
            avm_grads.append(grad_avm[pt[0], pt[1]])
            num_grads.append(grad_num)

        return avm_grads, num_grads

    def plt_objs(self, ax=None):
        """Plot objective functions over iteration"""
        ax.plot(range(1, len(self.objfn_list) + 1),  self.objfn_list)
        ax.set_xlabel('iteration number')
        ax.set_ylabel('objective function')
        ax.set_title('optimization results')
        return ax

    def compute_index_shift(self, simulation):
        """ computes array of nonlinear refractive index shift"""           
        _ = self.simulation.solve_fields_nl()
        dn = np.sqrt(np.real(simulation.eps_nl))
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

            # compute the fields
            (_, _, Ez) = sim_new.solve_fields()
            (_, _, Ez_nl, _) = sim_new.solve_fields_nl()

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

    def _step_adam(self, grad, mopt_old, vopt_old, iteration_index, epsilon=1e-8, beta1=0.999, beta2=0.999):
        mopt = beta1 * mopt_old + (1 - beta1) * grad
        mopt_t = mopt / (1 - beta1**(iteration_index + 1))
        vopt = beta2 * vopt_old + (1 - beta2) * (np.square(grad))
        vopt_t = vopt / (1 - beta2**(iteration_index + 1))
        grad_adam = mopt_t / (np.sqrt(vopt_t) + epsilon)

        return (grad_adam, mopt, vopt)
