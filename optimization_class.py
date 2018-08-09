from adjoint import dJdeps_linear, dJdeps_nonlinear

import numpy as np
import copy
import progressbar


class Optimization():


	def __init__(self, Nsteps=100, eps_max=5, step_size=None, regions={}, nonlin_fns={}, field_start='linear', solver='born'):

		# store all of the parameters associated with the optimization

		self.Nsteps 	 = Nsteps
		self.eps_max	 = eps_max
		self.step_size   = step_size
		self.regions     = regions
		self.nonlin_fns  = nonlin_fns
		self.field_start = field_start					
		self.solver      = solver


	def run(self, simulation, b, J, dJdE):
		
		# store the parameters specific to this simulation and obj_fn

		self.simulation  = simulation
		self.b           = b
		self.J    		 = J
		self.dJdE 		 = dJdE


		# stores objective functions
		obj_fns = np.zeros((self.Nsteps,1))

		# determine problem state ('linear', 'nonlinear', or 'both') from J and dJdE dictionaries
		self._check_J_state()    # sets self.state

		# unpack design and nonlinear function dictionaries
		(design_region, nl_region, nonlinear_fn, dnl_de) = self._unpack_dicts()

		# make progressbar
		bar = progressbar.ProgressBar(max_value=self.Nsteps)

		for i in range(self.Nsteps):

			# display progressbar	
			bar.update(i+1)

			# if the problem has a linear component
			if self.state == 'linear' or self.state == 'both':

				# solve for the linear fields and gradient of the linear objective function
				(Hx,Hy,Ez) = self.simulation.solve_fields(b)
				grad_lin = dJdeps_linear(self.simulation, design_region, self.J['linear'], self.dJdE['linear'], averaging=False)

			# if the problem is purely nonlinear
			else:

				# just set the fields and gradients to zeros so they don't affect the nonlinear part
				Ez = np.zeros(self.simulation.eps_r.shape)
				grad_lin = np.zeros(self.simulation.eps_r.shape)


			# if the problem has a nonlinear component
			if self.state == 'nonlinear' or self.state == 'both':

				# Store the starting linear permittivity (it will be changed by the nonlinear solvers...)
				eps_lin = self.simulation.eps_r

				# error checking on the field_start parameter
				if self.field_start not in ['linear','previous']:
					raise AssertionError("field_start must be one of {'linear', 'previous'}")

				# construct the starting field for the linear solver based on field_start and the iteration
				Estart = None if self.field_start =='linear' or i==0 else Ez

				# solve for the nonlinear fields
				(Hx_nl, Hy_nl, Ez_nl, _) = self.simulation.solve_fields_nl(self.b, nonlinear_fn, nl_region, 
											   dnl_de=dnl_de, timing=False, 
											   averaging=False, Estart=None, 
											   solver_nl=self.solver, conv_threshold=1e-10,
											   max_num_iter=50)
				
				# compute the gradient of the nonlinear objective function
				grad_nonlin = dJdeps_nonlinear(simulation, design_region, self.J['nonlinear'], self.dJdE['nonlinear'],
											 nonlinear_fn, nl_region, dnl_de, averaging=False)

				# Restore just the linear permittivity
				self.simulation.reset_eps(eps_lin)

			# if the problem is purely linear
			else:

				# just set the fields and gradients to zero so they don't affect linear part.
				Ez_nl = np.zeros(self.simulation.eps_r.shape)
				grad_nonlin = np.zeros(self.simulation.eps_r.shape)

			# add the gradients together depending on problem
			grad = self.dJdE['total'](grad_lin, grad_nonlin)

			# update permittivity based on gradient
			new_eps = self._update_permittivity(grad, design_region)

			# compute the objective function depending on what was supplied
			obj_fn = self._compute_objectivefn(Ez, Ez_nl)
			obj_fns[i] = obj_fn

			# want: some way to print the obj function in the progressbar without adding new lines

		return (new_eps, obj_fns)


	def _update_permittivity(self, grad, design_region):
		# updates the permittivity with the gradient info

		# deep copy original permittivity (deep for safety)
		eps_old = copy.deepcopy(self.simulation.eps_r)

		# update the old eps to get a new eps with the gradient
		eps_new = eps_old + self.regions['design']*self.step_size*grad

		# push back inside bounds
		eps_new[eps_new < 1] = 1
		eps_new[eps_new > self.eps_max] = self.eps_max

		# reset the epsilon of the simulation
		self.simulation.reset_eps(eps_new)

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
			state = 'nonlinear'	if state == 'NA' else 'both'

		if state == 'both':
			if 'total' not in self.J or self.J['total'] is None or 'total' not in self.dJdE or self.dJdE['total'] is None:
				raise ValueError("must supply functions in J['total'] and dJdE['total']")

		elif state == 'linear':
			self.J['total']    = lambda J_lin, J_nonlin: J_lin
			self.dJdE['total'] = lambda dJdE_lin, dJdE_nonlin: dJdE_lin

		elif state == 'nonlinear':
			self.J['total']    = lambda J_lin, J_nonlin: J_nonlin
			self.dJdE['total'] = lambda dJdE_lin, dJdE_nonlin: dJdE_nonlin

		elif state == 'NA':
			raise ValueError("must supply both J and dJdE with functions for 'linear', 'nonlinear' or both")

		self.state = state


	def _compute_objectivefn(self, Ez, Ez_nl):
		# does some error checking and returns the objective function

		assert self.state in ['linear', 'nonlinear', 'both']

		# give different objective function depending on state
		if self.state == 'linear':
			return self.J['total'](self.J['linear'](Ez), 0)
		elif self.state == 'nonlinear':
			return self.J['total'](0, self.J['nonlinear'](Ez_nl))
		elif self.state == 'both':
			return self.J['total'](self.J['linear'](Ez), self.J['nonlinear'](Ez_nl))


	def _unpack_dicts(self):
		# does error checking on the regions and nonlin_fns dictionary and returns results.

		# unpack regions
		if 'design' not in self.regions:
			raise ValueError("must supply a 'design' region to regions dictionary")
		design_region = self.regions['design']

		if self.state == 'nonlinear' or self.state == 'both':
			if 'nonlin' not in self.regions:
				raise ValueError("must supply a 'nonlin' region to regions dictionary")
			nonlin_region = self.regions['nonlin']

			# unpack nonlinear functions (if state is 'nonlinear' or 'both')
			if 'deps_de' not in self.nonlin_fns or 'dnl_de' not in self.nonlin_fns:
				raise ValueError("must supply 'deps_de' and 'dnl_de' functions to nonlin_fns dictionary")
			deps_de = self.nonlin_fns['deps_de']
			dnl_de  = self.nonlin_fns['dnl_de']
			if deps_de is None or dnl_de is None:
				raise ValueError("must supply 'deps_de' and 'dnl_de' functions to nonlin_fns dictionary")
		else:
			nonlin_region = deps_de = dnl_de = None

		return (design_region, nonlin_region, deps_de, dnl_de)
