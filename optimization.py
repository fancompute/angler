from adjoint import dJdeps_linear, dJdeps_nonlinear

import numpy as np
import copy
import progressbar

def _update_permittivity(simulation, grad, design_region, step_size, eps_max):
	# updates the permittivity with the gradient info

	# deep copy original permittivity (deep for safety)
	eps_old = copy.deepcopy(simulation.eps_r)

	# update the old eps to get a new eps with the gradient
	eps_new = eps_old + design_region*step_size*grad

	# push back inside bounds
	eps_new[eps_new < 1] = 1
	eps_new[eps_new > eps_max] = eps_max

	# reset the epsilon of the simulation
	simulation.reset_eps(eps_new)


def check_J_state(J, dJdE):
	# does error checking on the objective function dictionaries and complains if they are wrong
	# sets a flag for the run_optimization function

	keys = ['linear', 'nonlinear', 'total']

	# first, set any unspecified values = None
	for k in keys:
		if k not in J:
			J[k] = None
		if k not in dJdE:
			dJdE[k] = None

	# next, determine if linear problem, nonlinear problem, or both
	state = 'NA'
	if J['linear'] is not None and dJdE['linear'] is not None:
		state = 'linear'
	if J['nonlinear'] is not None and dJdE['nonlinear'] is not None:
		state = 'nonlinear'	if state == 'NA' else 'both'

	if state == 'both':
		if 'total' not in J or J['total'] is None or 'total' not in dJdE or dJdE['total'] is None:
			raise ValueError("must supply functions in J['total'] and dJdE['total']")

	elif state == 'linear':
		J['total']    = lambda J_lin, J_nonlin: J_lin
		dJdE['total'] = lambda dJdE_lin, dJdE_nonlin: dJdE_lin

	elif state == 'nonlinear':
		J['total']    = lambda J_lin, J_nonlin: J_nonlin
		dJdE['total'] = lambda dJdE_lin, dJdE_nonlin: dJdE_nonlin

	elif state == 'NA':
		raise ValueError("must supply both J and dJdE with functions for 'linear', 'nonlinear' or both")

	return state


def compute_objectivefn(Ez, Ez_nl, J, state):
	# does some error checking and returns the objective function

	assert state in ['linear', 'nonlinear', 'both']

	# give different objective function depending on state
	if state == 'linear':
		return J['total'](J['linear'](Ez), 0)
	elif state == 'nonlinear':
		return J['total'](0, J['nonlinear'](Ez_nl))
	elif state == 'both':
		return J['total'](J['linear'](Ez), J['nonlinear'](Ez_nl))


def unpack_dicts(state, regions, nonlin_fns):
	# does error checking on the regions and nonlin_fns dictionary and returns results.

	# unpack regions
	if 'design' not in regions:
		raise ValueError("must supply a 'design' region to regions dictionary")
	design_region = regions['design']

	if state == 'nonlinear' or state == 'both':
		if 'nonlin' not in regions:
			raise ValueError("must supply a 'nonlin' region to regions dictionary")
		nonlin_region = regions['nonlin']

		# unpack nonlinear functions (if state is 'nonlinear' or 'both')
		if 'deps_de' not in nonlin_fns or 'dnl_de' not in nonlin_fns:
			raise ValueError("must supply 'deps_de' and 'dnl_de' functions to nonlin_fns dictionary")
		deps_de = nonlin_fns['deps_de']
		dnl_de  = nonlin_fns['dnl_de']
		if deps_de is None or dnl_de is None:
			raise ValueError("must supply 'deps_de' and 'dnl_de' functions to nonlin_fns dictionary")
	else:
		nonlin_region = deps_de = dnl_de = None

	return (design_region, nonlin_region, deps_de, dnl_de)


def run_optimization(simulation, b, J, dJdE, Nsteps, eps_max, regions={}, nonlin_fns={}, field_start='linear', solver='born', step_size=0.1):
	# performs an optimization with gradient descent
	# NOTE:  will add adam or other methods later -T
	# NOTE2: this only works for objective functions of the nonlinear field for now

	# stores objective functions
	obj_fns = np.zeros((Nsteps,1))

	# determine problem state ('linear', 'nonlinear', or 'both') from J and dJdE dictionaries
	state = check_J_state(J, dJdE)

	# unpack design and nonlinear function dictionaries
	(design_region, nl_region, nonlinear_fn, dnl_de) = unpack_dicts(state, regions, nonlin_fns)

	# make progressbar
	bar = progressbar.ProgressBar(max_value=Nsteps)

	for i in range(Nsteps):

		# display progressbar	
		bar.update(i+1)

		# if the problem has a linear component
		if state == 'linear' or state == 'both':

			# solve for the linear fields and gradient of the linear objective function
			(Hx,Hy,Ez) = simulation.solve_fields(b)
			grad_lin = dJdeps_linear(simulation, design_region, J['linear'], dJdE['linear'], averaging=False)

		# if the problem is purely nonlinear
		else:

			# just set the fields and gradients to zeros so they don't affect the nonlinear part
			Ez = np.zeros(simulation.eps_r.shape)
			grad_lin = np.zeros(simulation.eps_r.shape)


		# if the problem has a nonlinear component
		if state == 'nonlinear' or state == 'both':

			# Store the starting linear permittivity (it will be changed by the nonlinear solvers...)
			eps_lin = simulation.eps_r

			# error checking on the field_start parameter
			if field_start not in ['linear','previous']:
				raise AssertionError("field_start must be one of {'linear', 'previous'}")

			# construct the starting field for the linear solver based on field_start and the iteration
			Estart = None if field_start =='linear' or i==0 else Ez

			# solve for the nonlinear fields
			(Hx_nl, Hy_nl, Ez_nl, _) = simulation.solve_fields_nl(b, nonlinear_fn, nl_region, 
										   dnl_de=dnl_de, timing=False, 
										   averaging=False, Estart=None, 
										   solver_nl=solver, conv_threshold=1e-10,
										   max_num_iter=50)
			
			# compute the gradient of the nonlinear objective function
			grad_nonlin = dJdeps_nonlinear(simulation, design_region, J['nonlinear'], dJdE['nonlinear'],
										 nonlinear_fn, nl_region, dnl_de, averaging=False)

			# Restore just the linear permittivity
			simulation.reset_eps(eps_lin)

		# if the problem is purely linear
		else:

			# just set the fields and gradients to zero so they don't affect linear part.
			Ez_nl = np.zeros(simulation.eps_r.shape)
			grad_nonlin = np.zeros(simulation.eps_r.shape)

		# add the gradients together depending on problem
		grad = dJdE['total'](grad_lin, grad_nonlin)

		# update permittivity based on gradient
		new_eps = _update_permittivity(simulation, grad, design_region, step_size, eps_max)

		# compute the objective function depending on what was supplied
		obj_fn = compute_objectivefn(Ez, Ez_nl, J, state)
		obj_fns[i] = obj_fn

		# want: some way to print the obj function in the progressbar without adding new lines

	return (new_eps, obj_fns)



