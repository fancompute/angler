from adjoint import dJdeps_linear
from nonlinear_solvers import born_solve, newton_solve

import numpy as np
import copy
import progressbar


def _solve_nl(simulation, b, nonlinear_fn=None, nl_region=None, solver='born'):
	# convenience function for running solver

	if nonlinear_fn is None or nl_region is None:
		raise ValueError("'nonlinear_fn' and 'nl_region' must be supplied")
		
	if solver == 'born':
		(Ez_nl, convergence_array) = born_solve(simulation, simulation.eps_r, b, nonlinear_fn, nl_region,
												conv_threshold=1e-10,
												max_num_iter=50)
	elif solver == 'newton':
		(Ez_nl, convergence_array) = newton_solve(simulation, simulation.eps_r, b, nonlinear_fn, kerr_nl_de, nl_region,
												conv_threshold=1e-10,
												max_num_iter=50)
	else:
		raise AssertionError("solver must be one of {'born', 'newton'}")

	return (Ez_nl, convergence_array)


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

	# Thats IT!


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

	if state == 'NA':
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
		return J['total'](J_lin(Ez), J_nonlin(Ez_nl))



def run_optimization(simulation, b, J, dJdE, design_region, Nsteps, eps_max, solver='born', step_size=0.1):
	# performs an optimization with gradient descent
	# NOTE:  will add adam or other methods later -T
	# NOTE2: this only works for objective functions of the nonlinear field for now

	# stores objective functions
	obj_fns = np.zeros((Nsteps,1))

	# determine problem state from J and dJdE dictionaries
	state = check_J_state(J, dJdE)

	# make progressbar
	bar = progressbar.ProgressBar(max_value=Nsteps)

	for i in range(Nsteps):

		# display progressbar	
		bar.update(i+1)

		# solve for the gradient of the linear objective function (if supplied)
		if 'linear' in J and J['linear'] is not None:
			(Hx,Hy,Ez) = simulation.solve_fields(b)
			grad_lin = dJdeps_linear(simulation, design_region, J['linear'], dJdE['linear'], averaging=False)
		else:
			Ez = np.zeros(simulation.eps_r.shape)
			grad_lin = np.zeros(simulation.eps_r.shape)

		# solve for the gradient of the linear objective function (if supplied)
		if 'nonlinear' in J and J['nonlinear'] is not None:
			(Ez_nl, convergence_array) = _solve_nl(simulation, b, nonlinear_fn=None, nl_region=None, solver='born')
			grad_nonlin = dJdeps_nonlinear(simulation, design_region, J['linear'], dJdE['linear'],  nonlinear_fn, nl_region, averaging=False)
		else:
			Ez_nl = np.zeros(simulation.eps_r.shape)
			grad_nonlin = np.zeros(simulation.eps_r.shape)

		# add the gradients together depending on problem
		if 'total' in J and 'total' in dJdE and J['total'] is not None and dJdE['total'] is not None:
			grad = dJdE['total'](grad_lin, grad_nonlin)
		else:
			raise ValueError("J['total'] and dJdE['total'] must be supplied")

		# update permittivity based on gradient
		new_eps = _update_permittivity(simulation, grad, design_region, step_size, eps_max)

		# compute the objective function depending on what was supplied
		obj_fn = compute_objectivefn(Ez, Ez_nl, J, state)
		obj_fns[i] = obj_fn

		# want some way to print the obj function in the progressbar without adding new lines

	return obj_fns



