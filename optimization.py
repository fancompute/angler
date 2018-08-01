from adjoint import dJdeps
from nonlinear_solvers import born_solve, newton_solve

import numpy as np
import copy
import progressbar


def _solve_nl(simulation, b, nonlinear_fn, nl_region, solver='born'):
	# convenience function for running solver

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


def run_optimization(simulation, b, nonlinear_fn, obj_fn, nl_region, design_region, Nsteps, eps_max, solver='born', step_size=0.2):
	# performs an optimization with gradient descent
	# NOTE:  will add adam or other methods later -T
	# NOTE2: this only works for objective functions of the nonlinear field for now

	# stores objective functions
	obj_fns = np.zeros((Nsteps,1))

	# make progressbar
	bar = progressbar.ProgressBar(max_value=Nsteps)

	for i in range(Nsteps):

		# display progressbar	
		bar.update(i)

		# solve for nonlinear fields
		(Ez_nl, convergence_array) = _solve_nl(simulation, b, nonlinear_fn, nl_region, solver='born')

		# compute the gradient
		grad = dJdeps(simulation, Ez_nl, nonlinear_fn, nl_region)

		# update permittivity based on gradient
		new_eps = _update_permittivity(simulation, grad, design_region, step_size, eps_max)

	return obj_fns



