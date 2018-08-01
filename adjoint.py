import numpy as np


def dJdeps(simulation, Ez_nl, nonlinear_fn, nl_region):
	# computes the derivative of the objective function with respect to the permittivity

	# note, just making a fake gradient that is 1 at the center of the space
	grad = np.zeros(simulation.eps_r.shape)
	grad[int(simulation.Nx/2), int(simulation.Ny/2)] = 1

	return grad

# ADJOINT FUNCTIONS BELOW! VVVVV (or above if you want)