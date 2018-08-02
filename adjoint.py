import numpy as np
import scipy.sparse as sp
from FDFD.linalg import solver_direct, unpack_derivs
from FDFD.constants import *


def dJdeps(simulation, Ez_nl, nonlinear_fn, nl_region):
	# computes the derivative of the objective function with respect to the permittivity

	# note, just making a fake gradient that is 1 at the center of the space
	grad = np.zeros(simulation.eps_r.shape)
	grad[int(simulation.Nx/2), int(simulation.Ny/2)] = 1

	return grad

# ADJOINT FUNCTIONS BELOW! VVVVV (or above if you want)

def dJdeps_linear(simulation, deps_region, J, dJdfield):
	# dJdfield is either dJdez or dJdhz
	# Note: we are assuming that the partial derivative of J w.r.t. e* is just (dJde)* and same for h

	(Nx,Ny) = (simulation.Nx, simulation.Ny)

	if simulation.pol == 'Ez':
		dAdeps = deps_region     # Note: physical constants go here if need be!
		Ez = simulation.fields['Ez']
		b_aj = -dJdfield(Ez)
		Ez_aj = adjoint_linear(simulation, b_aj)		 

		dJdeps = 2*np.real(Ez_aj*dAdeps*Ez)

	elif simulation.pol == 'Hz':
		dAdeps = deps_region/MU_0     # Note: physical constants go here if need be!
		Hz = simulation.fields['Hz']
		Ex = simulation.fields['Ex']
		Ey = simulation.fields['Ey']

		b_aj = -dJdfield(Hz)

		(Ex_aj, Ey_aj) = adjoint_linear(simulation, b_aj)	

		dJdeps = 2*np.real(Ex_aj*dAdeps*Ex) + 2*np.real(Ey_aj*dAdeps*Ey)

	else:
		raise ValueError('Invalid polarization: {}'.format(str(self.pol)))

	return dJdeps


def adjoint_linear(simulation, b_aj):
	# Compute the adjoint field for a linear problem
	# Note: the correct definition requires simulating with the transpose matrix A.T

	(Nx,Ny) = (simulation.Nx, simulation.Ny)
	A = simulation.A

	if simulation.pol == 'Ez':
		ez = solver_direct(A.T, b_aj)
		Ez = ez.reshape((Nx, Ny))

		return Ez

	elif simulation.pol == 'Hz':
		hz = solver_direct(A.T, b_aj)
		(Dyb, Dxb, Dxf, Dyf) = unpack_derivs(simulation.derivs)	
		(Dyb, Dxb, Dxf, Dyf) = (Dyb.T, Dxb.T, Dxf.T, Dyf.T)

		ex = -1/1j/simulation.omega/EPSILON_0 * Dyb.dot(hz)
		ey =  1/1j/simulation.omega/EPSILON_0 * Dxb.dot(hz)

		Ex = ex.reshape((Nx, Ny))
		Ey = ey.reshape((Nx, Ny))

		return (Ex, Ey)

	else:
		raise ValueError('Invalid polarization: {}'.format(str(self.pol)))


