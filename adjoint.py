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

	EPSILON_0_ = EPSILON_0*simulation.L0
	MU_0_ = MU_0*simulation.L0
	omega = simulation.omega
	(Nx,Ny) = (simulation.Nx, simulation.Ny)

	if simulation.pol == 'Ez':
		dAdeps = deps_region*omega**2*EPSILON_0_    # Note: physical constants go here if need be!
		Ez = simulation.fields['Ez']
		b_aj = -dJdfield(Ez)
		Ez_aj = adjoint_linear(simulation, b_aj)		 

		dJdeps = 2*np.real(Ez_aj*dAdeps*Ez)

	elif simulation.pol == 'Hz':
		dAdeps = deps_region*omega**2*EPSILON_0_    # Note: physical constants go here if need be!
		Hz = simulation.fields['Hz']
		Ex = simulation.fields['Ex']
		Ey = simulation.fields['Ey']

		b_aj = -dJdfield(Hz)

		(Ex_aj, Ey_aj) = adjoint_linear(simulation, b_aj)	

		dJdeps = 2*np.real(Ex_aj*dAdeps*Ex) + 2*np.real(Ey_aj*dAdeps*Ey)

	else:
		raise ValueError('Invalid polarization: {}'.format(str(self.pol)))

	return dJdeps


def adjoint_linear(simulation, b_aj, averaging=False, solver=DEFAULT_SOLVER, matrix_format=DEFAULT_MATRIX_FORMAT):
	# Compute the adjoint field for a linear problem
	# Note: the correct definition requires simulating with the transpose matrix A.T
	EPSILON_0_ = EPSILON_0*simulation.L0
	MU_0_ = MU_0*simulation.L0
	omega = simulation.omega

	(Nx,Ny) = (simulation.Nx, simulation.Ny)
	M = Nx*Ny
	A = simulation.A

	if simulation.pol == 'Ez':
		ez = solver_direct(A.T, b_aj, solver=solver)
		Ez = ez.reshape((Nx, Ny))

		return Ez

	elif simulation.pol == 'Hz':
		hz = solver_direct(A.T, b_aj, solver=solver)
		(Dyb, Dxb, Dxf, Dyf) = unpack_derivs(simulation.derivs)	
		(Dyb, Dxb, Dxf, Dyf) = (Dyb.T, Dxb.T, Dxf.T, Dyf.T)

		if averaging:
			vector_eps_x = grid_average(EPSILON_0_*simulation.eps_r, 'x').reshape((-1,))
			vector_eps_y = grid_average(EPSILON_0_*simulation.eps_r, 'y').reshape((-1,))
		else:
			vector_eps_x = EPSILON_0_*simulation.eps_r.reshape((-1,))
			vector_eps_y = EPSILON_0_*simulation.eps_r.reshape((-1,))

		T_eps_x_inv = sp.spdiags(1/vector_eps_x, 0, M, M, format=matrix_format)
		T_eps_y_inv = sp.spdiags(1/vector_eps_y, 0, M, M, format=matrix_format)
		
		# Note: to get the correct gradient in the end, we must use Dxf, Dyf here 	
		ex = -1/1j/omega * T_eps_x_inv.dot(Dyf).dot(hz)
		ey =  1/1j/omega * T_eps_y_inv.dot(Dxf).dot(hz)

		Ex = ex.reshape((Nx, Ny))
		Ey = ey.reshape((Nx, Ny))

		return (Ex, Ey)

	else:
		raise ValueError('Invalid polarization: {}'.format(str(self.pol)))


