import numpy as np
import scipy.sparse as sp
from fdfdpy.linalg import solver_direct, grid_average
from fdfdpy.derivatives import unpack_derivs
from fdfdpy.constants import *


def dJdeps(simulation, Ez_nl, nonlinear_fn, nl_region):
	# computes the derivative of the objective function with respect to the permittivity

	# note, just making a fake gradient that is 1 at the center of the space
	grad = np.zeros(simulation.eps_r.shape)
	grad[int(simulation.Nx/2), int(simulation.Ny/2)] = 1

	return grad

# ADJOINT FUNCTIONS BELOW! VVVVV (or above if you want)

def dJdeps_linear(simulation, design_region, J, dJdfield, averaging=False):
	# dJdfield is either dJdez or dJdhz
	# Note: we are assuming that the partial derivative of J w.r.t. e* is just (dJde)* and same for h

	EPSILON_0_ = EPSILON_0*simulation.L0
	MU_0_ = MU_0*simulation.L0
	omega = simulation.omega
	(Nx,Ny) = (simulation.Nx, simulation.Ny)

	if simulation.pol == 'Ez':
		dAdeps = design_region*omega**2*EPSILON_0_    # Note: physical constants go here if need be!
		Ez = simulation.fields['Ez']
		b_aj = -dJdfield(Ez)
		Ez_aj = adjoint_linear(simulation, b_aj)

		dJdeps = 2*np.real(Ez_aj*dAdeps*Ez)

	elif simulation.pol == 'Hz':
		dAdeps = design_region*omega**2*EPSILON_0_    # Note: physical constants go here if need be!
		Hz = simulation.fields['Hz']
		Ex = simulation.fields['Ex']
		Ey = simulation.fields['Ey']

		b_aj = -dJdfield(Hz)

		if averaging:
			(Ex_aj, Ey_aj) = adjoint_linear(simulation, b_aj, averaging=True)

			design_region_x = grid_average(design_region, 'x')
			dAdeps_x = design_region_x*omega**2*EPSILON_0_
			design_region_y = grid_average(design_region, 'y')
			dAdeps_y = design_region_y*omega**2*EPSILON_0_
			dJdeps = 2*np.real(Ex_aj*dAdeps_x*Ex) + 2*np.real(Ey_aj*dAdeps_y*Ey)

		else:
			(Ex_aj, Ey_aj) = adjoint_linear(simulation, b_aj, averaging=False)
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


# TO DO:

def dJdeps_nonlinear(simulation, design_region, J, dJdfield, nonlinear_fn, nl_region, nl_de, averaging=False):
	# Note: written only for Ez!
	# Note: we are assuming that the partial derivative of J w.r.t. e* is just (dJde)*

	EPSILON_0_ = EPSILON_0*simulation.L0
	MU_0_ = MU_0*simulation.L0
	omega = simulation.omega
	(Nx,Ny) = (simulation.Nx, simulation.Ny)

	if simulation.pol == 'Ez':
		dAdeps = design_region*omega**2*EPSILON_0_    # Note: physical constants go here if need be!
		Ez = simulation.fields['Ez']
		b_aj = -dJdfield(Ez)
		Ez_aj = adjoint_nonlinear(simulation, b_aj, nonlinear_fn, nl_region, nl_de)

		dJdeps = 2*np.real(Ez_aj*dAdeps*Ez)

		return dJdeps
	else:
		raise ValueError("Nonlinear adjoint works only for Ez polarization")


def adjoint_nonlinear(simulation, b_aj, nonlinear_fn, nl_region, nl_de,
					 averaging=False, solver=DEFAULT_SOLVER, matrix_format=DEFAULT_MATRIX_FORMAT):
	# Compute the adjoint field for a nonlinear problem
	# Note: written only for Ez!

	EPSILON_0_ = EPSILON_0*simulation.L0
	MU_0_ = MU_0*simulation.L0
	omega = simulation.omega

	(Nx,Ny) = (simulation.Nx, simulation.Ny)
	M = Nx*Ny

	if simulation.pol == 'Ez':
		Ez = simulation.fields['Ez']
		eps_lin = simulation.eps_r
		Anl = simulation.A + sp.spdiags(omega**2*EPSILON_0_*nonlinear_fn(Ez*nl_region, eps_lin).reshape((-1,)), 0, M, M, format=matrix_format)
		dAde = omega**2*EPSILON_0_*nl_de(Ez*nl_region, eps_lin)

		C11 = Anl + sp.spdiags((dAde*Ez).reshape((-1,)), 0, M, M, format=matrix_format)
		C12 = sp.spdiags((np.conj(dAde)*Ez).reshape((-1)), 0, M, M, format=matrix_format)
		C_full = sp.vstack((sp.hstack((C11, C12)), np.conj(sp.hstack((C12, C11)))))
		b_aj = b_aj.reshape((-1,))

		ez = solver_direct(C_full.T, np.vstack((b_aj, np.conj(b_aj))), solver=solver)

		if np.linalg.norm(ez[range(M)] - np.conj(ez[range(M, 2*M)])) > 1e-8:
			print('Adjoint field and conjugate do not match; something might be wrong')

		Ez = ez[range(M)].reshape((Nx, Ny))

		return Ez

	else:
		raise ValueError("Nonlinear adjoint works only for Ez polarization")
