import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from copy import deepcopy

from scipy.optimize import newton_krylov, anderson

from angler.linalg import grid_average, solver_direct, solver_complex2real
from angler.derivatives import unpack_derivs
from angler.constants import (DEFAULT_LENGTH_SCALE, DEFAULT_MATRIX_FORMAT,
							  DEFAULT_SOLVER, EPSILON_0, MU_0)


def born_solve(simulation,
			   Estart=None, conv_threshold=1e-10, max_num_iter=50,
			   averaging=True):
	# solves for the nonlinear fields

	# Stores convergence parameters
	conv_array = np.zeros((max_num_iter, 1))

	# Defne the starting field for the simulation
	if Estart is None:
		if simulation.fields[simulation.pol] is None:
			# if its not specified, nor already, solve for the linear fields			
			(_, _, Fz) = simulation.solve_fields()
		else:
			# if its not specified, but stored, use that one
			Fz = deepcopy(simulation.fields[simulation.pol])
	else:
		# otherwise, use the specified version
		Fz = Estart

	# Solve iteratively
	for istep in range(max_num_iter):

		Fprev = Fz

		# set new permittivity
		simulation.compute_nl(Fprev)

		(Fx, Fy, Fz) = simulation.solve_fields(include_nl=True)

		# get convergence and break
		convergence = la.norm(Fz - Fprev)/la.norm(Fz)
		conv_array[istep] = convergence

		# if below threshold, break and return
		if convergence < conv_threshold:
			break

	if convergence > conv_threshold:
		print("the simulation did not converge, reached {}".format(convergence))

	return (Fx, Fy, Fz, conv_array)

def newton_solve(simulation,
				 Estart=None, conv_threshold=1e-10, max_num_iter=50,
				 averaging=True, solver=DEFAULT_SOLVER, jac_solver='c2r',
				 matrix_format=DEFAULT_MATRIX_FORMAT):
	# solves for the nonlinear fields using Newton's method

	# Stores convergence parameters
	conv_array = np.zeros((max_num_iter, 1))

	# num. columns and rows of A
	Nbig = simulation.Nx*simulation.Ny

	if simulation.pol == 'Ez':
		# Defne the starting field for the simulation
		if Estart is None:
			if simulation.fields['Ez'] is None:
				(_, _, Ez) = simulation.solve_fields()
			else:
				Ez = deepcopy(simulation.fields['Ez'])
		else:
			Ez = Estart

		# Solve iteratively
		for istep in range(max_num_iter):
			Eprev = Ez

			(fx, Jac11, Jac12) = nl_eq_and_jac(simulation, Ez=Eprev,
											   matrix_format=matrix_format)

			# Note: Newton's method is defined as a linear problem to avoid inverting the Jacobian
			# Namely, J*(x_n - x_{n-1}) = -f(x_{n-1}), where J = df/dx(x_{n-1})

			Ediff = solver_complex2real(Jac11, Jac12, fx,
										solver=solver, timing=False)
			# Abig = sp.sp_vstack((sp.sp_hstack((Jac11, Jac12)), \
			#   sp.sp_hstack((np.conj(Jac12), np.conj(Jac11)))))
			# Ediff = solver_direct(Abig, np.vstack((fx, np.conj(fx))))

			Ez = Eprev - Ediff[range(Nbig)].reshape(simulation.Nx, simulation.Ny)

			# get convergence and break
			convergence = la.norm(Ez - Eprev)/la.norm(Ez)
			conv_array[istep] = convergence

			# if below threshold, break and return
			if convergence < conv_threshold:
				break

		# Solve the fdfd problem with the final eps_nl
		simulation.compute_nl(Ez)
		(Hx, Hy, Ez) = simulation.solve_fields(include_nl=True)

		if convergence > conv_threshold:
			print("the simulation did not converge, reached {}".format(convergence))

		return (Hx, Hy, Ez, conv_array)

	else:
		raise ValueError('Invalid polarization: {}'.format(str(self.pol)))

def nl_eq_and_jac(simulation,
				  averaging=True, Ex=None, Ey=None, Ez=None, compute_jac=True,
				  matrix_format=DEFAULT_MATRIX_FORMAT):
	# Evaluates the nonlinear function f(E) that defines the problem to solve f(E) = 0, as well as the Jacobian df/dE
	# Could add a check that only Ez is None for Hz polarization and vice-versa

	omega = simulation.omega
	EPSILON_0_ = EPSILON_0*simulation.L0
	MU_0_ = MU_0*simulation.L0

	Nbig = simulation.Nx*simulation.Ny

	if simulation.pol == 'Ez':
		simulation.compute_nl(Ez)
		Anl = simulation.A + simulation.Anl
		fE = (Anl.dot(Ez.reshape(-1,)) - simulation.src.reshape(-1,)*1j*omega)

		# Make it explicitly a column vector
		fE = fE.reshape(-1,)

		if compute_jac:
			simulation.compute_nl(Ez)
			dAde = (simulation.dnl_de).reshape((-1,))*omega**2*EPSILON_0_
			Jac11 = Anl + sp.spdiags(dAde*Ez.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)
			Jac12 = sp.spdiags(np.conj(dAde)*Ez.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)

	elif simulation.pol == 'Hz':
		raise ValueError('angler doesnt support newton method for Hz polarization yet')		

	else:
		raise ValueError('Invalid polarization: {}'.format(str(self.pol)))

	if compute_jac:
		return (fE, Jac11, Jac12)
	else:
		return fE


def newton_krylov_solve(simulation, Estart=None, conv_threshold=1e-10, max_num_iter=50,
				 averaging=True, solver=DEFAULT_SOLVER, jac_solver='c2r',
				 matrix_format=DEFAULT_MATRIX_FORMAT):
	# THIS DOESNT WORK YET! 

	# Stores convergence parameters
	conv_array = np.zeros((max_num_iter, 1))

	# num. columns and rows of A
	Nbig = simulation.Nx*simulation.Ny

	# Defne the starting field for the simulation
	if Estart is None:
		if simulation.fields['Ez'] is None:
			(_, _, E0) = simulation.solve_fields()
		else:
			E0 = deepcopy(simulation.fields['Ez'])
	else:
		E0 = Estart

	E0 = np.reshape(E0, (-1,))

	# funtion for roots
	def _f(E):
		E = np.reshape(E, simulation.eps_r.shape)
		fx = nl_eq_and_jac(simulation, Ez=E, compute_jac=False, matrix_format=matrix_format)
		return np.reshape(fx, (-1,))

	print(_f(E0))

	E_nl = newton_krylov(_f, E0, verbose=True)

	# I'm returning these in place of Hx and Hy to not break things
	return (E_nl, E_nl, E_nl, conv_array)
