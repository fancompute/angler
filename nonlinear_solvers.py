import numpy as np
import scipy.sparse as sp

from FDFD.Fdfd import Fdfd
from FDFD.linalg import solver_direct


def born_solve(simulation, eps_r, b, nonlinear_fn, nl_region, conv_threshold=1e-8, max_num_iter=10):
	# solves for the nonlinear fields using direct substitution / Born approximation / Picard / whatever you want to call it
	# eps_r is the linear permittivity

	# Solve the linear problem to start
	(Hx,Hy,Ez) = simulation.solve_fields(b)

	# Stores convergence parameters
	conv_array = np.zeros((max_num_iter, 1))

	# Solve iteratively
	for istep in range(max_num_iter):

		Eprev = Ez

		# set new permittivity
		eps_nl = eps_r + nonlinear_fn(Eprev)*nl_region

		# get new fields
		simulation.reset_eps(eps_nl)
		(Hx, Hy, Ez) = simulation.solve_fields(b)

		# get convergence and break
		convergence = np.linalg.norm(Ez - Eprev)/np.linalg.norm(Ez)
		conv_array[istep] = convergence

		# if below threshold, break and return
		if convergence < conv_threshold:
			break

	if istep == max_num_iter-1:
		raise Warning("the simulation never converged, reached {}".format(convergence))

	return (Ez, conv_array)


def newton_solve(simulation, eps_r, b, nonlinear_fn, nonlinear_de, nl_region, conv_threshold=1e-18, max_num_iter=5):
	# solves for the nonlinear fields using Newton's method
	# NOTE: DOES NOT WORK YET!
	# Can we break this up into a few functions? -T

	# Solve the linear problem to start
	(Hx,Hy,Ez) = simulation.solve_fields(b)
	Ez = Ez.ravel(order='F')
	nl_region = nl_region.ravel(order='F')
	Al = simulation.A

	# Stores convergence parameters
	conv_array = np.zeros((max_num_iter, 1))

	# num. columns and rows of A
	Nbig = simulation.Nx*simulation.Ny

	# Solve iteratively
	for istep in range(max_num_iter):

		Eprev = Ez

		# set new permittivity
		eps_nl = eps_r + (nonlinear_fn(Eprev)*nl_region).reshape(simulation.Nx, simulation.Ny, order = 'F')

		# reset simulation for matrix A (note: you don't need to solve for the fields!) 
		simulation.reset_eps(eps_nl)

		# perform newtons method to get new fields
		Anl = simulation.A 
		fx = (Anl.dot(Eprev) - b.ravel(order='F')).reshape(Nbig, 1, order = 'F')
		Jac11 = Anl + sp.spdiags((nonlinear_de(Eprev)*Eprev*nl_region), 0, Nbig, Nbig, format='csc')
		Jac12 = sp.spdiags((np.conj(nonlinear_de(Eprev))*Eprev*nl_region), 0, Nbig, Nbig, format='csc')

		n2 = 10;
		# fx1 = (Al + sp.spdiags(n2*Eprev*np.conj(Eprev)*nl_region, 0, Nbig, Nbig, format='csc')).dot(Eprev) - b.ravel(order = 'F')
		# Jac111 = Al + sp.spdiags(2*n2*Eprev*np.conj(Eprev)*nl_region, 0, Nbig, Nbig, format='csc')
		# Jac121 = sp.spdiags(n2*Eprev*Eprev*nl_region, 0, Nbig, Nbig, format='csc')

		# These should be two equivalent ways to make the nonlinear matrix and the norm should be zero but it's not!!
		Anl1 = Al + sp.spdiags(n2*np.square(np.abs(Eprev))*nl_region, 0, Nbig, Nbig, format='csc')
		print(sp.linalg.norm(Anl1 - Anl))	

		# Note: I'm phrasing Newton's method as a linear problem to avoid inverting the Jacobian
		# Namely, J*(x_n - x_{n-1}) = -f(x_{n-1}), where J = df/dx(x_{n-1})
		fx_full = np.vstack((fx, np.conj(fx)))
		Jac_full = sp.vstack((sp.hstack((Jac11, Jac12)), np.conj(sp.hstack((Jac12, Jac11)))))
		Ediff = solver_direct(Jac_full, fx_full)
		# Ediff = solver_direct(Jac11, fx)
		Ez = Eprev - Ediff[range(Nbig)]

		# get convergence and break
		convergence = np.linalg.norm(Ez - Eprev)/np.linalg.norm(Ez)
		conv_array[istep] = convergence

		# if below threshold, break and return
		if convergence < conv_threshold:
			break

	Ez = Ez.reshape(simulation.Nx, simulation.Ny, order = 'F')


	if istep == max_num_iter-1:
		raise Warning("the simulation never converged, reached {}".format(convergence))
		
	return (Ez, conv_array)
