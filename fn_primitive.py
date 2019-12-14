import autograd.numpy as np

import autograd as ag

def sp_solve_nl(parameters, a_indices, b, fn_nl):
	""" 
		a_entries: entries into sparse A matrix
		a_indices: indices into sparse A matrix
		b: source vector for Ax = b
		fn_nl: describes how the entries of a depend on the solution of Ax = b and the parameters  `a_entries = fn_nl(params, x)`
	"""

	# do the actual nonlinear solve in `_solve_nl_problem` (using newton, picard, whatever)
	# this tells you the final entries into A given the parameters and the nonlinear function.
	a_entries = _solve_nl_problem(parameters, a_indices, fn_nl, a_entries0=None)  # optinally, give starting a_entries
	x = sp_solve(a_entries, a_indices, b)  # the final solution to A(x) x = b
	return x

def grad_sp_solve_nl_parameters(x, parameters, a_indices, b, fn_nl):

	""" 
	We are finding the solution (x) to the nonlinear function:

	    f = A(x, p) @ x - b = 0

	And need to define the vjp of the solution (x) with respect to the parameters (p)

		vjp(v) = (dx / dp)^T @ v

	To do this (see Eq. 5 of https://pubs-acs-org.stanford.idm.oclc.org/doi/pdf/10.1021/acsphotonics.8b01522)
	we need to solve the following linear system:

		[ df  / dx,  df  / dx*] [ dx  / dp ] = -[ df  / dp ]
		[ df* / dx,  df* / dx*] [ dx* / dp ]    [ df* / dp]
	
	In our case:

		df / dx = (dA / dx) @ x + A
		df / dp = (dA / dp) @ x

	How do we put this into code?  Let

		A(x, p) @ x -> b = sp_mult(entries_a(x, p), indices_a, x)

	Since we already defined the primitive of sp_mult, we can just do:

		(dA / dx) @ x -> ag.grad(b, 0)

	Now how about the source term?

		(dA / dp) @ x -> ag.grad(b, 1)

	Assuminging entries_a(x, p) is fully autograd compatible, we can get these terms no problem!

	So then we have to solve our big system

		A_big @ x_big = b_big
		x_big = sp_spolve(entries_a_big, indices_a_big, b_big)

	And then strip out the non complex-conjugated part of the solution

		x_nl = x_big[:N]
	"""

	def vjp(v):
		# do the nonlinear gradient stuff here.
	return vjp

ag.extend.defvjp(obj2, grad_obj2, None, None)


@ag.primitive
def obj2(x0, fn, y):
	x = x0
	for _ in range(1):
		x = fn(x) * y
	return x

def grad_obj2(ans, x0, fn, y):
	def vjp(v):
		# return v * 4 * y**2 * (x0**3 + x0)
		return v * 2 * x0 * y
	return vjp

ag.extend.defvjp(obj2, grad_obj2, None, None)

if __name__ == '__main__':

	x0 = 3.0
	y = .3

	def nl_fn(x):
		return np.square(x) + 1

	x1 = obj1(x0, nl_fn, y)
	x2 = obj2(x0, nl_fn, y)

	print(x1)
	print(x2)
	print(ag.grad(obj1)(x0, nl_fn, y))
	print(ag.grad(obj2)(x0, nl_fn, y))