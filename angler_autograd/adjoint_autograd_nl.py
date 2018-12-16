import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve
from autograd.extend import primitive, defvjp
from autograd.test_util import check_grads
from autograd import grad, jacobian
from copy import deepcopy

DTYPE = np.complex64

def make_random_sparse(N, density=0.01):
    # make a random, complex, sparse matrix of size (N, N)
    return sp.random(N, N, density=0.01) + 1j*sp.random(N, N, density=0.01)

def make_sparse_diag(eps_r, N):
    # make a diagonal, complex matrix of size (N, N) using `eps_r` \in (N,) as diagonal
    return sp.diags(diagonals=eps_r, offsets=0, shape=(N, N), dtype=DTYPE)

def make_random_complex(N, amplitude=1):
    # make a random complex vector of size (N, 1)
    real_part = amplitude * (np.random.rand(N, 1) - 0.5)
    imag_part = amplitude * (np.random.rand(N, 1) - 0.5)
    return real_part + 1j * imag_part

def make_A(eps_r):
    # constructs the Maxwell operator, or 'system matrix' A(eps)
    N = eps_r.size
    eps_diag = make_sparse_diag(eps_r, N)
    A = A0 + eps_diag
    return A

def adjoint_kernel(adjoint, E):
    # returns gradient based on forward field and adjoint field
    return np.real(adjoint * E)

def numerical_grad(fn, eps_r, d_eps=1e-4):
    # simple function to check brute force gradient (autograd.test_util.check_grads not working for me...)
    N = eps_r.size
    num_grad = np.zeros((N,))
    fn_old = fn(eps_r)
    for i in range(N):
        eps_new = deepcopy(eps_r)
        eps_new[i] += d_eps
        fn_new = fn(eps_new)
        num_grad[i] = (fn_new - fn_old) / d_eps
    return num_grad

# define function for solving electric fields as a function of permittivity
@primitive
def solve_fields_nl(eps_r, f_nl, N_max_iter=10, error=1e-8):
    E = np.zeros(eps_r.shape)
    # picard iterations to solve nonlinear fn
    for i in range(N_max_iter):
        eps_nl = eps_r + f_nl(E)
        A = make_A(eps_nl)
        E = spsolve(A, b)
        res = np.linalg.norm(A.dot(E) - b[:,0]) / np.linalg.norm(b)
        if res < error:
            return E.reshape(eps_r.shape)
    raise ValueError("didnt converge")

# this function returns the gradient as a function of the adjoint source (v = dJdE)
def vjp_maker(E, eps_r, f_nl):
    # wrap the vector-jacobian product
    def vjp(v):
        # get the system matrix again
        A = make_A(eps_r)
        # solve the adjoint problem
        adjoint = spsolve(A.T, -v)
        # adjoint kernel says how to compute gradient using both E and E_adj
        return adjoint_kernel(adjoint, E)
    # return this function for autograd
    return vjp

# 'link' solve_fields_nl with its vjp_maker function, which will let us call autograd.grad() on it
defvjp(solve_fields_nl, vjp_maker)

if __name__ == '__main__':

    N = 10                           # number of grid points
    eps_r = 1 + np.random.rand(N,)   # relative permittivity
    b = make_random_complex(N)       # source vector
    A0 = make_random_sparse(N)       # system matrix without permittivity

    # A0 and b are just global variables right now for testing 
    # but eventually we will want to pass b and other simulation parameters
    # as arguments into the above functions

    # the nonlinear relative permittivity a fn of electric field
    def f_nl(E, chi3=1j):
        return chi3 * np.square(np.abs(E))

    # the partial objective function as a fn of E-fields
    def J(E):
        return npa.sum(npa.square(npa.abs(E)))/ N

    # the full objective function as a fn of eps_r
    def objective(eps_r):
        E = solve_fields_nl(eps_r, f_nl)
        return J(E)

    # use autograd to construct gradient
    gradient = grad(objective)

    # solve the fields
    E = solve_fields_nl(eps_r, f_nl)

    # compute adjoint gradient
    grad_analytical = gradient(eps_r)
    print('analytical gradient of {}'.format(grad_analytical))

    # compute numerical gradient
    grad_num = numerical_grad(objective, eps_r, d_eps=1e-3)
    print('numerical gradient of {}'.format(grad_num))
