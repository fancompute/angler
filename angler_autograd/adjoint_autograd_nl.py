import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp

from scipy.sparse.linalg import spsolve
from autograd.extend import primitive, defvjp
from autograd.test_util import check_grads
from autograd import grad, jacobian, elementwise_grad
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

def make_adjoint_system():
    pass

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
def solve_fields_nl(eps_r, f_nl, N_max_iter=100, error=1e-4):
    E_prev = np.zeros(eps_r.shape)
    # picard iterations to solve nonlinear fn
    for i in range(N_max_iter):
        eps_nl = eps_r + f_nl(E_prev, np.conj(E_prev))
        A = make_A(eps_nl)
        E = spsolve(A, b)
        res = np.linalg.norm(E - E_prev) / np.linalg.norm(E)
        if res < error:
            return E.reshape(eps_r.shape)
        E_prev = E

    raise ValueError("didnt converge")

# this function returns the gradient as a function of the adjoint source (v = dJdE)
def vjp_maker(E_nl, eps_r, f_nl):

    # wrap the vector-jacobian product
    def vjp(v):

        E_nl_star = np.conj(E_nl)

        # get the nonlinear permittivity
        eps_nl = eps_r + f_nl(E_nl, E_nl_star)        

        # get the nonlinear system matrix        
        A = make_A(eps_nl)

        # get the derivatives of the nonlinear function
        dfnl_de = elementwise_grad(f_nl, 0)(E_nl, E_nl_star)
        dfnl_star_de = np.conj(elementwise_grad(f_nl, 1)(E_nl, E_nl_star))

        # setup the adjoint system for nonlinear problem
        N = E_nl.size
        df_de = A + make_sparse_diag(dfnl_de * E_nl, N)
        df_star_de = make_sparse_diag(dfnl_star_de * E_nl_star, N)

        A_big_top = sp.hstack((df_de.T,      df_star_de.T), format='csr')
        A_big_bot = sp.hstack((df_star_de.H, df_de.H), format='csr')

        A_big = sp.vstack((A_big_top, A_big_bot), format='csr')
        v_big = np.vstack((-v[:, None], np.conj(-v[:, None])))        

        # solve the adjoint problem
        adjoint = spsolve(A_big, v_big)
        adjoint = adjoint[:N]

        # adjoint kernel says how to compute gradient using both E and E_adj
        return adjoint_kernel(adjoint, E_nl)

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
    def f_nl(E, E_star, chi3=5):
        return npa.abs(chi3 * E * E_star)

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
