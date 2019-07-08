import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

try:
    from pyMKL import pardisoSolver
    SOLVER = 'pardiso'
except:
    SOLVER = 'scipy'

from time import time

from angler.constants import DEFAULT_MATRIX_FORMAT, DEFAULT_SOLVER
from angler.constants import EPSILON_0, MU_0
from angler.pml import S_create
from angler.derivatives import createDws


def grid_average(center_array, w):
    # computes values at cell edges

    xy = {'x': 0, 'y': 1}
    center_shifted = np.roll(center_array, 1, axis=xy[w])
    avg_array = (center_shifted+center_array)/2
    return avg_array


def dL(N, xrange, yrange=None):
    # solves for the grid spacing

    if yrange is None:
        L = np.array([np.diff(xrange)[0]])  # Simulation domain lengths
    else:
        L = np.array([np.diff(xrange)[0],
                      np.diff(yrange)[0]])  # Simulation domain lengths
    return L/N


def is_equal(matrix1, matrix2):
    # checks if two sparse matrices are equal

    return (matrix1 != matrix2).nnz == 0


def construct_A(omega, xrange, yrange, eps_r, NPML, pol, L0,
                averaging=True,
                use_dirichlet_bcs=False,
                timing=False,
                matrix_format=DEFAULT_MATRIX_FORMAT):
    # makes the A matrix
    N = np.asarray(eps_r.shape)  # Number of mesh cells
    M = np.prod(N)  # Number of unknowns

    EPSILON_0_ = EPSILON_0*L0
    MU_0_ = MU_0*L0

    if pol == 'Ez':
        vector_eps_z = EPSILON_0_*eps_r.reshape((-1,))
        T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format=matrix_format)

        (Sxf, Sxb, Syf, Syb) = S_create(omega, L0, N, NPML, xrange, yrange, matrix_format=matrix_format)

        # Construct derivate matrices
        Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format, use_dirichlet_bcs=use_dirichlet_bcs))
        Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format, use_dirichlet_bcs=use_dirichlet_bcs))
        Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format, use_dirichlet_bcs=use_dirichlet_bcs))
        Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format, use_dirichlet_bcs=use_dirichlet_bcs))

        A = (Dxf*1/MU_0_).dot(Dxb) \
            + (Dyf*1/MU_0_).dot(Dyb) \
            + omega**2*T_eps_z

    elif pol == 'Hz':
        if averaging:
            vector_eps_x = grid_average(EPSILON_0_*eps_r, 'x').reshape((-1,))
            vector_eps_y = grid_average(EPSILON_0_*eps_r, 'y').reshape((-1,))
        else:
            vector_eps_x = EPSILON_0_*eps_r.reshape((-1,))
            vector_eps_y = EPSILON_0_*eps_r.reshape((-1,))

        # Setup the T_eps_x, T_eps_y, T_eps_x_inv, and T_eps_y_inv matrices
        T_eps_x = sp.spdiags(vector_eps_x, 0, M, M, format=matrix_format)
        T_eps_y = sp.spdiags(vector_eps_y, 0, M, M, format=matrix_format)
        T_eps_x_inv = sp.spdiags(1/vector_eps_x, 0, M, M, format=matrix_format)
        T_eps_y_inv = sp.spdiags(1/vector_eps_y, 0, M, M, format=matrix_format)

        (Sxf, Sxb, Syf, Syb) = S_create(omega, L0, N, NPML, xrange, yrange, matrix_format=matrix_format)

        # Construct derivate matrices
        Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format, use_dirichlet_bcs=use_dirichlet_bcs))
        Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format, use_dirichlet_bcs=use_dirichlet_bcs))
        Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format, use_dirichlet_bcs=use_dirichlet_bcs))
        Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format, use_dirichlet_bcs=use_dirichlet_bcs))

        A =   Dxf.dot(T_eps_x_inv).dot(Dxb) \
            + Dyf.dot(T_eps_y_inv).dot(Dyb) \
            + omega**2*MU_0_*sp.eye(M)

    else:
        raise ValueError("something went wrong and pol is not one of Ez, Hz, instead was given {}".format(pol))

    derivs = {
        'Dyb' : Dyb,
        'Dxb' : Dxb,
        'Dxf' : Dxf,
        'Dyf' : Dyf
    }

    return (A, derivs)


def solver_eigs(A, Neigs, guess_value=0, guess_vector=None, timing=False):
    # solves for the eigenmodes of A

    if timing:
        start = time()
    (values, vectors) = spl.eigs(A, k=Neigs, sigma=guess_value, v0=guess_vector, which='LM')
    if timing:
        end = time()
        print('Elapsed time for eigs() is %.4f secs' % (end - start))
    return (values, vectors)


def solver_direct(A, b, timing=False, solver=SOLVER):
    # solves linear system of equations

    b = b.astype(np.complex128)
    b = b.reshape((-1,))

    if not b.any():
        return np.zeros(b.shape)

    if timing:
        t = time()

    if solver.lower() == 'pardiso':
        pSolve = pardisoSolver(A, mtype=13) # Matrix is complex unsymmetric due to SC-PML
        pSolve.factor()
        x = pSolve.solve(b)
        pSolve.clear()

    elif solver.lower() == 'scipy':
        x = spl.spsolve(A, b)

    else:
        raise ValueError('Invalid solver choice: {}, options are pardiso or scipy'.format(str(solver)))

    if timing:
        print('Linear system solve took {:.2f} seconds'.format(time()-t))

    return x


def solver_complex2real(A11, A12, b, timing=False, solver=SOLVER):
    # solves linear system of equations [A11, A12; A21*, A22*]*[x; x*] = [b; b*]

    b = b.astype(np.complex128)
    b = b.reshape((-1,))
    N = b.size

    if not b.any():
        return np.zeros(b.shape)

    b_re = np.real(b).astype(np.float64)
    b_im = np.imag(b).astype(np.float64)

    Areal = sp.vstack((sp.hstack((np.real(A11) + np.real(A12), - np.imag(A11) + np.imag(A12))),
                       sp.hstack((np.imag(A11) + np.imag(A12), np.real(A11) - np.real(A12)))))

    if timing:
        t = time()

    if solver.lower() == 'pardiso':
        pSolve = pardisoSolver(Areal, mtype=11)  # Matrix is real unsymmetric
        pSolve.factor()
        x = pSolve.solve(np.hstack((b_re, b_im)))
        pSolve.clear()

    elif solver.lower() == 'scipy':
        x = spsolve(Areal, np.hstack((b_re, b_im)))

    else:
        raise ValueError('Invalid solver choice: {}, options are pardiso or scipy'.format(str(solver)))

    if timing:
        print('Linear system solve took {:.2f} seconds'.format(time()-t))

    return (x[:N] + 1j*x[N:2*N])
