import numpy as np
import scipy.sparse as sp
import autograd.numpy as npa
from autograd import grad
from functools import partial

from angler.linalg import solver_direct, grid_average
from angler.derivatives import unpack_derivs
from angler.filter import deps_drhob, drhob_drhot
from angler.constants import *

def gradient(simulation, dJ, design_region, arguments, eps_m, W):
    """Gives full derivative of J with respect to eps"""
    (Ez, Ez_nl) = arguments
    return grad_linear(simulation, dJ, design_region, arguments, eps_m=eps_m, W=W) + \
           grad_nonlinear(simulation, dJ, design_region, arguments, eps_m=eps_m, W=W)

def grad_linear(simulation, dJ, design_region, arguments, eps_m, W):
    """gives the linear field gradient: partial J/ partial * E_lin dE_lin / deps"""
    (Ez, Ez_nl) = arguments

    b_aj = -dJ['lin'](Ez, Ez_nl)
    Ez_aj = adjoint_linear(simulation, b_aj)

    EPSILON_0_ = EPSILON_0*simulation.L0
    omega = simulation.omega
    dAdeps = design_region*omega**2*EPSILON_0_

    return 1*np.real(Ez_aj * dAdeps * Ez)


def grad_nonlinear(simulation, dJ, design_region, arguments, eps_m, W):
    """gives the linear field gradient: partial J/ partial * E_lin dE_lin / deps"""
    (Ez, Ez_nl, rho) = arguments

    rho_bar = rho2rho_bar(rho)
    eps = rho_bar2eps(rho_bar, eps_m)

    b_aj = -dJ['nl'](Ez, Ez_nl, eps)
    Ez_aj = adjoint_nonlinear(simulation, b_aj)

    EPSILON_0_ = EPSILON_0*simulation.L0
    omega = simulation.omega    
    dAdeps = design_region*omega**2*EPSILON_0_
    dAnldeps = dAdeps + design_region*omega**2*EPSILON_0_*simulation.dnl_deps

    return 1*np.real(Ez_aj * dAnldeps * Ez_nl)


def adjoint_linear(simulation, b_aj, averaging=False, solver=DEFAULT_SOLVER, matrix_format=DEFAULT_MATRIX_FORMAT):
    # Compute the adjoint field for a linear problem
    # Note: the correct definition requires simulating with the transpose matrix A.T
    EPSILON_0_ = EPSILON_0*simulation.L0
    MU_0_ = MU_0*simulation.L0
    omega = simulation.omega

    (Nx, Ny) = (simulation.Nx, simulation.Ny)
    M = Nx*Ny
    A = simulation.A

    ez = solver_direct(A.T, b_aj, solver=solver)
    Ez = ez.reshape((Nx, Ny))

    return Ez


def adjoint_nonlinear(simulation, b_aj,
                     averaging=False, solver=DEFAULT_SOLVER, matrix_format=DEFAULT_MATRIX_FORMAT):
    # Compute the adjoint field for a nonlinear problem
    # Note: written only for Ez!

    EPSILON_0_ = EPSILON_0*simulation.L0
    MU_0_ = MU_0*simulation.L0
    omega = simulation.omega

    (Nx, Ny) = (simulation.Nx, simulation.Ny)
    M = Nx*Ny

    Ez = simulation.fields_nl['Ez']
    Anl = simulation.A + simulation.Anl
    dAde = omega**2*EPSILON_0_*simulation.dnl_de

    C11 = Anl + sp.spdiags((dAde*Ez).reshape((-1,)), 0, M, M, format=matrix_format)
    C12 = sp.spdiags((np.conj(dAde)*Ez).reshape((-1)), 0, M, M, format=matrix_format)
    C_full = sp.vstack((sp.hstack((C11, C12)), np.conj(sp.hstack((C12, C11)))))
    b_aj = b_aj.reshape((-1,))

    ez = solver_direct(C_full.T, np.vstack((b_aj, np.conj(b_aj))), solver=solver)

    if np.linalg.norm(ez[range(M)] - np.conj(ez[range(M, 2*M)])) > 1e-8:
        print('Adjoint field and conjugate do not match; something might be wrong')

    Ez = ez[range(M)].reshape((Nx, Ny))

    return Ez
