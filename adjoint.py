import numpy as np
import scipy.sparse as sp
from fdfdpy.linalg import solver_direct, grid_average
from fdfdpy.derivatives import unpack_derivs
from fdfdpy.constants import *
import autograd.numpy as npa
from autograd import grad
from functools import partial

def eps2rho_bar(eps, eps_m):
    return (eps - 1) / (eps_m - 1)

def rho_bar2eps(rho_bar, eps_m):
    return rho_bar * (eps_m - 1) + 1

def deps_drho_bar(rho_bar, eps_m):
    return (eps_m - 1)

def rho2rho_bar(rho, eta=0.5, beta=10):
    num = npa.tanh(beta*eta) + npa.tanh(beta*(rho - eta))
    den = npa.tanh(beta*eta) + npa.tanh(beta*(1 - eta))
    return num / den

def drho_bar_drho(rho, eta=0.5, beta=10):
    rho_bar = partial(rho2rho_bar, eta=eta, beta=beta)
    grad_rho_bar = grad(rho2rho_bar)
    grad_rho_bar = np.vectorize(grad_rho_bar)
    return grad_rho_bar(rho)

def gradient(simulation, dJ, design_region, arguments, eps_m):
    """Gives full derivative of J with respect to eps"""
    (Ez, Ez_nl, rho) = arguments
    rho_bar = rho2rho_bar(rho)
    eps = rho_bar2eps(rho_bar, eps_m)
    return dJ['eps'](Ez, Ez_nl, eps) + \
           grad_linear(simulation, dJ, design_region, arguments, eps_m=eps_m) + \
           grad_nonlinear(simulation, dJ, design_region, arguments, eps_m=eps_m)

def grad_linear(simulation, dJ, design_region, arguments, eps_m):
    """gives the linear field gradient: partial J/ partial * E_lin dE_lin / deps"""
    (Ez, Ez_nl, rho) = arguments

    rho_bar = rho2rho_bar(rho)
    eps = rho_bar2eps(rho_bar, eps_m)

    b_aj = -dJ['lin'](Ez, Ez_nl, eps)
    Ez_aj = adjoint_linear(simulation, b_aj)

    EPSILON_0_ = EPSILON_0*simulation.L0
    omega = simulation.omega
    dAdeps = design_region*omega**2*EPSILON_0_

    dedrb = deps_drho_bar(rho_bar, eps_m)
    drbdr = drho_bar_drho(rho)

    projected_Ez = dedrb * drbdr * Ez

    return 1*np.real(Ez_aj * dAdeps * projected_Ez)


def grad_nonlinear(simulation, dJ, design_region, arguments, eps_m):
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


    dedrb = deps_drho_bar(rho_bar, eps_m)
    drbdr = drho_bar_drho(rho)

    projected_Ez_nl = dedrb * drbdr * Ez_nl

    return 1*np.real(Ez_aj*dAnldeps*projected_Ez_nl)


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

    (Nx,Ny) = (simulation.Nx, simulation.Ny)
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
