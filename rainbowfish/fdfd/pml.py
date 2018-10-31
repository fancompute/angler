import numpy as np
import scipy.sparse as sp

from fdfdpy.constants import ETA_0, EPSILON_0, DEFAULT_MATRIX_FORMAT


def sig_w(l, dw, m=4, lnR=-12):
    # helper for S()

    sig_max = -(m+1)*lnR/(2*ETA_0*dw)
    return sig_max*(l/dw)**m


def S(l, dw, omega, L0):
    # helper for create_sfactor()

    return 1 - 1j*sig_w(l, dw)/(omega*EPSILON_0*L0)


def create_sfactor(wrange, L0, s, omega, Nw, Nw_pml):
    # used to help construct the S matrices for the PML creation

    sfactor_array = np.ones(Nw, dtype=np.complex128)
    if Nw_pml < 1:
        return sfactor_array
    hw = np.diff(wrange)[0]/Nw
    dw = Nw_pml*hw
    for i in range(0, Nw):
        if s is 'f':
            if i <= Nw_pml:
                sfactor_array[i] = S(hw * (Nw_pml - i + 0.5), dw, omega, L0)
            elif i > Nw - Nw_pml:
                sfactor_array[i] = S(hw * (i - (Nw - Nw_pml) - 0.5), dw, omega, L0)
        if s is 'b':
            if i <= Nw_pml:
                sfactor_array[i] = S(hw * (Nw_pml - i + 1), dw, omega, L0)
            elif i > Nw - Nw_pml:
                sfactor_array[i] = S(hw * (i - (Nw - Nw_pml) - 1), dw, omega, L0)
    return sfactor_array


def S_create(omega, L0, N, Npml, xrange,
             yrange=None, matrix_format=DEFAULT_MATRIX_FORMAT):
    # creates S matrices for the PML creation

    M = np.prod(N)
    if np.isscalar(Npml):
        Npml = np.array([Npml])
    if len(N) < 2:
        N = np.append(N, 1)
        Npml = np.append(Npml, 0)
    Nx = N[0]
    Nx_pml = Npml[0]
    Ny = N[1]
    Ny_pml = Npml[1]

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor(xrange, L0, 'f', omega, Nx, Nx_pml)
    s_vector_x_b = create_sfactor(xrange, L0, 'b', omega, Nx, Nx_pml)
    s_vector_y_f = create_sfactor(yrange, L0, 'f', omega, Ny, Ny_pml)
    s_vector_y_b = create_sfactor(yrange, L0, 'b', omega, Ny, Ny_pml)

    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_2D = np.zeros(N, dtype=np.complex128)
    Sx_b_2D = np.zeros(N, dtype=np.complex128)
    Sy_f_2D = np.zeros(N, dtype=np.complex128)
    Sy_b_2D = np.zeros(N, dtype=np.complex128)

    for i in range(0, Ny):
        Sx_f_2D[:, i] = 1/s_vector_x_f
        Sx_b_2D[:, i] = 1/s_vector_x_b

    for i in range(0, Nx):
        Sy_f_2D[i, :] = 1/s_vector_y_f
        Sy_b_2D[i, :] = 1/s_vector_y_b

    # Reshape the 2D s-factors into a 1D s-array
    Sx_f_vec = Sx_f_2D.reshape((-1,))
    Sx_b_vec = Sx_b_2D.reshape((-1,))
    Sy_f_vec = Sy_f_2D.reshape((-1,))
    Sy_b_vec = Sy_b_2D.reshape((-1,))

    # Construct the 1D total s-array into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f_vec, 0, M, M, format=matrix_format)
    Sx_b = sp.spdiags(Sx_b_vec, 0, M, M, format=matrix_format)
    Sy_f = sp.spdiags(Sy_f_vec, 0, M, M, format=matrix_format)
    Sy_b = sp.spdiags(Sy_b_vec, 0, M, M, format=matrix_format)

    return (Sx_f, Sx_b, Sy_f, Sy_b)
