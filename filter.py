import matplotlib.pylab as plt
import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp

from autograd import grad
from functools import partial

# eps = np.load('data/figs/data/2port_eps.npy')


def dist(r1, r2):
    return np.sqrt(np.sum(np.square(r1 - r2)))

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def get_W(Nx, Ny, design_region, R=10):

    diffs = range(-R//2, R//2+1)
    N = Nx*Ny

    row_indeces = []
    col_indeces = []
    vals = []

    for i1 in range(Nx):
        for j1 in range(Ny):
            r1 = np.array([i1, j1])

            row_index = sub2ind((Nx, Ny), i1, j1)

            if i1 <= R//2 or i1 >= Nx-R//2-1:
                pass
                row_indeces.append(row_index)
                col_indeces.append(row_index)
                vals.append(R)
            elif j1 <= R//2 or j1 >= Ny-R//2-1:
                pass
                # row_indeces.append(row_index)
                # col_indeces.append(row_index)
                # vals.append(R)
            else:
                for i_diff in diffs:
                    i2 = i1 + i_diff
                    for j_diff in diffs:
                        j2 = j1 + j_diff
                        r2 = np.array([i2, j2])
                        col_index = sub2ind((Nx, Ny), i2, j2)

                        val = R - dist(r1, r2)

                        if val > 0:
                            row_indeces.append(row_index)
                            col_indeces.append(col_index)
                            vals.append(val)

    W = sp.csr_matrix((vals, (row_indeces, col_indeces)), shape=(N, N))

    des_vec = design_region.reshape((-1,))
    no_des_vec = 1-des_vec
    des_mat = sp.diags(des_vec, shape=(N, N))
    no_des_mat = sp.diags(no_des_vec, shape=(N, N))

    W_des = W.dot(des_mat) + no_des_mat

    norm_vec = W.dot(np.ones((Nx*Ny,)))
    norm_vec[norm_vec == 0] = 1
    norm_mat = sp.diags(1/norm_vec, shape=(N, N))

    W = W.dot(norm_mat)

    return des_mat.dot(W) + no_des_mat


""" THESE ARE THE OPERATIONS GOING FROM A DENSITY TO A PERMITTIVITY
    THROUGH FILTERING AND PROJECTING """


def rho2rhot(rho, W):
    # density to filtered density
    (Nx, Ny) = rho.shape
    rho_vec = rho.reshape((-1,))
    rhot_vec = W.dot(rho_vec)
    rhot = np.reshape(rhot_vec, (Nx, Ny))
    return rhot


def rhot2rhob(rhot, eta=0.5, beta=100):
    # filtered density to projected density
    num = np.tanh(beta*eta) + np.tanh(beta*(rhot - eta))
    den = np.tanh(beta*eta) + np.tanh(beta*(1 - eta))
    return num / den


def rhob2eps(rhob, eps_m):
    # filtered density to permittivity
    return 1 + rhob * (eps_m - 1)


def eps2rho(eps, eps_m):
    # permittivity to density (ONLY USED FOR STARTING THE SIMULATION!)
    return (eps - 1) / (eps_m - 1)


def rho2eps(rho, eps_m, W, eta=0.5, beta=100):
    rhot = rho2rhot(rho, W)
    rhob = rhot2rhob(rhot, eta=eta, beta=beta)
    eps = rhob2eps(rhob, eps_m=eps_m)
    return eps


"""" DERIVATIVE OPERATORS """


def drhot_drho(W,):
    # derivative of filtered density with respect to design density
    return W


def drhob_drhot(rho_t, eta=0.5, beta=100):
    # change in projected density with respect to filtered density
    rhot_vec = np.reshape(rho_t, (-1,))
    num = beta - beta*np.square(np.tanh(beta*(rhot_vec - eta)))
    den = np.tanh(beta*eta) + np.tanh(beta*(1 - eta))
    return num / den

def deps_drhob(rhob, eps_m):
    return (eps_m - 1)


""" THIS IS THE OPPOSITE PROCEDURE, TAKING PERMITTIVITY AND GETTING DENSITY """


if __name__ == '__main__':

    Nx = 100
    Ny = 100
    R = 5
    eps_m = 6
    beta = 100
    eta = 0.5

    rho = np.zeros((Nx, Ny))
    rho[20:Nx-30, Ny//2:Ny//2+30] = 0.7
    rho[Nx//2:Nx//2+30, 19:Ny-20] = 1

    design_region = np.zeros((Nx, Ny))
    design_region[20:80, 20:80] = 1

    W = get_W(Nx, Ny, design_region, R=R)

    rhot = rho2rhot(rho, W)
    rhob = rhot2rhob(rhot, eta=eta, beta=beta)
    eps = rhob2eps(rhob, eps_m=eps_m)


    RTs = np.linspace(0,1,1000)
    plt.plot(RTs, drhob_drhot(RTs, eta=0.5, beta=10))
    plt.plot(RTs, rhot2rhob(RTs, eta=0.5, beta=10))
    plt.show()

    # change in projected density with respect to filtered density


    # f1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
    # im1 = ax1.imshow(rho)
    # plt.colorbar(im1, ax=ax1)
    # im2 = ax2.imshow(rhot)
    # plt.colorbar(im2, ax=ax2)
    # im3 = ax3.imshow(rhob)
    # plt.colorbar(im3, ax=ax3)
    # im4 = ax4.imshow(eps)
    # plt.colorbar(im4, ax=ax4)

    # eps_r = rho2eps(rho, eps_m, W, eta=0.5, beta=100)

    # im5 = ax5.imshow(eps_r)
    # plt.colorbar(im5, ax=ax5)
    # plt.show()

    # R = np.random.random((100, 100))
    # # R = np.zeros((Nx, Ny))
    # # R[Nx//2:Nx//2+5, Ny//2:Ny//2+5] = 1
    # eps = rho2eps(R, eps_m=eps_m, W=W, eta=0.5, beta=100)
    # plt.imshow(eps)
    # plt.show()

    # rhob = eps2rhob(eps, eps_m=eps_m)
    # rhot = rhob2rhot(rhob, eta=eta, beta=beta)

    # f1, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    # im1 = ax1.imshow(eps)
    # plt.colorbar(im1, ax=ax1)
    # im2 = ax2.imshow(rhob)
    # plt.colorbar(im2, ax=ax2)
    # im3 = ax3.imshow(rhot)
    # plt.colorbar(im3, ax=ax3)
    # # im4 = ax4.imshow(rho)
    # # plt.colorbar(im4, ax=ax4)
    # plt.show()    
