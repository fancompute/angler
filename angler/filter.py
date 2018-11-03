import matplotlib.pylab as plt
import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp
import progressbar

from autograd import grad
from functools import partial

def wrap(i, N):
    return i + (i >= N)*(i - N) + (i < 0)*(i + N)

def dist(r1, r2):
    return np.sqrt(np.sum(np.square(r1 - r2)))

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def get_W(Nx, Ny, design_region, NPML, R=10):

    diffs = range(-R, R+1)
    N = Nx*Ny

    # if there is no PML on one or more sides, need to wrap filter around
    is_periodic_x = NPML[0] == 0
    is_periodic_y = NPML[1] == 0

    row_indeces = []
    col_indeces = []
    vals = []

    if is_periodic_x:
        i_range = range(Nx)
    else:
        i_range = range(R + NPML[0] + 1, Nx - R - 1 - NPML[0])

    if is_periodic_y:
        j_range = range(Ny)
    else:
        j_range = range(R + NPML[1] + 1, Ny - R - 1 - NPML[1])

    bar = progressbar.ProgressBar(max_value=len(i_range)+1)

    for counti, i1 in enumerate(i_range):
        bar.update(counti)        
        for countj, j1 in enumerate(j_range):

            r1 = np.array([i1, j1])

            row_index = sub2ind((Nx, Ny), i1, j1)

            for i_diff in diffs:
                i2 = i1 + i_diff
                for j_diff in diffs:
                    j2 = j1 + j_diff
                    r2 = np.array([i2, j2])

                    # wrap around for periodic BC
                    i2 = wrap(i2, Nx)
                    j2 = wrap(j2, Ny)

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


def drhot_drho(W):
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



if __name__ == '__main__':

    ## NOTE: Running this file directly will generate some filtering plots used for testing
    # and as some figures in the supplementary information.

    """ some isolated tests of the filtering and projection """
    import matplotlib.style
    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mpl.style.use('seaborn-colorblind')    

    Nx = 100
    Ny = 100
    R = 5
    eps_m = 6
    beta = 100
    eta = 0.5
    NPML = [5, 5]

    rho = np.zeros((Nx, Ny))
    rho[20:Nx-20, Ny//2:Ny//2+20] = 0.7
    rho[Nx//2:Nx//2+5, 20:Ny-20] = 1
    # plt.imshow(rho)
    # plt.show()

    design_region = np.ones((Nx, Ny))

    W = get_W(Nx, Ny, design_region, NPML, R=R)

    rhot = rho2rhot(rho, W)
    rhob = rhot2rhob(rhot, eta=eta, beta=beta)
    eps = rhob2eps(rhob, eps_m=eps_m)

    f = plt.figure(figsize=(8, 4))
    RTs = np.linspace(0,1,1000)
    plt.plot(RTs, rhot2rhob(RTs, eta=0.5, beta=0.1))
    plt.plot(RTs, rhot2rhob(RTs, eta=0.5, beta=10))
    plt.plot(RTs, rhot2rhob(RTs, eta=0.5, beta=1000))
    plt.legend((r"$\beta = .1$", r"$\beta = 10$", r"$\beta = 1000$"))
    plt.xlabel(r"$\tilde{\rho}_i$")
    plt.ylabel(r"$\bar{\rho}_i$")
    plt.title(r"projection with varying $\beta$ ($\eta=0.5$)")
    # plt.savefig('data/figs/img/project.pdf', dpi=300)
    plt.show()

    # # change in projected density with respect to filtered density


    def circle(Nx, Ny, R=10):

        diffs = range(-R, R+1)
        N = Nx*Ny

        circ = np.zeros((Nx, Ny))

        for i1 in range(Nx//2, Nx//2+1):
            for j1 in range(Ny//2, Ny//2+1):
                r1 = np.array([i1, j1])

                if i1 <= R or i1 >= Nx-R-1:
                    pass
                elif j1 <= R or j1 >= Ny-R-1:
                    pass
                else:
                    for i_diff in diffs:
                        i2 = i1 + i_diff
                        for j_diff in diffs:
                            j2 = j1 + j_diff
                            r2 = np.array([i2, j2])

                            val = R - dist(r1, r2)

                            if val > 0:
                                circ[i2, j2] = val
        return circ

    f = plt.figure(figsize=(5, 5))    
    circ = circle(Nx, Ny, R=10)
    im = plt.imshow(circ, cmap='inferno')
    plt.colorbar()
    plt.title(r'filter response ($R = 10$ pixels)')
    plt.xlabel('pixels')
    plt.ylabel('pixels')
    # plt.savefig('data/figs/img/response.pdf', dpi=300)
    plt.show()

    def colorbar(mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(mappable, cax=cax)

    f = plt.figure(figsize=(9, 7), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=f, height_ratios=[1, 1], wspace=0.1, hspace=0.5)
    ax_rho = plt.subplot(gs[0, 0])
    ax_rhot = plt.subplot(gs[0, 1])
    ax_rhob = plt.subplot(gs[1, 0])
    ax_eps = plt.subplot(gs[1, 1])

    im1 = ax_rho.imshow(rho, cmap='Greys')
    colorbar(im1)
    ax_rho.set_title(r'design density ($\rho$)')
    ax_rho.set_xlabel('pixels (x)')
    ax_rho.set_ylabel('pixels (y)')
    im2 = ax_rhot.imshow(rhot, cmap='Greys')
    colorbar(im2)
    ax_rhot.set_title(r'filtered density ($\tilde{\rho}$)')   
    ax_rhot.set_xlabel('pixels (x)')
    ax_rhot.set_ylabel('pixels (y)')
    im3 = ax_rhob.imshow(rhob, cmap='Greys')
    colorbar(im3)
    ax_rhob.set_title(r'projected density ($\bar{\rho}$)') 
    ax_rhob.set_xlabel('pixels (x)')
    ax_rhob.set_ylabel('pixels (y)')   
    im4 = ax_eps.imshow(eps, cmap='Greys')
    ax_eps.set_title(r'rel. permittivity ($\epsilon_r$)')
    ax_eps.set_xlabel('pixels (x)')
    ax_eps.set_ylabel('pixels (y)')
    colorbar(im4)

    # plt.savefig('data/figs/img/filter.pdf', dpi=300)
    plt.show()
