import matplotlib.pylab as plt
import numpy as np
import autograd.numpy as npa
from autograd import grad
from functools import partial


eps = np.load('data/figs/data/2port_eps.npy')


def dist(r1, r2):
    return np.sqrt(np.sum(np.square(r1 - r2)))

def get_W(xi1, yi1, xi2, yi2, R=10):
    diff = [xi1 - xi2, yi1 - yi2]
    dist = np.sqrt(diff[0]**2 + diff[1]**2)
    return R - dist

# def get_W(xi2, yi2, R=10):
#     diff = [xi1 - xi2, yi1 - yi2]
#     dist = np.sqrt(diff[0]**2 + diff[1]**2)
#     return R - dist

def eps2rho(eps):
    return (eps - 1) / (eps.max() - 1)

def filter_rho(rho, R=3):
    (Nx, Ny) = rho.shape
    rho_tilde = np.zeros((Nx, Ny))

    diffs = np.arange(-R//2-1, R//2+2)

    for xi1 in range(R//2+1, Nx-R//2-1):
        for yi1 in range(R//2+1, Ny-R//2-1):
            # print('x1 = {}/{}, y1 = {}/{}'.format(xi1, Nx, yi1, Ny))

            den = 0.0
            num = 0.0

            x2s = xi1 + diffs
            y2s = xi1 + diffs

            for x_diff in diffs:
                for y_diff in diffs:
                    xi2 = xi1 + x_diff
                    yi2 = yi1 + y_diff
                    W = get_W(xi1, yi1, xi2, yi2, R=R)
                    num += W*rho[xi2, yi2]
                    den += W
            rho_tilde[xi1, yi1] = num/den
    return rho_tilde

def project_rho(rho, eta=0.5, beta=1):
    num = npa.tanh(beta*eta) + npa.tanh(beta*(rho - eta))
    den = npa.tanh(beta*eta) + npa.tanh(beta*(1 - eta))
    return num / den

rho = eps2rho(eps)
rho_tilde = filter_rho(rho)
rho_bar = project_rho(rho_tilde, eta=0.5, beta=500)

plt.imshow(rho.T)
plt.colorbar()
plt.show()

plt.imshow(rho_tilde.T)
plt.colorbar()
plt.show()

plt.imshow(rho_bar.T)
plt.colorbar()
plt.show()