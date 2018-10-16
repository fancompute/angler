import matplotlib.pylab as plt
import numpy as np
import autograd.numpy as npa
from autograd import grad
from functools import partial

## TESTS THE THRESHOLDING OF A FILTERED FIELD TO A PROJECTED FIELD

eta = 0.5
beta = 10

def bar(p, beta=1):
    num = npa.tanh(beta*eta) + npa.tanh(beta*(p - eta))
    den = npa.tanh(beta*eta) + npa.tanh(beta*(1 - eta))
    return num / den

def plot_bar(betas=[10]):
    N = 10000
    ps = np.linspace(0, 1, N)
    for beta in betas:
        plt.plot(ps, bar(ps, beta=beta))
    plt.legend(['beta = ' + str(b) for b in betas])
    plt.xlabel('original density')
    plt.ylabel('projected density')
    plt.show()

def plot_grad_bar(betas=[10]):
    N = 10000
    ps = np.linspace(0, 1, N)
    grad_bar = grad(bar)
    for beta in betas:
        bar_beta = partial(bar, beta=beta)
        grad_bar = grad(bar_beta)
        # grad_bar = np.vectorize(grad_bar)
        plt.plot(ps, grad_bar(ps))
    plt.legend(['beta = ' + str(b) for b in betas])
    plt.show()

def plot_eps_bars(betas=[10]):
    n_betas = len(betas)
    f, ax_list = plt.subplots(n_betas+1, constrained_layout=False)
    ax0 = ax_list[0]
    eps = np.load('data/figs/data/2port_eps.npy')
    p = (eps - 1) / (eps.max() - 1)
    im1 = ax0.imshow(p.T)
    plt.colorbar(im1, ax=ax0)
    ax0.set_title('original rho')
    num_middle = p.size - np.sum(p == 0) + np.sum(p == 1)
    print('{} grid points not 1 or eps_max'.format(num_middle))
    for i in range(1, n_betas+1):
        axi = ax_list[i]
        beta = betas[i-1]
        p_bar = bar(p, beta=beta)
        imi = axi.imshow(p_bar.T)
        axi.set_title('beta = {}'.format(beta))
        num_middle = p_bar.size - np.sum(p_bar <= 0.01) - np.sum(p_bar >= 0.99)
        print('{} grid points not 1 or eps_max'.format(num_middle))        
    plt.show()

betas = [1, 10, 100, 1000, 10000, 100000]
# plot_bar(betas)
# plot_grad_bar(betas[:-1])
plot_eps_bars(betas=betas)
