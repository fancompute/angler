import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from numpy import in1d

from string import ascii_lowercase
import copy

import dill as pickle

import sys
sys.path.append('../')

from angler import Simulation
from angler.structures import two_port, three_port, ortho_port
from device_saver import Device, load_device

scale_bar_pad = 0.75
scale_bar_font_size = 10
fontprops = fm.FontProperties(size=scale_bar_font_size)

def plot_Device(D):

    structure_type = D.structure_type
    if structure_type == 'two_port':
        f = plot_two_port(D)
    elif structure_type == 'three_port':
        f = plot_three_port(D)
    elif structure_type == 'ortho_port':
        f = plot_ortho_port(D)
    else:
        raise ValueError("Incorrect structure_type: {}".format(structure_type))
    return f


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

def plot_ortho_port(D):

    # max_shift = np.max(D.simulation.compute_index_shift())

    # print(np.sum(D.simulation.eps_r[1,:]>1))
    # print(np.sum(D.simulation.eps_r[:,1]>1))
    # print(np.sum(D.simulation.eps_r[:,-1]>1))

    eps_disp, design_region = ortho_port(D.L, D.L2, D.H, D.H2, D.w, D.l, D.dl, D.NPML, D.eps_m)
    # f, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 2, figsize=(7, 10), constrained_layout=True)

    f = plt.figure(figsize=(3.5, 1.375))
    gs = gridspec.GridSpec(1, 2, figure=f)
    ax_lin = plt.subplot(gs[0, 0])
    ax_nl = plt.subplot(gs[0, 1])

    eps_disp = np.flipud(eps_disp.T)

    # linear fields
    E_lin = np.flipud(np.abs(D.Ez.T))
    E_lin = E_lin / np.sqrt(D.W_in)

    vmin = 8
    vmax = E_lin.max()/1.5
    im = ax_lin.pcolormesh(D.x_range, D.y_range, E_lin, cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax))
    im.set_rasterized(True)

    ax_lin.contour(D.x_range, D.y_range, eps_final, levels=2, linewidths=0.2, colors='w')
    ax_lin.set_xlabel('x position ($\mu$m)')
    ax_lin.set_ylabel('y position ($\mu$m)')
    # ax_lin.set_title('linear fields')
    cbar = colorbar(im)
    cbar.ax.set_title('$|E_z|/P_{in}^{1/2}$')    
    # cbar.ax.tick_params(axis='x', direction='in', labeltop=True)
    ax_lin.annotate('low power', xy=(0.5, 0.5), xytext=(0.5, 0.94),
                    xycoords='axes fraction',
                    textcoords='axes fraction',
                    size='medium',
                    color='w',
                    horizontalalignment='center',
                    verticalalignment='center')

    ax_lin.get_xaxis().set_visible(False)
    ax_lin.get_yaxis().set_visible(False)
    scalebar = AnchoredSizeBar(ax_lin.transData,
                               5, '5 $\mu$m', 'lower left', 
                               pad=scale_bar_pad,
                               color='white',
                               frameon=False,
                               size_vertical=0.3,
                               fontproperties=fontprops)
    ax_lin.add_artist(scalebar)
    ax_lin.set_aspect('equal', anchor='C', share=True)

    # nonlinear fields
    E_nl = np.flipud(np.abs(D.Ez_nl.T))
    E_nl = E_nl / np.sqrt(D.W_in)

    vmin = 8
    im = ax_nl.pcolormesh(D.x_range, D.y_range, E_nl, cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax))
    im.set_rasterized(True)

    ax_nl.contour(D.x_range, D.y_range, eps_final, levels=2, linewidths=0.2, colors='w')
    ax_nl.set_xlabel('x position ($\mu$m)')
    ax_nl.set_ylabel('y position ($\mu$m)')
    cbar = colorbar(im)
    cbar.ax.set_title('$|E_z|/P_{in}^{1/2}$')    
    ax_nl.annotate('high power', xy=(0.5, 0.5), xytext=(0.5, 0.94),
                    xycoords='axes fraction',
                    textcoords='axes fraction',
                    size='medium',
                    color='w',
                    horizontalalignment='center',
                    verticalalignment='center')

    ax_nl.get_xaxis().set_visible(False)
    ax_nl.get_yaxis().set_visible(False)
    scalebar = AnchoredSizeBar(ax_nl.transData,
                               5, '5 $\mu$m', 'lower left', 
                               pad=scale_bar_pad,
                               color='white',
                               frameon=False,
                               size_vertical=0.3,
                               fontproperties=fontprops)
    ax_nl.add_artist(scalebar)
    ax_nl.set_aspect('equal', anchor='C', share=True)

    f.tight_layout()
    return f


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


if __name__ == '__main__':

    # fname2 = "data/figs/devices/2_port.p"
    # D2 = load_device(fname2)
    # fig = plot_Device(D2)
    # plt.savefig('data/figs/img/2_port_10_29.pdf', dpi=400)
    # plt.show()

    # fname3 = "data/figs/devices/3_port.p"
    # D3 = load_device(fname3)
    # fig = plot_Device(D3)
    # plt.show()

    fnameT = "data/figs/devices/T_port.p"
    DT = load_device(fnameT)
    fig = plot_Device(DT)
    plt.savefig('data/figs/img/T_port_10_29.pdf', dpi=400)
    plt.show()