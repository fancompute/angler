import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np
import copy
from collections import namedtuple
import dill as pickle
import sys
sys.path.append('../')

from fdfdpy import Simulation
from structures import two_port, three_port

from string import ascii_lowercase
from numpy import in1d

scale_bar_pad = 0.75
scale_bar_font_size = 10
fontprops = fm.FontProperties(size=scale_bar_font_size)

def plot_Device(D):

    structure_type = D.structure_type
    if structure_type == 'two_port':
        f = plot_two_port(D)
    return f

def plot_two_port(D):

    eps_disp, design_region = two_port(D.L, D.H, D.w, D.l, D.spc, D.dl, D.NPML, D.eps_m)
    f, (ax_top, ax_bot) = plt.subplots(2, 2, figsize=(7, 5), constrained_layout=True)

    # draw structure
    ax_drawing = ax_top[0]
    im = ax_drawing.pcolormesh(D.x_range, D.y_range, eps_disp.T, cmap='Greys')
    ax_drawing.set_xlabel('x position ($\mu$m)')
    ax_drawing.set_ylabel('y position ($\mu$m)')
    base_in = 9.5
    tip_in = 5.5
    y_shift = 0.05
    y_dist = 1.5    
    arrow_in = mpatches.FancyArrowPatch((-base_in, y_shift), (-tip_in, y_shift),
                                        mutation_scale=20, facecolor='#cc99ff')
    ax_drawing.add_patch(arrow_in)
    arrow_top = mpatches.FancyArrowPatch((tip_in, y_shift), (base_in, y_shift),
                                         mutation_scale=20, facecolor='#3366ff')
    ax_drawing.add_patch(arrow_top)
    arrow_bot = mpatches.FancyArrowPatch((tip_in, -y_dist), (base_in, -y_dist),
                                         mutation_scale=20, facecolor='#ff5050')
    ax_drawing.add_patch(arrow_bot)
    design_box = mpatches.Rectangle(xy=(-D.L/2, -D.H/2), width=D.L, height=D.H,
                                    alpha=0.5, edgecolor='k', linestyle='--')
    ax_drawing.add_patch(design_box)
    ax_drawing.annotate('design region', (0.5, 0.5), xytext=(0.0, 0.75),
                    xycoords='axes fraction',
                    textcoords='data',
                    size='small',
                    color='k',
                    horizontalalignment='center',
                    verticalalignment='center')
    ax_drawing.annotate('linear', (0.5, 0.5), xytext=(6, 1),
                    xycoords='axes fraction',
                    textcoords='data',
                    size='small',
                    color='k',
                    horizontalalignment='left',
                    verticalalignment='center')
    ax_drawing.annotate('nonlinear', (0.5, 0.5), xytext=(6, -y_dist - 1),
                    xycoords='axes fraction',
                    textcoords='data',
                    size='small',
                    color='k',
                    horizontalalignment='left',
                    verticalalignment='center')
    ax_drawing.annotate('X', (0.5, 0.5), xytext=(base_in-2.5, -y_dist-0.1),
                    xycoords='axes fraction',
                    textcoords='data',
                    size='large',
                    color='k',
                    fontweight='extra bold',
                    horizontalalignment='left',
                    verticalalignment='center')
    ax_drawing.get_xaxis().set_visible(False)
    ax_drawing.get_yaxis().set_visible(False)
    scalebar = AnchoredSizeBar(ax_drawing.transData,
                               5, '5 $\mu$m', 'lower left', 
                               pad=scale_bar_pad,
                               color='black',
                               frameon=False,
                               size_vertical=0.3,
                               fontproperties=fontprops)
    ax_drawing.add_artist(scalebar)
    ax_drawing.annotate('optimization definition', xy=(0.5, 0.5), xytext=(0.5, 0.94),
                    xycoords='axes fraction',
                    textcoords='axes fraction',
                    size='medium',
                    color='k',
                    horizontalalignment='center',
                    verticalalignment='center')


    # permittivity
    ax_eps = ax_top[1]
    im = ax_eps.pcolormesh(D.x_range, D.y_range, D.simulation.eps_r.T, cmap='Greys')
    ax_eps.set_xlabel('x position ($\mu$m)')
    ax_eps.set_ylabel('y position ($\mu$m)')
    # ax_eps.set_title('relative permittivity')
    cbar = plt.colorbar(im, ax=ax_eps)
    cbar.ax.set_title('$\epsilon_r$')

    ax_eps.get_xaxis().set_visible(False)
    ax_eps.get_yaxis().set_visible(False)
    scalebar = AnchoredSizeBar(ax_eps.transData,
                               5, '5 $\mu$m', 'lower left', 
                               pad=scale_bar_pad,
                               color='black',
                               frameon=False,
                               size_vertical=0.3,
                               fontproperties=fontprops)

    ax_eps.add_artist(scalebar)
    ax_eps.annotate('final structure', xy=(0.5, 0.5), xytext=(0.5, 0.94),
                    xycoords='axes fraction',
                    textcoords='axes fraction',
                    size='medium',
                    color='k',
                    horizontalalignment='center',
                    verticalalignment='center')

    # linear fields
    ax_lin = ax_bot[0]
    E_lin = np.abs(D.Ez.T)
    vmin = 1
    vmax = E_lin.max()
    im = ax_lin.pcolormesh(D.x_range, D.y_range, E_lin, cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax))
    ax_lin.contour(D.x_range, D.y_range, D.simulation.eps_r.T, levels=2, linewidths=0.2, colors='w')
    ax_lin.set_xlabel('x position ($\mu$m)')
    ax_lin.set_ylabel('y position ($\mu$m)')
    # ax_lin.set_title('linear fields')
    cbar = plt.colorbar(im, ax=ax_lin)
    cbar.ax.set_title('$|E_z|$')
    # cbar.ax.tick_params(axis='x', direction='in', labeltop=True)
    ax_lin.annotate('linear fields', xy=(0.5, 0.5), xytext=(0.5, 0.94),
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

    # nonlinear fields
    ax_nl = ax_bot[1]
    E_nl = np.abs(D.Ez_nl.T)
    vmin = 1
    vmax = E_nl.max()    
    im = ax_nl.pcolormesh(D.x_range, D.y_range, E_nl, cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax))
    ax_nl.contour(D.x_range, D.y_range, D.simulation.eps_r.T, levels=2, linewidths=0.2, colors='w')
    ax_nl.set_xlabel('x position ($\mu$m)')
    ax_nl.set_ylabel('y position ($\mu$m)')
    cbar = plt.colorbar(im, ax=ax_nl)
    cbar.ax.set_title('$|E_z|$')    
    ax_nl.annotate('nonlinear fields', xy=(0.5, 0.5), xytext=(0.5, 0.94),
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

    apply_sublabels([ax_drawing, ax_eps, ax_lin, ax_nl], invert_color_inds=[False, False, True, True])

    return f

def apply_sublabels(axs, invert_color_inds, x=19, y=-5, size='large', ha='right', va='top', prefix='(', postfix=')'):
    # axs = list of axes
    # invert_color_ind = list of booleans (True to make sublabels white, else False)
    for n, ax in enumerate(axs):
        if invert_color_inds[n]:
            color='w'
        else:
            color='k'
        ax.annotate(prefix + ascii_lowercase[n] + postfix,
                    xy=(0, 1),
                    xytext=(x, y),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    size=size,
                    color=color,
                    horizontalalignment=ha,
                    verticalalignment=va)

def load_device(fname):
    """ Loads the pickled Device object """

    D_dict = pickle.load(open(fname, "rb"))
    D = namedtuple('Device', D_dict.keys())(*D_dict.values())
    return D

if __name__ == '__main__':
    fname = "2_port.p"
    D = load_device(fname)
    fig = plot_Device(D)
    plt.show()
