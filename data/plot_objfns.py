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

from fdfdpy import Simulation
from structures import two_port, three_port, ortho_port
from device_saver import Device, load_device

scale_bar_pad = 0.75
scale_bar_font_size = 10
fontprops = fm.FontProperties(size=scale_bar_font_size)


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def set_axis_font(ax, font_size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)

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

def pad_array(array, padding, val):
    (Nx, Ny) = array.shape
    pad = val*np.ones((Nx, padding))
    new_array = array.T
    new_array = np.vstack([pad.T, new_array, pad.T])
    return new_array

def pad_list(ls, padding, dl):
    N = len(ls)
    N_new = N + 2*padding
    n = N_new/2
    ls_new = [dl*(l-n) for l in range(N_new)]
    return ls_new



############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

def plot_two_port(D):

    # scale_bar_pad = 0.1
    pad_grids = 50

    f = plt.figure(figsize=(7, 3.9))
    gs = gridspec.GridSpec(2, 2, figure=f, height_ratios=[1, 1])
    ax_drawing = plt.subplot(gs[0, 0])
    ax_power = plt.subplot(gs[0, 1])
    ax_lin = plt.subplot(gs[1, 0])
    ax_nl = plt.subplot(gs[1, 1])

    _, design_region = two_port(D.L, D.H, D.w, D.l, D.spc, D.dl, D.NPML, D.eps_m)
    eps_disp = pad_array(D.simulation.eps_r, pad_grids, 1)

    # draw structure
    y_range = pad_list(list(D.y_range), pad_grids, D.dl)
    im = ax_drawing.pcolormesh(D.x_range, y_range, eps_disp, cmap='Greys')
    im.set_rasterized(True)

    ax_drawing.set_xlabel('x position ($\mu$m)')
    ax_drawing.set_ylabel('y position ($\mu$m)')
    base_in = 9.5
    tip_in = 5.5
    y_shift = 0.0
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
                                    alpha=1, facecolor='none', edgecolor='k', linestyle='--')
    ax_drawing.add_patch(design_box)
    ax_drawing.annotate('design region', (0.5, 0.5), xytext=(0.0, 1),
                    xycoords='axes fraction',
                    textcoords='data',
                    size='medium',
                    color='k',
                    horizontalalignment='center',
                    verticalalignment='center')
    ax_drawing.annotate('linear', (0.5, 0.5), xytext=(5.6, 1),
                    xycoords='axes fraction',
                    textcoords='data',
                    size='medium',
                    color='k',
                    horizontalalignment='left',
                    verticalalignment='center')
    ax_drawing.annotate('nonlinear', (0.5, 0.5), xytext=(4.6, -y_dist - 1),
                    xycoords='axes fraction',
                    textcoords='data',
                    size='medium',
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
    # ax_drawing.annotate('optimization', xy=(0.5, 0.5), xytext=(0.5, 0.9),
    #                 xycoords='axes fraction',
    #                 textcoords='axes fraction',
    #                 size='medium',
    #                 color='k',
    #                 horizontalalignment='center',
    #                 verticalalignment='center')
    ax_drawing.set_aspect('equal', anchor='C', share=True)

    # power plot
    ax_power.plot(D.powers, D.transmissions[0], color='#0066cc')
    ax_power.plot(2*[D.W_in], [0, 1], linestyle='dashed', linewidth=1, color='k')    
    ax_power.set_xscale('log')
    ax_power.set_xlabel('input power (W / $\mu$m)')
    ax_power.set_ylabel('transmission')
    ax_power.set_ylim([-0.01, 1.01])

    # linear fields
    E_lin = np.abs(D.Ez)
    E_lin = pad_array(E_lin, pad_grids, 1e-3)
    E_lin = E_lin / np.sqrt(D.W_in)

    vmin = 3
    vmax = E_lin.max()/1.5

    im = ax_lin.pcolormesh(D.x_range, y_range, E_lin, cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax))
    im.set_rasterized(True)

    ax_lin.contour(D.x_range, y_range, eps_disp, levels=2, linewidths=0.2, colors='w')
    ax_lin.set_xlabel('x position ($\mu$m)')
    ax_lin.set_ylabel('y position ($\mu$m)')
    # ax_lin.set_title('linear fields')
    cbar = colorbar(im)
    cbar.ax.set_title('$|E_z|/P_{in}^{1/2}$')
    # cbar.ax.tick_params(axis='x', direction='in', labeltop=True)
    ax_lin.annotate('low power', xy=(0.5, 0.5), xytext=(0.5, 0.9),
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
    E_nl = np.abs(D.Ez_nl)
    E_nl = pad_array(E_nl, pad_grids, 1e-3)
    E_nl = E_nl / np.sqrt(D.W_in)

    vmin = 3
    # vmax = E_nl.max()    
    im = ax_nl.pcolormesh(D.x_range, y_range, E_nl, cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax))
    im.set_rasterized(True)

    ax_nl.contour(D.x_range, y_range, eps_disp, levels=2, linewidths=0.2, colors='w')
    ax_nl.set_xlabel('x position ($\mu$m)')
    ax_nl.set_ylabel('y position ($\mu$m)')
    cbar = colorbar(im)
    cbar.ax.set_title('$|E_z|/P_{in}^{1/2}$')    
    ax_nl.annotate('high power', xy=(0.5, 0.5), xytext=(0.5, 0.9),
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


    # # objective function
    # obj_list = D.optimization.objfn_list
    # iter_list = range(1, len(obj_list) + 1)
    # ax_obj.plot(iter_list, obj_list, color='k')
    # ax_obj.set_xlabel('iteration')
    # ax_obj.set_ylabel('objective function')
    # ax_obj.set_ylim([-0.01, 1.01])



    apply_sublabels([ax_drawing, ax_power, ax_lin, ax_nl], invert_color_inds=[False, False, True, True])
    f.tight_layout()

    return f

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

def plot_three_port(D):

    f = plt.figure(figsize=(7, 8))
    gs = gridspec.GridSpec(3, 2, figure=f, height_ratios=[1, 1, 0.3])
    ax_drawing = plt.subplot(gs[0, 0])
    ax_eps = plt.subplot(gs[0, 1])
    ax_lin = plt.subplot(gs[1, 0])
    ax_nl = plt.subplot(gs[1, 1])
    ax_obj = plt.subplot(gs[2, 0])
    ax_power = plt.subplot(gs[2, 1])

    eps_disp, design_region = three_port(D.L, D.H, D.w, D.l, D.spc, D.dl, D.NPML, D.eps_m)

    # draw structure
    im = ax_drawing.pcolormesh(D.x_range, D.y_range, eps_disp.T, cmap='Greys')
    im.set_rasterized(True)

    ax_drawing.set_xlabel('x position ($\mu$m)')
    ax_drawing.set_ylabel('y position ($\mu$m)')
    y_dist = 1.6
    base_in = 7
    tip_in = 3
    y_shift = 0.01
    arrow_in = mpatches.FancyArrowPatch((-base_in, +y_shift), (-tip_in, y_shift),
                                     mutation_scale=20, facecolor='#cc99ff')
    ax_drawing.add_patch(arrow_in)
    arrow_top = mpatches.FancyArrowPatch((tip_in, y_dist+0.1+y_shift), (base_in, y_dist+0.1+y_shift),
                                     mutation_scale=20, facecolor='#3366ff',
                                     edgecolor='k')
    ax_drawing.add_patch(arrow_top)
    arrow_bot = mpatches.FancyArrowPatch((tip_in, -y_dist+y_shift), (base_in, -y_dist+y_shift),
                                     mutation_scale=20, facecolor='#ff5050')
    ax_drawing.add_patch(arrow_bot)

    design_box = mpatches.Rectangle(xy=(-D.L/2, -D.H/2), width=D.L, height=D.H,
                                    alpha=0.5,
                                    edgecolor='k',
                                    linestyle='--')
    ax_drawing.add_patch(design_box)

    ax_drawing.annotate('design\nregion', (0.5, 0.5), xytext=(0.0, 0.0),
                    xycoords='axes fraction',
                    textcoords='data',
                    size='small',
                    color='k',
                    horizontalalignment='center',
                    verticalalignment='center')
    ax_drawing.annotate('linear', (0, 0), xytext=(3.5, 0.8),
                    xycoords='axes fraction',
                    textcoords='data',
                    size='small',
                    color='k',
                    horizontalalignment='left',
                    verticalalignment='center')
    ax_drawing.annotate('nonlinear', (0, 0), xytext=(3.5, -2.4),
                    xycoords='axes fraction',
                    textcoords='data',
                    size='small',
                    color='k',
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
    ax_drawing.annotate('optimization definition', xy=(0.5, 0.5), xytext=(0.5, 0.9),
                    xycoords='axes fraction',
                    textcoords='axes fraction',
                    size='medium',
                    color='k',
                    horizontalalignment='center',
                    verticalalignment='center')
    ax_drawing.set_aspect('equal', anchor='C', share=True)


    # permittivity
    im = ax_eps.pcolormesh(D.x_range, D.y_range, D.simulation.eps_r.T, cmap='Greys')
    im.set_rasterized(True)

    ax_eps.set_xlabel('x position ($\mu$m)')
    ax_eps.set_ylabel('y position ($\mu$m)')
    # ax_eps.set_title('relative permittivity')
    cbar = colorbar(im)
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
    ax_eps.annotate('final structure', xy=(0.5, 0.5), xytext=(0.5, 0.85),
                    xycoords='axes fraction',
                    textcoords='axes fraction',
                    size='medium',
                    color='k',
                    horizontalalignment='center',
                    verticalalignment='center')
    ax_eps.set_aspect('equal', anchor='C', share=True)

    # linear fields
    E_lin = np.abs(D.Ez.T)
    vmin = 3
    vmax = E_lin.max()
    im = ax_lin.pcolormesh(D.x_range, D.y_range, E_lin, cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax))
    im.set_rasterized(True)

    ax_lin.contour(D.x_range, D.y_range, D.simulation.eps_r.T, levels=2, linewidths=0.2, colors='w')
    ax_lin.set_xlabel('x position ($\mu$m)')
    ax_lin.set_ylabel('y position ($\mu$m)')
    # ax_lin.set_title('linear fields')
    cbar = colorbar(im)
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
    ax_lin.set_aspect('equal', anchor='C', share=True)

    # nonlinear fields
    E_nl = np.abs(D.Ez_nl.T)
    vmin = 3
    vmax = E_nl.max()    
    im = ax_nl.pcolormesh(D.x_range, D.y_range, E_nl, cmap='inferno', norm=LogNorm(vmin=vmin, vmax=vmax))
    im.set_rasterized(True)

    ax_nl.contour(D.x_range, D.y_range, D.simulation.eps_r.T, levels=2, linewidths=0.2, colors='w')
    ax_nl.set_xlabel('x position ($\mu$m)')
    ax_nl.set_ylabel('y position ($\mu$m)')
    cbar = colorbar(im)
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
    ax_nl.set_aspect('equal', anchor='C', share=True)


    # objective function
    obj_list = D.optimization.objfn_list
    iter_list = range(1, len(obj_list) + 1)
    ax_obj.plot(iter_list, obj_list)
    ax_obj.set_xlabel('iteration')
    ax_obj.set_ylabel('objective function')


############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


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

def plot_objs(D2, DT):

    f, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 2.5))

    # objective function
    obj_list2 = D2.optimization.objfn_list
    iter_list2 = range(1, len(obj_list2) + 1)

    obj_listT = DT.optimization.objfn_list
    iter_listT = range(1, len(obj_listT) + 1)

    ax1.plot(iter_list2, obj_list2)
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('objective function')
    ax1.set_ylim(0, 1)

    ax2.plot(iter_listT, obj_listT)
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('objective function')
    ax2.set_ylim(0, 1)

    # ax2.set_xscale('log')

    apply_sublabels([ax1, ax2], [False, False], x=19, y=-5, size='large', ha='right', va='top', prefix='(', postfix=')')    
    plt.savefig('data/figs/img/objfns_11_1.pdf', dpi=400)
    plt.show()

if __name__ == '__main__':

    fname2 = "data/figs/devices/2_port.p"
    D2 = load_device(fname2)

    fnameT = "data/figs/devices/T_port.p"
    DT = load_device(fnameT)
    
    plot_objs(D2, DT)
