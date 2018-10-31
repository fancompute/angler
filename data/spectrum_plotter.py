import numpy as np
import autograd.numpy as npa
import progressbar
import copy
import matplotlib.pylab as plt
from string import ascii_lowercase

from device_saver import load_device

""" Opens a device and prints its stored stats for the paper"""

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


def scan_frequency(D, probe, Nf=300, df=1/200, pbar=True):
    """ Scans the objective function vs. frequency """

    # create frequencies (in Hz)
    delta_f = D.simulation.omega*df
    freqs = 1/2/np.pi*np.linspace(D.simulation.omega - delta_f/2,
                                  D.simulation.omega + delta_f/2,  Nf)

    if pbar:
        bar = progressbar.ProgressBar(max_value=Nf)

    # loop through frequencies
    spectrum = []
    for i, f in enumerate(freqs):

        if pbar:
            bar.update(i + 1)

        # make a new simulation object
        sim_new = copy.deepcopy(D.simulation)

        # reset the simulation to compute new A (hacky way of doing it)
        sim_new.omega = 2*np.pi*f
        sim_new.eps_r = D.simulation.eps_r

        # # solve fields
        _ = sim_new.solve_fields()
        _ = sim_new.solve_fields_nl()

        # compute objective function and append to list
        transmission = probe(sim_new) / D.W_in
        spectrum.append(transmission)

    return freqs, spectrum


def scan_frequency_bistable(D, probe, Nf=300, df=1/200, pbar=True):
    """ Scans the objective function vs. frequency """

    # create frequencies (in Hz)
    delta_f = D.simulation.omega*df
    freqs = 1/2/np.pi*np.linspace(D.simulation.omega - delta_f/2,
                                  D.simulation.omega + delta_f/2,  Nf)

    if pbar:
        bar = progressbar.ProgressBar(max_value=2*Nf)


    (_, _, E_prev) = sim_new.solve_fields()

    # loop through frequencies
    spectrum_down = []
    for i, f in enumerate(freqs):

        if pbar:
            bar.update(i)

        # make a new simulation object
        sim_new = copy.deepcopy(D.simulation)

        # reset the simulation to compute new A (hacky way of doing it)
        sim_new.omega = 2*np.pi*f
        sim_new.eps_r = D.simulation.eps_r

        sim_new.src = 2*sim_new.src

        # # solve fields
        _ = sim_new.solve_fields()
        _ = sim_new.solve_fields_nl(Estart=E_prev)

        # compute objective function and append to list
        transmission = probe(sim_new) / D.W_in
        spectrum_down.append(transmission)

    # loop through frequencies
    spectrum_up = []
    for i, f in enumerate(freqs):

        if pbar:
            bar.update(i + Nf)

        # make a new simulation object
        sim_new = copy.deepcopy(D.simulation)

        # reset the simulation to compute new A (hacky way of doing it)
        sim_new.omega = 2*np.pi*f
        sim_new.eps_r = D.simulation.eps_r

        # # solve fields
        _ = sim_new.solve_fields()
        _ = sim_new.solve_fields_nl()

        # compute objective function and append to list
        transmission = probe(sim_new) / D.W_in
        spectrum_up.append(transmission)

    return freqs, spectrum_down, spectrum_up   


def get_spectrum(D):

    if D.structure_type == 'two_port':
        probe_out_low = lambda simulation: simulation.flux_probe('x', [-D.NPML[0]-int(D.l/2/D.dl), D.ny], int(D.Ny/2), nl=False)        
        probe_out_high = lambda simulation: simulation.flux_probe('x', [-D.NPML[0]-int(D.l/2/D.dl), D.ny], int(D.Ny/2), nl=True)
        probes = [probe_out_low, probe_out_high]            

    elif D.structure_type == 'ortho_port':
        probe_right_low = lambda simulation: simulation.flux_probe('x', [-D.NPML[0]-int(D.l/2/D.dl), D.ny], int(D.H/2/D.dl), nl=False)
        probe_top_low = lambda simulation: simulation.flux_probe('y', [D.nx, -D.NPML[1]-int(D.l/2/D.dl)], int(D.H/2/D.dl), nl=False)        
        probe_right_high = lambda simulation: simulation.flux_probe('x', [-D.NPML[0]-int(D.l/2/D.dl), D.ny], int(D.H/2/D.dl), nl=True)
        probe_top_high = lambda simulation: simulation.flux_probe('y', [D.nx, -D.NPML[1]-int(D.l/2/D.dl)], int(D.H/2/D.dl), nl=True)
        probes = [probe_right_low, probe_top_low, probe_right_high, probe_top_high]

    spectra = []
    for probe_index, probe in enumerate(probes):
        freqs, spectrum = scan_frequency(D, probe)  
        spectra.append(spectrum)

    return freqs, spectra

def plot_spectra(D, freqs, spectra):

    if D.structure_type == 'two_port':
        leg = ('low power', 'high power')
    elif D.structure_type == 'ortho_port':
        leg = ('low power (right)', 'low power (bottom)', 'high power (right)', 'high power (bottom)')

    for spectrum in spectra:
        plt.plot(freqs, spectrum)
    plt.legend(leg)
    plt.xlabel('frequency difference (GHz)')
    plt.ylabel('transmission')
    plt.show()

def plot_from_data(fname_freqs2, fname_spectra2, fname_freqsT, fname_spectraT):

    freqs2 = np.load(fname_freqs2)
    spectra2 = np.load(fname_spectra2)
    freqsT = np.load(fname_freqsT)
    spectraT = np.load(fname_spectraT)
        
    f, (ax1, ax2, ax3) = plt.subplots(3, constrained_layout=True, figsize=(6, 6))

    freqs_GHz2 = [(f-150e12)/1e9 for f in freqs2]

    for spectrum in spectra2:
        ax1.plot(freqs_GHz2, spectrum) 
    ax1.legend(('low power', 'high power'))
    ax1.set_xlabel('frequency difference (GHz)')
    ax1.set_ylabel('transmission')
    ax1.set_ylim(0, 1)

    freqs_GHzT = [(f-150e12)/1e9 for f in freqsT]

    ax2.plot(freqs_GHzT, spectraT[0], color='#2ca02c')
    ax2.plot(freqs_GHzT, spectraT[1], color='#d62728')    
    ax2.legend(('right port', 'bottom port'), loc='upper right')     
    ax2.set_xlabel('frequency difference (GHz)')
    ax2.set_ylabel('transmission')
    ax2.set_title('low power regime')
    ax2.set_ylim(0, 1)

    ax3.plot(freqs_GHzT, spectraT[2], color='#9467bd')
    ax3.plot(freqs_GHzT, spectraT[3], color='#8c564b')
    ax3.legend(('right port', 'bottom port'), loc='upper right')
    ax3.set_xlabel('frequency difference (GHz)')
    ax3.set_ylabel('transmission')
    ax3.set_title('high power regime')
    ax3.set_ylim(0, 1)
    apply_sublabels([ax1, ax2, ax3], [False, False, False], x=19, y=-5, size='large', ha='right', va='top', prefix='(', postfix=')')
    plt.savefig('data/figs/img/spectra.pdf', dpi=400)
    plt.show()


if __name__ == '__main__':

    fname2 = 'data/figs/devices/2_port.p'
    D = load_device(fname2)
    freqs, spectra2 = get_spectrum(D)
    freqs_GHz = [(f-150e12)/1e9 for f in freqs]
    plot_spectra(D, freqs_GHz, spectra2)

    # np.save('data/freqs2', freqs)
    # np.save('data/spectra2', spectra2)

    # fnameT = 'data/figs/devices/T_port.p'
    # D = load_device(fnameT)
    # freqs, spectraT = get_spectrum(D)
    # freqs_GHz = [(f-150e12)/1e9 for f in freqs]
    # plot_spectra(D, freqs_GHz, spectraT)

    # fname_freqs2 = 'data/spectra/freqs2.npy'
    # fname_spectra2 = 'data/spectra/spectra2.npy'
    # fname_freqsT = 'data/spectra/freqs.npy'
    # fname_spectraT = 'data/spectra/spectraT.npy'

    # plot_from_data(fname_freqs2, fname_spectra2, fname_freqsT, fname_spectraT)



