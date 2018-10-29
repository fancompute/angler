import numpy as np
import autograd.numpy as npa
import progressbar
import copy
import matplotlib.pylab as plt

from device_saver import load_device

""" Opens a device and prints its stored stats for the paper"""

def scan_frequency(D, probe, Nf=5, df=1/200, pbar=True):
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


def get_spectrum(fname, D):

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

if __name__ == '__main__':

    fname2 = 'data/figs/devices/2_port.p'
    D = load_device(fname2)
    freqs, spectra2 = get_spectrum(fname2, D)
    freqs_GHz = [(f-150e12)/1e9 for f in freqs]
    plot_spectra(D, freqs_GHz, spectra2)

    # fnameT = 'data/figs/devices/T_port.p'
    # freqs, spectraT = plot_spectrum(fnameT)
    # plot_spectra(D, freqs, spectraT)

