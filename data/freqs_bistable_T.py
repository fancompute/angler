import numpy as np
import autograd.numpy as npa
import progressbar
import copy
import matplotlib.pylab as plt
from string import ascii_lowercase

from device_saver import load_device


def scan_power(simulation, probes=None, omega=None, Ns=10, s_min=1e-1, s_max=1e2, solver='newton'):
    """ Scans the source amplitude and computes the objective function
        probes is a list of functions for computing the power, for example:
        [lambda simulation: simulation.flux_probe('x', [-NPML[0]-int(l/2/dl), ny + int(d/2/dl)], int(H/2/dl))]
    """

    if probes is None:
        raise ValueError("need to specify 'probes' kwarg as a list of functions for computing the power in each port.")
    num_probes = len(probes)

    # create src_amplitudes
    s_list = np.logspace(np.log10(s_min), np.log10(s_max), Ns)        

    # bar = progressbar.ProgressBar(max_value=2*Ns)

    # transmission
    transmissions_up = [[] for _ in range(num_probes)]
    powers = []

    # (_,_,E_prev) = simulation.solve_fields()
    E_prev = np.zeros(simulation.eps_r.shape)
    for i, s in enumerate(s_list):

        # bar.update(i)

        # make a new simulation object
        sim_new = copy.deepcopy(simulation)
        if omega is not None:
            sim_new.omega = omega
            sim_new.eps_r = sim_new.eps_r

        sim_new.modes[0].scale = s
        sim_new.modes[0].setup_src(sim_new)
        W_in = sim_new.W_in

        powers.append(W_in)

        (_,_,E_prev,c) = sim_new.solve_fields_nl(timing=False, averaging=True,
                    Estart=E_prev, solver_nl=solver, conv_threshold=1e-3,
                    max_num_iter=300)

        # compute power transmission using each probe
        for probe_index, probe in enumerate(probes):
            W_out = probe(sim_new)
            T_up = W_out / W_in
            transmissions_up[probe_index].append(T_up)

    # transmission
    transmissions_down = [[] for _ in range(num_probes)]

    for i, s in enumerate(reversed(s_list)):

        # bar.update(i + Ns)

        # make a new simulation object
        sim_new = copy.deepcopy(simulation)
        if omega is not None:
            sim_new.omega = omega
            sim_new.eps_r = sim_new.eps_r

        sim_new.modes[0].scale = s
        sim_new.modes[0].setup_src(sim_new)
        W_in = sim_new.W_in

        (_,_,E_prev,c) = sim_new.solve_fields_nl(timing=False, averaging=True,
                    Estart=E_prev, solver_nl=solver, conv_threshold=1e-3,
                    max_num_iter=100)
        # compute power transmission using each probe
        for probe_index, probe in enumerate(probes):
            W_out = probe(sim_new)
            T_down = W_out / W_in
            transmissions_down[probe_index].append(T_down)

    transmissions_down = [list(reversed(t)) for t in transmissions_down]
    return powers, transmissions_up, transmissions_down


def get_branches(D, simulation, probes):

    powers, Ts_up, Ts_down = scan_power(simulation, probes=probes)

    Ts_up_Win = []
    Ts_down_Win = []

    for i, _ in enumerate(probes):

        Ts_up_Win.append(np.interp(D.W_in, powers, Ts_up[i]))
        Ts_down_Win.append(np.interp(D.W_in, powers, Ts_down[i]))

    return Ts_up_Win, Ts_down_Win


def scan_frequency(D, probes, Nf=40, df=1/200, fmin=None, fmax=None, pbar=True):
    """ Scans the objective function vs. frequency """

    # create frequencies (in Hz)
    delta_f = D.simulation.omega*df
    freqs = 1/2/np.pi*np.linspace(D.simulation.omega - delta_f/2,
                                  D.simulation.omega + delta_f/2,  Nf)

    if fmin is not None and fmax is not None:
        freqs = np.linspace(fmin, fmax, Nf)


    if pbar:
        bar = progressbar.ProgressBar(max_value=Nf)

    spectra = [[] for _ in range(2*len(probes))]
    for fi, f in enumerate(freqs):
        # print('  freq = {} / {}'.format(fi, Nf))

        if pbar:
            bar.update(fi)

        # make a new simulation object
        sim_new = copy.deepcopy(D.simulation)

        # reset the simulation to compute new A (hacky way of doing it)
        sim_new.omega = 2*np.pi*f
        sim_new.eps_r = D.simulation.eps_r

        Ts_up, Ts_down = get_branches(D, sim_new, probes)

        for pi, p in enumerate(probes):
            spectra[2*pi].append(Ts_up[pi])
            spectra[2*pi+1].append(Ts_down[pi])

    return freqs, spectra

def get_spectra(D):

    if D.structure_type == 'two_port':
        probe_out = lambda simulation: simulation.flux_probe('x', [-D.NPML[0]-int(D.l/2/D.dl), D.ny], int(D.Ny/2), nl=True)
        probes = [probe_out]            

    elif D.structure_type == 'ortho_port':
        probe_right = lambda simulation: simulation.flux_probe('x', [-D.NPML[0]-int(D.l/2/D.dl), D.ny], int(D.H/2/D.dl), nl=True)
        probe_top = lambda simulation: simulation.flux_probe('y', [D.nx, -D.NPML[1]-int(D.l/2/D.dl)], int(D.H/2/D.dl), nl=True)
        probes = [probe_right, probe_top]

    freqs, spectra = scan_frequency(D, probes)

    return freqs, spectra

def plot_spectra(D, freqs, spectra):

    freqs_GHz = [(f-150e12)/1e9 for f in freqs]

    if D.structure_type == 'two_port':
        Np = 1
        leg = ('up', 'down')
    elif D.structure_type == 'ortho_port':
        Np = 2
        leg = ('up (right)', 'down (right)', 'up (bottom)','down (bottom)')

    for i in range(2*Np):
        plt.plot(freqs_GHz, spectra[i])
    plt.legend(leg)
    plt.yscale('log')
    plt.xlabel('frequency difference (GHz)')
    plt.ylabel('nonlinear transmission')
    plt.show()

if __name__ == '__main__':

    # fname2 = 'data/figs/devices/2_port.p'
    # D = load_device(fname2)
    # print(D.W_in)
    # freqs, spectra2 = get_spectra(D)
    # plot_spectra(D, freqs, spectra2)    

    # np.save('data/freqs2_bi', freqs)
    # np.save('data/spectra2_bi', spectra2)

    fnameT = 'data/figs/devices/T_port.p'
    D = load_device(fnameT)
    print(D.W_in)
    freqs, spectraT = get_spectra(D)
    plot_spectra(D, freqs, spectraT)  

    # np.save('data/freqsT_bi', freqs)
    # np.save('data/spectraT_bi', spectra2)

    # fname_freqs2 = 'data/spectra/freqs2.npy'
    # fname_spectra2 = 'data/spectra/spectra2.npy'
    # fname_freqsT = 'data/spectra/freqs.npy'
    # fname_spectraT = 'data/spectra/spectraT.npy'

    # plot_from_data(fname_freqs2, fname_spectra2, fname_freqsT, fname_spectraT)

freqs_GHz = [(f-150e12)/1e9 for f in freqs]
Np = 2
leg = ('up (right)', 'down (right)', 'up (bottom)','down (bottom)')


for i in range(2*Np):
    plt.scatter(freqs_GHz, spectraT[i])


plt.legend(leg)
plt.yscale('linear')
plt.xlabel('frequency difference (GHz)')
plt.ylabel('nonlinear transmission')
plt.show()


