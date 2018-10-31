import numpy as np
import autograd.numpy as npa
import progressbar
import copy
import matplotlib.pylab as plt
from string import ascii_lowercase

from device_saver import load_device


def trans_bistable(D, omega=None, Np=50, s_min=1e-1, s_max=1e3):

    if D.structure_type == 'two_port':
        probe_out = lambda simulation: simulation.flux_probe('x', [-D.NPML[0]-int(D.l/2/D.dl), D.ny], int(D.Ny/2), nl=True)
        probes = [probe_out]            

    elif D.structure_type == 'three_port':
        probe_top = lambda simulation: simulation.flux_probe('x', [-D.NPML[0]-int(D.l/2/D.dl), D.ny+int(D.d/2/D.dl)], int(D.H/2/D.dl), nl=True)
        probe_bot = lambda simulation: simulation.flux_probe('x', [-D.NPML[0]-int(D.l/2/D.dl), D.ny-int(D.d/2/D.dl)], int(D.H/2/D.dl), nl=True)
        probes = [probe_top, probe_bot]            

    elif D.structure_type == 'ortho_port':
        probe_right = lambda simulation: simulation.flux_probe('x', [-D.NPML[0]-int(D.l/2/D.dl), D.ny], int(D.H/2/D.dl), nl=True)
        probe_top = lambda simulation: simulation.flux_probe('y', [D.nx, -D.NPML[1]-int(D.l/2/D.dl)], int(D.H/2/D.dl), nl=True)
        probes = [probe_right, probe_top]

    powers, T_up, T_down = scan_power(D.simulation, probes=probes, omega=omega, Ns=Np, s_min=s_min, s_max=s_max)

    return powers, T_up, T_down

def scan_power(simulation, probes=None, omega=None, Ns=50, s_min=1e-2, s_max=1e2, solver='hybrid'):
    """ Scans the source amplitude and computes the objective function
        probes is a list of functions for computing the power, for example:
        [lambda simulation: simulation.flux_probe('x', [-NPML[0]-int(l/2/dl), ny + int(d/2/dl)], int(H/2/dl))]
    """

    if probes is None:
        raise ValueError("need to specify 'probes' kwarg as a list of functions for computing the power in each port.")
    num_probes = len(probes)

    # create src_amplitudes
    s_list = np.logspace(np.log10(s_min), np.log10(s_max), Ns)        

    bar = progressbar.ProgressBar(max_value=2*Ns)

    # transmission
    transmissions_up = [[] for _ in range(num_probes)]
    powers = []

    (_,_,E_prev) = simulation.solve_fields()

    for i, s in enumerate(s_list):

        bar.update(i)

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
                    Estart=E_prev, solver_nl=solver, conv_threshold=1e-10,
                    max_num_iter=100)

        # compute power transmission using each probe
        for probe_index, probe in enumerate(probes):
            W_out = probe(sim_new)
            transmissions_up[probe_index].append(W_out / W_in)


    # transmission
    transmissions_down = [[] for _ in range(num_probes)]

    for i, s in enumerate(reversed(s_list)):

        bar.update(i + Ns)

        # make a new simulation object
        sim_new = copy.deepcopy(simulation)
        if omega is not None:
            sim_new.omega = omega
            sim_new.eps_r = sim_new.eps_r

        sim_new.modes[0].scale = s
        sim_new.modes[0].setup_src(sim_new)
        W_in = sim_new.W_in

        (_,_,E_prev,c) = sim_new.solve_fields_nl(timing=False, averaging=True,
                    Estart=E_prev, solver_nl=solver, conv_threshold=1e-10,
                    max_num_iter=100)

        # compute power transmission using each probe
        for probe_index, probe in enumerate(probes):
            W_out = probe(sim_new)
            transmissions_down[probe_index].append(W_out / W_in)

    transmissions_down = [list(reversed(t)) for t in transmissions_down]

    return powers, transmissions_up, transmissions_down


if __name__ == '__main__':

    fnameT = 'data/figs/devices/T_port.p'
    D = load_device(fnameT)
    detuning = -2*np.pi*80*1e9

    # print(detuning)

    powers, T_up, T_down = trans_bistable(D, omega=D.omega+detuning, Np=10, s_min=1e-1, s_max=1e3)
    for i in range(2):
        plt.plot(powers, T_up[i])
        plt.plot(powers, T_down[i])
    plt.xscale('log')
    plt.show()
