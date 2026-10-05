"""
Step 5: L-edge RIXS via the Kramers-Heisenberg formula.

Uses the CASCI CI vectors and state-specific MC-PDFT energies produced by
04_xas_casci_pdft.py. Only the doublet ("spin-allowed") core-excited
states can appear as intermediates for a doublet ground state (verified,
not assumed, in step 4); the "final" states are the doublet valence
(ligand-field) excited states from the same CASCI diagononalization.

Orientation averaging: this uses the common "sum over Cartesian dipole
components" isotropic/powder approximation (see common.py docstring for
the exact formula and its limits) rather than a full angular/tensor
average -- adequate for locating RIXS resonances and estimating relative
intensities, not for polarization-dependent single-crystal work.
"""
from __future__ import annotations

import os
import numpy as np
from pyscf import gto, scf

import common

RUN_DIR = os.path.join(os.path.dirname(__file__), '..', 'run')

GAMMA_CORE_EV = 0.4    # core-hole lifetime broadening (intermediate states)
GAMMA_FINAL_EV = 0.15  # final-state / instrumental broadening


if __name__ == '__main__':
    data = np.load(os.path.join(RUN_DIR, 'xas_results.npz'), allow_pickle=True)
    ground = int(data['ground'])
    finals = data['finals'].tolist()
    intermediates = data['intermediates'].tolist()
    e_pdft_all = data['e_pdft_all']
    mult = data['mult']
    ci = list(data['ci'])
    nelec = tuple(data['nelec'])
    cas_mo, ncore, ncas = data['cas_mo'], int(data['ncore']), int(data['ncas'])

    # only spin-allowed (doublet) intermediates can contribute to a
    # doublet-ground-state RIXS process
    doublet_intermediates = [n for n, m in zip(intermediates, mult) if m < 2.5]

    # include the elastic channel (final state = ground state) alongside
    # the inelastic (ligand-field-excited) final states
    finals = [ground] + finals

    import pickle
    with open(os.path.join(RUN_DIR, 'active_space.pkl'), 'rb') as f:
        d = pickle.load(f)
    mol = gto.M(atom=d['atom'], basis=d['basis'], spin=d['spin'],
                charge=d['charge'], verbose=0)
    from pyscf.fci import direct_spin1
    fcisolver = direct_spin1.FCISolver(mol)

    dip_mo = common.active_space_dipole_ints(mol, cas_mo, ncore, ncas)

    mu_gn = {n: common.transition_dipole(fcisolver, ci[ground], ci[n],
                                          ncas, nelec, dip_mo)
             for n in doublet_intermediates}
    mu_nf = {(n, f): common.transition_dipole(fcisolver, ci[n], ci[f],
                                               ncas, nelec, dip_mo)
             for n in doublet_intermediates for f in finals}

    e_g = e_pdft_all[ground]
    e_n_list = [e_pdft_all[n] for n in doublet_intermediates]
    e_f_list = [e_pdft_all[f] for f in finals]

    mu_gn_list = [mu_gn[n] for n in doublet_intermediates]
    mu_nf_matrix = np.array([[mu_nf[(n, f)] for f in finals]
                              for n in doublet_intermediates])

    print('RIXS final (ligand-field) states, energy loss from ground state (eV):')
    for f in finals:
        print(f'  idx={f}  loss = {(e_pdft_all[f]-e_g)*common.HARTREE2EV:.3f} eV')

    print('\nRIXS intermediate (core-excited, spin-allowed) states, resonance (eV):')
    for n in doublet_intermediates:
        print(f'  idx={n}  resonance = {(e_pdft_all[n]-e_g)*common.HARTREE2EV:.3f} eV, '
              f'|mu_gn| = {np.linalg.norm(mu_gn[n]):.4f}')

    omega_in_grid = np.linspace(
        (min(e_n_list) - e_g) * common.HARTREE2EV - 5,
        (max(e_n_list) - e_g) * common.HARTREE2EV + 5, 300)
    loss_grid = np.linspace(-0.5, (max(e_f_list) - e_g) * common.HARTREE2EV + 1.5, 300)

    rixs_map = common.kramers_heisenberg_map(
        omega_in_grid, loss_grid,
        e_g * common.HARTREE2EV,
        np.array(e_n_list) * common.HARTREE2EV,
        np.array(e_f_list) * common.HARTREE2EV,
        mu_gn_list, mu_nf_matrix,
        GAMMA_CORE_EV, GAMMA_FINAL_EV)

    np.savez(os.path.join(RUN_DIR, 'rixs_results.npz'),
              omega_in_grid=omega_in_grid, loss_grid=loss_grid, rixs_map=rixs_map)
    print('\nSaved RIXS map to run/rixs_results.npz')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        fig, ax = plt.subplots(figsize=(6, 5))
        floor = rixs_map.max() * 1e-4
        pc = ax.pcolormesh(omega_in_grid, loss_grid, rixs_map + floor,
                            shading='auto', cmap='viridis',
                            norm=LogNorm(vmin=floor, vmax=rixs_map.max()))
        fig.colorbar(pc, ax=ax, label='RIXS intensity (arb. u., log scale)')
        ax.set_xlabel('Incident photon energy above ground state (eV)')
        ax.set_ylabel('Energy loss (eV)')
        ax.set_title('Ti L-edge RIXS map (CASCI dipoles + MC-PDFT energies)')
        fig.tight_layout()
        fig.savefig(os.path.join(RUN_DIR, 'rixs_map.png'), dpi=150)
        print('Saved plot to run/rixs_map.png')
    except ImportError:
        pass
