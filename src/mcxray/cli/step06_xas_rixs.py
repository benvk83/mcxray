"""Step 6: L-edge XAS and RIXS from the spin-orbit eigenstates.

Everything spectroscopic now runs on the SOC eigenbasis from step 5 rather
than on spin-free CASCI states. Two things follow from that:

* **L3/L2 structure exists.** Previously the two-peak XAS came purely from
  2p-3d and 3d-3d exchange splitting; there was no spin-orbit splitting at
  all. Now the 2p shell is split by zeta ~ 3.5 eV (SOMF, 2e-screened), so
  the edge separates into the familiar L3/L2 branches.

* **The doublet-only filter is gone.** The old `mult < 2.5` masks in steps
  4 and 5 discarded every quartet, which was correct while the calculation
  was spin-free -- quartets are rigorously dark then, and their computed
  oscillator strengths of 1e-13 to 1e-17 were a useful check that nothing
  was being imposed by hand. With SOC the quartets mix into the doublets
  and carry real intensity, so filtering them out would now delete physics
  rather than noise.

Transition dipoles between SOC eigenstates are complex, so |mu|^2 means
sum_x mu_x* mu_x throughout (see common.oscillator_strength), and the
Kramers-Heisenberg amplitude tensor is complex on both the absorption and
the emission step. `common.kramers_heisenberg_map` already keeps the full
3x3 Cartesian tensor uncontracted until after squaring, which is what lets
the inelastic d-d channels survive at all -- contracting it to the tensor
trace annihilates them (see that function's docstring).
"""
from __future__ import annotations

import os
import sys

import numpy as np

from mcxray import common
from mcxray import systems

GAMMA_CORE_EV = common.GAMMA_CORE_EV
GAMMA_FINAL_EV = common.GAMMA_FINAL_EV
CORE_WEIGHT_CUT = 0.5    # SO state counts as core-excited above this


def classify_so_states(core_weight, cut=CORE_WEIGHT_CUT):
    core = [k for k in range(len(core_weight)) if core_weight[k] > cut]
    valence = [k for k in range(len(core_weight)) if core_weight[k] <= cut]
    return valence, core


if __name__ == '__main__':
    cfg = systems.from_argv(sys.argv)
    RUN_DIR, METAL = cfg['run_dir'], cfg['metal']
    # --tag=NAME: spin-orbit states from soc_results_NAME.npz, outputs
    # named ..._NAME (common.tag_from_argv).
    etag = common.tag_from_argv(sys.argv)
    d = np.load(os.path.join(RUN_DIR, f'soc_results{etag}.npz'),
                allow_pickle=True)
    e_so = d['e_so']
    mu = d['mu_so_c']            # complex (n, n, 3)
    core_weight = d['core_weight']
    spin_forbidden_weight = d['spin_forbidden_weight']
    ev = common.HARTREE2EV

    valence, core = classify_so_states(core_weight)
    g = int(np.argmin(e_so))
    e_g = e_so[g]
    print(f'{len(e_so)} spin-orbit states: {len(valence)} valence, '
          f'{len(core)} core-excited; ground state index {g}')

    # ---------------- XAS ----------------------------------------------
    de = np.array([e_so[n] - e_g for n in core])
    osc = np.array([common.oscillator_strength(mu[g, n], e_so[n] - e_g)
                    for n in core])
    qw = spin_forbidden_weight[core]

    order = np.argsort(de)
    print(f'\nXAS stick spectrum (all spin-orbit states, no spin filter):')
    print(f"  {'dE(eV)':>10} {'f':>12} {'spin-forb':>11}")
    shown = [i for i in order if osc[i] > 1e-6 * osc.max()][:14]
    for i in shown:
        print(f'  {de[i] * ev:10.3f} {osc[i]:12.4e} {qw[i]:11.3f}')
    bright = osc > 0.01 * osc.max()
    print(f'  ({bright.sum()} of {len(core)} core states carry >1% of the '
          f'peak intensity; mean spin-forbidden weight among them: '
          f'{qw[bright].mean():.3f})')

    grid = np.linspace(de.min() * ev - 6, de.max() * ev + 6, 4000)
    spectrum = common.broaden_sticks(de * ev, osc, grid, GAMMA_CORE_EV)

    # ---------------- RIXS ----------------------------------------------
    finals = [g] + [f for f in valence if f != g]
    e_f = np.array([e_so[f] for f in finals])
    mu_gn = [mu[n, g] for n in core]                       # <n|mu|g>
    mu_nf = np.array([[np.conj(mu[n, f]) for f in finals]  # <f|mu|n>
                      for n in core])

    omega = np.linspace(de.min() * ev - 5, de.max() * ev + 5, 300)
    loss = np.linspace(-0.5, (e_f.max() - e_g) * ev + 1.5, 300)
    # --pol=pi|sigma [--theta=DEG]: randomly oriented sample in a definite
    # scattering geometry (common.geometry_weights). Outputs get a suffix
    # such as _pi90 so the polarization-summed ones are left alone.
    geom = common.geometry_from_argv(sys.argv)
    suffix = etag + (f'_{geom[1]}' if geom else '')
    if geom:
        weights, _, geom_text = geom
        print(f'\ngeometry: {geom_text}\n  invariant weights (G0, G1, G2) = '
              f'({weights[0]:.4f}, {weights[1]:.4f}, {weights[2]:.4f})')
        F = common.scattering_tensor(
            omega, e_g * ev, np.array([e_so[n] for n in core]) * ev,
            np.array(mu_gn), mu_nf, GAMMA_CORE_EV)
        w = sum(c * x for c, x in zip(weights,
                                      common.rotational_invariants(F)))
        lines = np.array([common.lorentzian(loss, (x - e_g) * ev,
                                            GAMMA_FINAL_EV) for x in e_f])
        rixs = lines.T @ w
    else:
        rixs = common.kramers_heisenberg_map(
            omega, loss, e_g * ev, np.array([e_so[n] for n in core]) * ev,
            e_f * ev, mu_gn, mu_nf, GAMMA_CORE_EV, GAMMA_FINAL_EV)

    print(f'\nRIXS: {len(core)} intermediates, {len(finals)} finals '
          f'(elastic + {len(finals) - 1} ligand-field)')
    for f, ef in zip(finals, e_f):
        tag = 'elastic' if f == g else 'ligand-field'
        print(f'  final {f:3d}  loss = {(ef - e_g) * ev:7.3f} eV  ({tag})')

    np.savez(os.path.join(RUN_DIR, f'spectra{suffix}.npz'),
             de_ev=de * ev, osc=osc, spin_forbidden_weight=qw, grid=grid,
             spectrum=spectrum, omega=omega, loss=loss, rixs=rixs,
             e_so=e_so, e_g=e_g)
    print(f'\nSaved spectra to '
          f'{os.path.relpath(os.path.join(RUN_DIR, f"spectra{suffix}.npz"))}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        fig, ax = plt.subplots(figsize=(7, 4.5))
        m = osc > 1e-8 * osc.max()
        ax.stem(de[m] * ev, osc[m] / osc.max(), basefmt=' ',
                markerfmt='C0o', linefmt='C0-')
        ax2 = ax.twinx()
        ax2.plot(grid, spectrum / spectrum.max(), 'C1-')
        ax.set_xlabel('Energy above ground state (eV)')
        ax.set_ylabel('Oscillator strength (norm.)')
        ax2.set_ylabel('Broadened intensity (norm.)')
        ax.set_title(f'{METAL} L-edge XAS with spin-orbit coupling\n'
                     f'CASSCF orbitals + {etag[1:] or "MC-PDFT"} energies '
                     '+ Breit-Pauli SOMF')
        fig.tight_layout()
        fig.savefig(os.path.join(RUN_DIR, f'xas_spectrum{etag}.png'), dpi=150)

        fig, ax = plt.subplots(figsize=(6.5, 5))
        floor = rixs.max() * 1e-4
        pc = ax.pcolormesh(omega, loss, rixs + floor, shading='auto',
                           cmap='viridis',
                           norm=LogNorm(vmin=floor, vmax=rixs.max()))
        fig.colorbar(pc, ax=ax, label='RIXS intensity (arb. u., log scale)')
        ax.set_xlabel('Incident photon energy above ground state (eV)')
        ax.set_ylabel('Energy loss (eV)')
        ax.set_title(f'{METAL} L-edge RIXS map (spin-orbit states'
                     + (f', {etag[1:]} energies)' if etag else ')')
                     + (f'\n{geom[2]}' if geom else ''), fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(RUN_DIR, f'rixs_map{suffix}.png'), dpi=150)
        print(f'Saved plots to {os.path.relpath(RUN_DIR)}/xas_spectrum{etag}.png '
              f'and rixs_map{suffix}.png')
    except ImportError:
        pass
