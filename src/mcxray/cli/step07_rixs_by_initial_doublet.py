"""Step 7: one RIXS map per populated Kramers doublet.

Step 6 puts the whole population in the lowest spin-orbit state. Here
each low-lying Kramers doublet in turn is taken as the populated level:
its two partner states are the initial state |i> of the
Kramers-Heisenberg expression (equal weights, intensities averaged),

  F^{beta alpha}_fi(Omega) = sum_n <f|mu_beta|n><n|mu_alpha|i>
                                   / (Omega - (E_n - E_i) + i*Gamma_n)

and the finals run over every valence spin-orbit state, as in step 6.
Incident energy and energy loss are both measured from the populated
level, so an excited initial doublet resonates at lower photon energy
and shows anti-Stokes features at negative loss (decay to states below
it). Dipoles, broadenings and the polarization sum are step 6's.

  --kmax=K   last spin-orbit state to use as an initial state
             (default 23: doublets 0-1 ... 22-23)
  --pol=pi|sigma, --theta=DEG
             experimental geometry for a randomly oriented sample (see
             common.geometry_weights); outputs get a suffix such as
             _pi90. Without --pol the nine-component sum is used.
  --tag=NAME spin-orbit states from soc_results_NAME.npz, outputs named
             ..._NAME (common.tag_from_argv)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from mcxray import common
from mcxray import systems

GAMMA_CORE_EV = common.GAMMA_CORE_EV
GAMMA_FINAL_EV = common.GAMMA_FINAL_EV
CORE_WEIGHT_CUT = 0.5
PAIR_TOL_EV = 1e-6       # Kramers partners agree to this or better


def rixs_map(i, omega, loss, e_so, mu, core, finals, weights=(1, 1, 1)):
    """KH map (nE, nO) with spin-orbit state i initial.

    `weights` multiply the three rotational invariants of the scattering
    tensor (common.geometry_weights); the default (1, 1, 1) is the
    nine-component sum of common.kramers_heisenberg_map.
    """
    F = common.scattering_tensor(omega, e_so[i], e_so[core], mu[core, i],
                                 np.conj(mu[np.ix_(core, finals)]),
                                 GAMMA_CORE_EV)
    w = sum(c * g for c, g in zip(weights, common.rotational_invariants(F)))
    lines = np.array([common.lorentzian(loss, e_so[f] - e_so[i],
                                        GAMMA_FINAL_EV) for f in finals])
    return lines.T @ w


if __name__ == '__main__':
    cfg = systems.from_argv(sys.argv)
    RUN_DIR, METAL = cfg['run_dir'], cfg['metal']
    kmax = 23
    geom = common.geometry_from_argv(sys.argv)
    weights, tag, geom_text = geom if geom else ((1, 1, 1), '', '')
    etag = common.tag_from_argv(sys.argv)
    suffix = etag + (f'_{tag}' if tag else '')
    for arg in sys.argv[1:]:
        if arg.startswith('--kmax='):
            kmax = int(arg.split('=', 1)[1])

    d = np.load(os.path.join(RUN_DIR, f'soc_results{etag}.npz'),
                allow_pickle=True)
    e_so = d['e_so'] * common.HARTREE2EV
    mu = d['mu_so_c']
    core_weight = d['core_weight']
    s2_basis = np.array([(m - 1) / 2 * ((m - 1) / 2 + 1)
                         for m in d['mult'][d['basis'][:, 0].astype(int)]])
    s2 = s2_basis @ np.abs(d['vecs']) ** 2

    core = np.flatnonzero(core_weight > CORE_WEIGHT_CUT)
    finals = np.flatnonzero(core_weight <= CORE_WEIGHT_CUT)
    g = int(np.argmin(e_so))

    # With state g initial and step 6's grids this must be step 6's map.
    sp = os.path.join(RUN_DIR, f'spectra{etag}.npz')
    if os.path.exists(sp):
        s = np.load(sp)
        ref = rixs_map(g, s['omega'], s['loss'], e_so, mu, core, finals)
        err = np.abs(ref - s['rixs']).max() / s['rixs'].max()
        print(f'initial state {g} on step-6 grids vs step-6 map: max '
              f'deviation {err:.1e} of the map maximum '
              f'({"PASS" if err < 1e-8 else "FAIL"})')

    low = [k for k in finals if k <= kmax]
    if len(low) % 2:
        raise SystemExit(f'--kmax={kmax} splits a Kramers doublet')
    pairs = [(int(low[k]), int(low[k + 1])) for k in range(0, len(low), 2)]
    e_top = max(e_so[a] for a, _ in pairs) - e_so[g]

    # One pair of grids for all maps: step 6's, extended downwards by the
    # highest initial-state energy (lower resonance, anti-Stokes losses).
    de = e_so[core] - e_so[g]
    omega = np.arange(de.min() - 5 - e_top, de.max() + 5, 0.09)
    loss = np.arange(-0.5 - e_top, e_so[finals].max() - e_so[g] + 1.5, 0.036)

    if geom:
        print(f'\ngeometry: {geom_text}\n  invariant weights (G0, G1, G2) = '
              f'({weights[0]:.4f}, {weights[1]:.4f}, {weights[2]:.4f})')
    print(f'\nRIXS maps by populated Kramers doublet: {len(core)} '
          f'intermediates, {len(finals)} finals, Gamma_n = {GAMMA_CORE_EV} '
          f'eV, Gamma_f = {GAMMA_FINAL_EV} eV')
    print(f"  {'initial':>7} {'E_i(eV)':>8} {'<S2>':>6} {'partners':>9} "
          f"{'map max':>10} {'at Omega':>9} {'loss':>7}")
    maps = []
    for a, b in pairs:
        if abs(e_so[a] - e_so[b]) > PAIR_TOL_EV:
            print(f'  WARNING: states {a},{b} differ by '
                  f'{(e_so[b] - e_so[a]) * 1e3:.4f} meV -- not a doublet?')
        ma = rixs_map(a, omega, loss, e_so, mu, core, finals, weights)
        mb = rixs_map(b, omega, loss, e_so, mu, core, finals, weights)
        m = 0.5 * (ma + mb)
        iE, iO = np.unravel_index(m.argmax(), m.shape)
        print(f'  {a:3d}-{b:<3d} {e_so[a] - e_so[g]:8.4f} {s2[a]:6.3f} '
              f'{np.abs(ma - mb).max() / m.max():9.1e} {m.max():10.3e} '
              f'{omega[iO]:9.2f} {loss[iE]:7.3f}')
        maps.append(m)
    maps = np.array(maps)
    print('  ("partners": max difference between the two partner-state '
          'maps, relative to the map maximum)')

    out = os.path.join(RUN_DIR, f'rixs_by_initial_doublet{suffix}.npz')
    np.savez(out, omega=omega, loss=loss, maps=maps, pairs=np.array(pairs),
             e_initial_ev=np.array([e_so[a] - e_so[g] for a, _ in pairs]),
             s2=s2[[a for a, _ in pairs]], invariant_weights=weights,
             geometry=geom_text)
    print(f'\nSaved maps to {os.path.relpath(out)}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        # one colour scale for all maps so they can be compared directly
        vmax = maps.max()
        floor = vmax * 1e-4
        e_fmax = e_so[finals].max()
        for (a, b), m in zip(pairs, maps):
            e_i = e_so[a] - e_so[g]
            fig, ax = plt.subplots(figsize=(6.5, 5))
            pc = ax.pcolormesh(omega, loss, m + floor, shading='auto',
                               cmap='viridis',
                               norm=LogNorm(vmin=floor, vmax=vmax))
            fig.colorbar(pc, ax=ax,
                         label='RIXS intensity (arb. u., log scale)')
            ax.set_xlim(de.min() - 5 - e_i, de.max() + 5 - e_i)
            ax.set_ylim(-0.5 - e_i, e_fmax - e_so[a] + 1.5)
            ax.set_xlabel('Incident photon energy above initial state (eV)')
            ax.set_ylabel('Energy loss (eV)')
            ax.set_title(f'{METAL} L-edge RIXS, initial states {a}-{b}\n'
                         f'{e_i:.3f} eV above ground, '
                         + (f'{etag[1:]} energies, ' if etag else '') +
                         f'<S$^2$> = {s2[a]:.2f}'
                         + (f'\n{geom_text}' if geom else ''),
                         fontsize=10 if not geom else 8)
            fig.tight_layout()
            fig.savefig(os.path.join(
                RUN_DIR, f'rixs_map_initial_{a:02d}-{b:02d}{suffix}.png'), dpi=150)
            plt.close(fig)
        print(f'Saved {len(pairs)} plots to '
              f'{os.path.relpath(RUN_DIR)}/rixs_map_initial_*{suffix}.png')
    except ImportError:
        pass
