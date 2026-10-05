"""
Step 4: L-edge XAS via CASCI (wavefunctions/transition dipoles) + state-
specific MC-PDFT (state energies).

Physical picture
------------------
A single CASCI(2p+3d) problem, fixed at Sz = +1/2 (na=4, nb=3 electrons
for this d1 Ti(III) example), is diagonalized for many roots. Because the
non-relativistic electronic Hamiltonian commutes with S^2, the roots sort
cleanly into pure-spin sectors even though S is not imposed explicitly
(verified below via <S^2>). The lowest 5 roots are the ground state and
its ligand-field ("d-d") excited states (2p^6 3d^1, doublet only, no 2p
hole); everything above the ~460 eV gap is a 2p^5 3d^2-like core-excited
state (some doublet, some quartet -- ONLY the doublet ones are dipole-
allowed from a doublet ground state; quartets are included and their
near-zero computed transition dipole is a self-consistency check on the
whole approach, not an assumption imposed by hand).

Energies entering the spectrum (peak positions, KH denominators) are
state-specific MC-PDFT energies. Transition dipoles are computed from the
underlying CASCI CI vectors. See scripts/common.py for why these are kept
separate.
"""
from __future__ import annotations

import os
import pickle
import numpy as np
from pyscf import gto, scf, mcscf, mcpdft
from pyscf.mcpdft import mcpdft as mcpdft_mod
from pyscf.fci import spin_op

import common

RUN_DIR = os.path.join(os.path.dirname(__file__), '..', 'run')

# Ti 2p core-hole lifetime broadening (FWHM/2 ~ typical literature value
# for 3d-metal L-edges, e.g. ~0.4 eV; adjust for your system/edge).
GAMMA_CORE_EV = 0.4


def run_casci_manifold(mf, mol, cas_mo, ncore, ncas, nelecas, nroots=30):
    na = (nelecas + 1) // 2
    nb = nelecas - na
    mc = mcscf.CASCI(mf, ncas, (na, nb))
    mc.ncore = ncore
    mc.fcisolver.nroots = nroots
    mc.kernel(cas_mo)
    return mc, (na, nb)


def state_specific_pdft_energies(mf, cas_mo, ncore, ncas, nelec, ci_list, otfnal='tPBE'):
    mcp = mcpdft.CASCI(mf, otfnal, ncas, nelec)
    mcp.ncore = ncore
    mcp.mo_coeff = cas_mo
    e_pdft = []
    for ci in ci_list:
        e_tot, _ = mcpdft_mod.energy_tot(mcp, mo_coeff=cas_mo, ci=ci, state=0)
        e_pdft.append(e_tot)
    return np.array(e_pdft)


if __name__ == '__main__':
    with open(os.path.join(RUN_DIR, 'active_space.pkl'), 'rb') as f:
        d = pickle.load(f)
    mol = gto.M(atom=d['atom'], basis=d['basis'], spin=d['spin'],
                charge=d['charge'], verbose=0)
    mf = scf.ROHF(mol)
    cas_mo, ncore, ncas = d['cas_mo'], d['ncore'], d['ncas']
    nelecas, cas_2p_local = d['nelecas'], d['cas_2p_local']

    mc, nelec = run_casci_manifold(mf, mol, cas_mo, ncore, ncas, nelecas, nroots=30)
    print(f'CASCI({nelecas}e,{ncas}o) solved for {len(mc.ci)} roots, '
          f'nelec(alpha,beta)={nelec}')

    info = common.classify_states(mc.fcisolver, mc.ci, ncas, nelec, cas_2p_local)
    ground = info['no_hole'][0]
    finals = info['no_hole'][1:]
    intermediates = info['core_hole']
    print(f'ground state idx={ground}, valence/final states={finals}, '
          f'core-excited (XAS) states={intermediates}')

    spins = [spin_op.spin_square(mc.ci[i], ncas, nelec)[1] for i in range(len(mc.ci))]
    print('multiplicities (2S+1) of core-excited states:',
          [round(spins[i], 1) for i in intermediates])

    e_pdft_all = state_specific_pdft_energies(
        mf, cas_mo, ncore, ncas, nelec, mc.ci)
    e_g_pdft = e_pdft_all[ground]

    dip_mo = common.active_space_dipole_ints(mol, cas_mo, ncore, ncas)
    mu_gn = {}
    for n in intermediates:
        mu_gn[n] = common.transition_dipole(mc.fcisolver, mc.ci[ground], mc.ci[n],
                                             ncas, nelec, dip_mo)

    de_ev = np.array([(e_pdft_all[n] - e_g_pdft) * common.HARTREE2EV
                       for n in intermediates])
    osc = np.array([common.oscillator_strength(mu_gn[n], e_pdft_all[n] - e_g_pdft)
                     for n in intermediates])
    mult = np.array([spins[n] for n in intermediates])

    print('\nXAS stick spectrum (doublet, i.e. spin-allowed, states only):')
    for e, f, m in zip(de_ev, osc, mult):
        flag = '' if m < 2.5 else '  (spin-forbidden; f should be ~0)'
        print(f'  dE = {e:7.2f} eV   f = {f:.3e}{flag}')

    grid = np.linspace(de_ev.min() - 5, de_ev.max() + 5, 4000)
    allowed = mult < 2.5  # doublet only
    spectrum = common.broaden_sticks(de_ev[allowed], osc[allowed], grid,
                                      GAMMA_CORE_EV)

    np.savez(os.path.join(RUN_DIR, 'xas_results.npz'),
             ground=ground, finals=np.array(finals),
             intermediates=np.array(intermediates),
             e_pdft_all=e_pdft_all, de_ev=de_ev, osc=osc, mult=mult,
             grid=grid, spectrum=spectrum,
             mu_gn=np.array([mu_gn[n] for n in intermediates]),
             ci=np.stack(mc.ci).astype(np.float64), nelec=nelec,
             cas_mo=cas_mo, ncore=ncore, ncas=ncas)
    print('\nSaved XAS results to run/xas_results.npz')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.stem(de_ev[allowed], osc[allowed] / osc[allowed].max(),
                basefmt=' ', markerfmt='C0o', linefmt='C0-')
        ax2 = ax.twinx()
        ax2.plot(grid, spectrum / spectrum.max(), 'C1-')
        ax.set_xlabel('Energy above ground state (eV)')
        ax.set_ylabel('Oscillator strength (norm.)')
        ax2.set_ylabel('Broadened intensity (norm.)')
        ax.set_title('Ti L-edge XAS: CASCI dipoles + state-specific MC-PDFT energies')
        fig.tight_layout()
        fig.savefig(os.path.join(RUN_DIR, 'xas_spectrum.png'), dpi=150)
        print('Saved plot to run/xas_spectrum.png')
    except ImportError:
        pass
