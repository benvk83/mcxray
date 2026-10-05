"""
Step 3: Select the L-edge active space: metal 2p (core, to be ionized) plus
metal 3d (valence, the d-manifold that both hosts the XAS final-state hole
and gives the ligand-field/RIXS final states).

Active-space choice for this workflow
--------------------------------------
Both the "no-hole" states (ground state + valence/ligand-field excited
states, used as RIXS final states) and the "core-hole" states (XAS/RIXS
intermediate states) are computed as roots of a *single* CASCI(2p+3d)
problem. This is deliberate: transition dipoles are only well-defined
between CI vectors that live in the same active space and the same
fixed-spin FCI problem (see scripts/common.py docstring). Putting both
manifolds in one CAS space, and letting the CI diagonalization itself sort
out which determinants dominate which root, avoids needing a restricted
active space (RAS/GAS) solver.

Selection procedure
--------------------
2p (core) orbitals are trivially separable by orbital energy -- for a 3d
transition metal they sit ~450-500 eV below anything else and are >99% Ti
2p in AO character (verified below). 3d (valence) orbitals are selected by
AO character among the frontier occupied/virtual MOs; because they are the
antibonding metal-ligand combinations, they are NOT 100% metal-d (that is
expected -- it is exactly the covalency an active space is supposed to
capture) -- ALWAYS eyeball common.print_orbital_table's output rather than
trusting a hard cutoff blindly.
"""
from __future__ import annotations

import os
import pickle
import numpy as np
from pyscf import gto

import common


def find_core_2p(mol, mo_energy, mo_coeff, metal_label, n=3):
    """2p orbitals: lowest-energy occupied MOs with dominant metal-2p
    character. Robust because of the huge energy gap (~450 eV) separating
    them from everything else -- no ambiguity expected.
    """
    frac = common.ao_character(mol, mo_coeff, f'{metal_label} 2p')
    candidates = np.argsort(-frac)[:n]
    candidates = np.sort(candidates)
    assert np.all(frac[candidates] > 0.9), (
        f'2p candidates do not look clean: {frac[candidates]}')
    return list(candidates)


def find_valence_3d(mol, mo_energy, mo_coeff, mo_occ, metal_label, n=5,
                     energy_window_ev=(-20, 20)):
    """3d-like orbitals: among frontier MOs (near the HOMO/LUMO region),
    rank by metal-3d AO character and take the top `n`. Intended to be
    used interactively -- print the candidate table and confirm/override.
    """
    frac = common.ao_character(mol, mo_coeff, f'{metal_label} 3d')
    ev = mo_energy * common.HARTREE2EV
    somo_e = mo_energy[mo_occ == 1][0] * common.HARTREE2EV if np.any(mo_occ == 1) else 0.0
    window = (ev > somo_e + energy_window_ev[0]) & (ev < somo_e + energy_window_ev[1])
    idx_window = np.where(window)[0]
    ranked = idx_window[np.argsort(-frac[idx_window])]
    return list(np.sort(ranked[:n]))


def build_cas_mo(mo_coeff, core_2p_idx, valence_3d_idx, mo_occ):
    """Reorder MOs into [core (all doubly-occ not in active) | active
    (2p + 3d) | remaining virtuals], as CASCI expects."""
    active_idx = sorted(core_2p_idx + valence_3d_idx)
    doubly_occ = set(np.where(mo_occ == 2)[0].tolist())
    core_idx = sorted(doubly_occ - set(active_idx))
    nmo = mo_coeff.shape[1]
    rest_idx = [i for i in range(nmo) if i not in core_idx and i not in active_idx]
    order = core_idx + active_idx + rest_idx
    ncore = len(core_idx)
    ncas = len(active_idx)
    # index of the 2p block *within* the active space (needed later to
    # classify core-hole vs no-hole CI roots)
    cas_2p_local = [active_idx.index(i) for i in sorted(core_2p_idx)]
    return mo_coeff[:, order], ncore, ncas, cas_2p_local


if __name__ == '__main__':
    run_dir = os.path.join(os.path.dirname(__file__), '..', 'run')
    with open(os.path.join(run_dir, 'mf_ticl3.pkl'), 'rb') as f:
        d = pickle.load(f)

    mol = gto.M(atom=d['atom'], basis=d['basis'], spin=d['spin'],
                charge=d['charge'], verbose=0)
    mo_coeff, mo_energy, mo_occ = d['mo_coeff'], d['mo_energy'], d['mo_occ']

    print('--- core region (2p search) ---')
    common.print_orbital_table(mol, mo_energy, mo_coeff, mo_occ, 'Ti', range(4, 9))
    core_2p = find_core_2p(mol, mo_energy, mo_coeff, 'Ti')
    print('Selected 2p indices:', core_2p)

    print('--- frontier region (3d search) ---')
    common.print_orbital_table(mol, mo_energy, mo_coeff, mo_occ, 'Ti', range(30, 45))
    valence_3d = find_valence_3d(mol, mo_energy, mo_coeff, mo_occ, 'Ti')
    print('Selected 3d indices:', valence_3d)

    cas_mo, ncore, ncas, cas_2p_local = build_cas_mo(mo_coeff, core_2p, valence_3d, mo_occ)
    nelecas = int(round(mo_occ[core_2p + valence_3d].sum()))
    print(f'ncore={ncore}, ncas={ncas}, nelecas={nelecas}, '
          f'2p local indices in CAS={cas_2p_local}')

    with open(os.path.join(run_dir, 'active_space.pkl'), 'wb') as f:
        pickle.dump({'atom': d['atom'], 'basis': d['basis'], 'spin': d['spin'],
                     'charge': d['charge'], 'cas_mo': cas_mo, 'ncore': ncore,
                     'ncas': ncas, 'nelecas': nelecas,
                     'cas_2p_local': cas_2p_local}, f)
    print('Saved active space to run/active_space.pkl')
