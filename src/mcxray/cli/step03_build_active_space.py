"""Step 3: splice the metal 2p core into the valence CASSCF active space.

Input : orbitals.pkl -- ORCA SA-CASSCF orbitals, validated, with the
        valence active space (Ti 3d) identified from natural-orbital
        occupations.
Output: active_space.pkl -- MOs reordered as CASCI expects, with the
        core block recorded.

The combined space is

    [ inactive | 2p (core) + valence active (3d) | virtual ]

so that the no-hole manifold (2p^6 3d^1: ground state and its ligand-field
excited states, which are the RIXS *final* states) and the core-hole
manifold (2p^5 3d^2: the XAS/RIXS *intermediate* states) are roots of one
and the same CI problem. That is what keeps every transition dipole well
defined: same orbitals, same active space, same fixed-spin FCI problem, so
transition density matrices need no biorthogonalization and no
restricted-active-space solver.

The orbitals themselves are frozen at their valence-optimized values. No
orbital relaxation in response to the core hole is included -- the standard
frozen-orbital approximation for core spectroscopy, and the main reason
absolute edge positions come out too high (see README).

2p identification is by AO character, not by orbital index. For a 3d metal
the 2p shell sits ~490 eV below everything else and is >99% metal-2p, so
this is unambiguous -- but `common.print_orbital_table` is printed anyway,
because the assumption should be checked, not trusted.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
from pyscf import gto

from mcxray import common
from mcxray import systems

N_CORE_ORB = 3          # a 2p shell
CORE_CHARACTER_MIN = 0.9


def find_core_orbitals(mol, mo_coeff, metal_label='Ti', shell='2p',
                       n=N_CORE_ORB, character_min=CORE_CHARACTER_MIN):
    """Return the `n` MOs with the highest metal-`shell` AO character.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mo_coeff : ndarray, shape (nao, nmo)
        MO coefficients in the AO basis of `mol`.
    metal_label : str
        Atomic symbol of the atom whose core shell is wanted
        (e.g. 'Ti', 'Fe').
    shell : str
        Core shell label (e.g. '2p').
    n : int
        Number of core orbitals to return.
    character_min : float
        Minimum `shell` character every selected orbital must exceed.

    Returns
    -------
    core_idx : list of int
        Indices (columns of `mo_coeff`) of the core orbitals, in ascending
        order.
    core_char : ndarray, shape (n,)
        `shell` character of each of those orbitals.

    Raises
    ------
    RuntimeError
        If any of the `n` selected orbitals has character at or below
        `character_min`.
    """
    frac = common.ao_character(mol, mo_coeff, f'{metal_label} {shell}')
    idx = np.sort(np.argsort(-frac)[:n])
    if not np.all(frac[idx] > character_min):
        raise RuntimeError(
            f'{metal_label} {shell} candidates are not clean: characters '
            f'{frac[idx].round(3).tolist()} at MOs {idx.tolist()}. Inspect '
            f'common.print_orbital_table output before proceeding.')
    return [int(i) for i in idx], frac[idx]


def build_cas_mo(mo_coeff, core_idx, valence_idx, n_inactive):
    """Reorder MOs into [inactive | core+valence active | virtual].

    Parameters
    ----------
    mo_coeff : ndarray, shape (nao, nmo)
        MO coefficients in the AO basis
    core_idx : list of int
        Indices of the core orbitals to be moved into the active space
    valence_idx : list of int
        Indices (columns of `mo_coeff`) of the valence active orbitals
    n_inactive : int
        CASSCF inactive count from the valence calculation. This still
        includes the core orbitals, which are removed from it here

    Returns
    -------
    cas_mo : ndarray, shape (nao, nmo)
        Reordered MO coefficient matrix.
    ncore : int
        Size of the inactive block, `n_inactive - len(core_idx)`.
    ncas : int
        Number of active orbitals, `len(core_idx) + len(valence_idx)`.
    core_local : list of int
        Indices of the core orbitals within the active space.
    """
    active_idx = sorted(list(core_idx) + list(valence_idx))
    inactive = [i for i in range(n_inactive) if i not in active_idx]
    nmo = mo_coeff.shape[1]
    rest = [i for i in range(nmo) if i not in inactive and i not in active_idx]
    order = inactive + active_idx + rest
    # index of the core block *within* the active space -- needed both to
    # classify CI roots and to drive the optional CVS solver
    core_local = [active_idx.index(i) for i in sorted(core_idx)]
    return mo_coeff[:, order], len(inactive), len(active_idx), core_local


if __name__ == '__main__':
    cfg = systems.from_argv(sys.argv)
    RUN_DIR, METAL = cfg['run_dir'], cfg['metal']
    with open(os.path.join(RUN_DIR, 'orbitals.pkl'), 'rb') as f:
        d = pickle.load(f)

    mol = gto.M(atom=d['atom'], basis=d['basis'], spin=d['spin'],
                charge=d['charge'], verbose=0)
    mo = d['mo_coeff']
    valence = list(d['valence_active'])
    n_inactive = d['ncore_valence']

    # Natural-orbital occupations stand in for orbital energies here: a
    # CASSCF .gbw carries NO occupations, not canonical eigenvalues.
    occ = d['mo_occ_no']

    core_idx, core_char = find_core_orbitals(mol, mo, METAL)
    print(f'--- core region ({METAL} 2p search) ---')
    common.print_orbital_table(mol, None, mo, occ, METAL,
                               range(max(0, core_idx[0] - 2), core_idx[-1] + 3))
    print(f'Selected {METAL} 2p MOs: {core_idx} '
          f'(2p character {np.round(100 * core_char, 2).tolist()} %)')

    print(f'\n--- valence active space (from ORCA SA-CASSCF) ---')
    common.print_orbital_table(mol, None, mo, occ, METAL, valence)
    print(f'Valence active MOs: {valence}')

    cas_mo, ncore, ncas, core_local = build_cas_mo(mo, core_idx, valence,
                                                   n_inactive)
    nelecas = 2 * len(core_idx) + d['nelec_valence']
    print(f'\nCombined CASCI({nelecas}e, {ncas}o): ncore={ncore}, '
          f'core block within CAS = {core_local}')

    with open(os.path.join(RUN_DIR, 'active_space.pkl'), 'wb') as f:
        pickle.dump({'atom': d['atom'], 'basis': d['basis'], 'spin': d['spin'],
                     'charge': d['charge'], 'cas_mo': cas_mo, 'ncore': ncore,
                     'ncas': ncas, 'nelecas': nelecas,
                     'cas_core_local': core_local,
                     'n_core_orb': len(core_idx),
                     'valence_active': valence,
                     'e_orca_roots': d['e_orca_roots'],
                     'e_orca_sa': d['e_orca_sa']}, f)
    print(f'Saved active space to '
          f'{os.path.relpath(os.path.join(RUN_DIR, "active_space.pkl"))}')
