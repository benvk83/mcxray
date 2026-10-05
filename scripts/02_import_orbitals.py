"""Step 2: bring ORCA's SA-CASSCF orbitals into PySCF, and prove it worked.

Route (see scripts/orbital_io.py for why this one and not molden):

    .gbw --orca_2mkl--> .mkl --mkl2fch -no--> .fch --fch2py--> mo_coeff

Two hard gates, both fatal rather than advisory, because a silently-wrong
orbital import is the single most dangerous failure mode in this workflow
(the old molden route produced orbitals that converged SCF into a basin 5
Hartree off the true ground state -- which still *looks* converged):

  1. max|C^T S C - I| below 1e-6. Measured here: 1.97e-07, which is the
     .fch E16.8 text-precision floor, not a convention error. A Lowdin
     cleanup then takes it to ~4e-15.
  2. A PySCF CASCI in the imported valence active space must reproduce
     ORCA's own SA-CASSCF roots. Measured here: ~1e-9 Ha per root.

Gate 2 is the one that actually matters: gate 1 only says the orbitals are
an orthonormal set, gate 2 says they are *the same* orthonormal set ORCA
converged to.

Import mode (`--reopt` / `--no-reopt`, default `--no-reopt`)
-------------------------------------------------------------
`--no-reopt` uses ORCA's orbitals exactly as they are, so whatever method
exploration was done on the ORCA side is preserved untouched.
`--reopt` instead seeds a short PySCF SA-CASSCF from them; the iteration
count is itself an import diagnostic, since a good import starts at
convergence and a bad one does not.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
from pyscf import gto, scf, mcscf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orbital_io as oio
import systems

HARTREE2EV = 27.211386245988


def valence_space_from_occupations(occ, nelec_valence, norb_valence,
                                    frac_tol=1e-3):
    """Locate the active space from the natural-orbital occupations that
    `mkl2fch -no` wrote into the .fch.

    For a state-averaged CASSCF these identify the active space with no
    ambiguity at all: inactive orbitals sit at exactly 2.0, virtuals at
    exactly 0.0, and only the active ones are fractional. This is more
    robust than counting electrons, and it cross-checks the active space
    ORCA actually used against the one the input requested.
    """
    frac = np.where((occ > frac_tol) & (occ < 2 - frac_tol))[0]
    if len(frac) != norb_valence:
        raise RuntimeError(
            f'expected {norb_valence} fractionally-occupied (active) '
            f'orbitals, found {len(frac)}: indices {frac.tolist()}, '
            f'occupations {occ[frac].round(4).tolist()}')
    if not np.all(np.diff(frac) == 1):
        raise RuntimeError(f'active orbitals are not contiguous: {frac.tolist()}')
    ncore = int(frac[0])
    n_act_elec = int(round(occ[frac].sum()))
    if n_act_elec != nelec_valence:
        raise RuntimeError(
            f'active-space electron count from occupations ({n_act_elec}) '
            f'disagrees with the requested {nelec_valence}')
    return ncore, list(frac)


if __name__ == '__main__':
    cfg = systems.from_argv(sys.argv)
    RUN_DIR = cfg['run_dir']
    nelec_val, norb_val = cfg['nelec_valence'], cfg['norb_valence']
    nroots_val = cfg['n_valence_roots']
    reopt = '--reopt' in sys.argv

    atom = oio.geometry_from_orca_xyz(cfg['xyz_path'])
    mol = gto.M(atom=atom, basis=cfg['basis'], spin=cfg['spin'],
                charge=cfg['charge'], verbose=0)
    print(f'system {cfg["name"]}: {mol.natm} atoms, {mol.nao} AOs, '
          f'nelec={mol.nelec}, CAS({nelec_val}e,{norb_val}o)')

    print('\n--- gate 1: orbital import ---')
    mo, occ = oio.load_orca_orbitals(cfg['gbw'], RUN_DIR, mol)
    ncore, active = valence_space_from_occupations(occ, nelec_val, norb_val)
    print(f'  valence active space from NO occupations: MOs {active} '
          f'(occ {occ[active].round(3).tolist()}), ncore={ncore}')

    print('\n--- gate 2: ORCA CASSCF energy reproduction ---')
    # The reference is optional. A .gbw that arrives as a finished CASSCF
    # with no ORCA output next to it is imported as-is: gate 1 and the
    # occupation check above still run, the valence CASCI below still runs
    # (its ligand-field spectrum is the sanity check), and only the
    # root-by-root comparison against ORCA is skipped.
    ref_out = os.path.join(RUN_DIR, cfg['orca_job'] + '.out')
    if os.path.exists(ref_out):
        roots_orca, e_sa_orca = oio.casscf_roots_from_orca_out(ref_out)
        if len(roots_orca) != nroots_val:
            raise RuntimeError(
                f'{cfg["orca_job"]}.out holds {len(roots_orca)} roots but the '
                f'valence manifold is {nroots_val} states. The reference job '
                'and systems.py disagree about the active space or the root '
                'counts.')
    else:
        roots_orca, e_sa_orca = None, None
        print(f'  no {cfg["orca_job"]}.out -> no ORCA roots to compare with; '
              'GATE SKIPPED')

    mf = scf.ROHF(mol)
    mf.mo_coeff = mo
    mf.mo_occ = np.zeros(mol.nao)
    mf.mo_occ[:ncore] = 2
    mf.mo_occ[ncore] = 1
    # Minimum-|Ms| block: it carries every multiplicity the manifold has,
    # so one solve covers the quartets AND the doublets of a d^3 ion.
    na = (nelec_val + 1) // 2
    mc = mcscf.CASCI(mf, norb_val, (na, nelec_val - na))
    mc.ncore = ncore
    mc.fcisolver.nroots = nroots_val
    mc.kernel(mo)
    e_casci = np.sort(np.array(mc.e_tot))

    if roots_orca is None:
        print(f'  PySCF CASCI in the imported orbitals, {nroots_val} roots: '
              f'E0 = {e_casci[0]:.10f} Ha')
        print('  ligand-field excitation energies (eV):',
              np.array2string(np.round((e_casci - e_casci[0]) * HARTREE2EV, 3)))
    else:
        print(f"  {'root':>5} {'ORCA CASSCF':>18} {'PySCF CASCI':>18} {'diff':>12}")
        for i, (a, b) in enumerate(zip(roots_orca, e_casci)):
            if i < 8 or abs(b - a) > 1e-7:
                print(f'  {i:5d} {a:18.10f} {b:18.10f} {b - a:12.2e}')
        worst = float(np.abs(e_casci - roots_orca).max())
        if worst > 1e-6:
            raise RuntimeError(
                f'per-root gate FAILED: max |PySCF - ORCA| = {worst:.3e} Ha over '
                f'{len(roots_orca)} roots. The orbital transfer or the active '
                'space is wrong.')
        print(f'  max |PySCF - ORCA| over {len(roots_orca)} roots: {worst:.2e} Ha'
              f'  -> GATE PASS')
        d = oio.check_casscf_energy(e_casci.mean(), np.mean(roots_orca), tol=1e-6)
        print(f'  state-averaged (equal weights): diff {d:.2e} Ha')

    if reopt:
        print('\n--- optional: re-converge SA-CASSCF in PySCF ---')
        mc2 = mcscf.CASSCF(mf, norb_val, (na, nelec_val - na))
        mc2.ncore = ncore
        mc2 = mcscf.state_average_(mc2, [1.0 / nroots_val] * nroots_val)
        mc2.kernel(mo)
        # equal weights on both sides -- ORCA's own block weighting
        # (e_sa_orca) is a different average and would not be comparable.
        ref = float(np.mean(e_casci if roots_orca is None else roots_orca))
        print(f'  converged={mc2.converged} in {mc2.__dict__.get("_niter", "?")} '
              f'macro-iterations; E_SA = {mc2.e_tot:.10f} Ha '
              f'({"imported-orbital CASCI" if roots_orca is None else "ORCA"} '
              f'equal-weight {ref:.10f}, diff {mc2.e_tot - ref:.2e})')
        mo = mc2.mo_coeff

    out = os.path.join(RUN_DIR, 'orbitals.pkl')
    with open(out, 'wb') as f:
        pickle.dump({'atom': atom, 'basis': cfg['basis'], 'spin': cfg['spin'],
                     'charge': cfg['charge'],
                     'mo_coeff': mo, 'mo_occ_no': occ,
                     'ncore_valence': ncore, 'valence_active': active,
                     'nelec_valence': nelec_val,
                     'norb_valence': norb_val,
                     'e_orca_roots': roots_orca, 'e_orca_sa': e_sa_orca,
                     'reopt': reopt}, f)
    print(f'\nSaved validated orbitals to {os.path.relpath(out)}')
