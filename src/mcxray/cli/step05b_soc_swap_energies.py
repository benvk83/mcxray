"""Step 5b: spin-orbit states for a second set of spin-free energies.

MC-PDFT enters the spin-orbit Hamiltonian only on the diagonal: H_SO and
the transition dipoles are built from the CASCI CI vectors and do not
depend on the on-top functional. So for another functional on the same
states (e.g. tPBE0 = 0.25 E_CASCI + 0.75 E_tPBE) nothing from step 5
needs recomputing -- not the SOMF operator, not the Ms ladder, not the
SOC matrix. This script takes h_soc from soc_results.npz, replaces the
diagonal with the energies in casci_pdft_TAG.npz, diagonalizes, and
carries the dipoles over to the new eigenbasis.

  --tag=NAME   read casci_pdft_NAME.npz, write soc_results_NAME.npz
               (same keys as soc_results.npz; steps 6 and 7 take --tag)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from mcxray import common
from mcxray import soc
from mcxray import systems


if __name__ == '__main__':
    cfg = systems.from_argv(sys.argv)
    RUN_DIR = cfg['run_dir']
    etag = common.tag_from_argv(sys.argv)
    if not etag:
        raise SystemExit('--tag=NAME is required (casci_pdft_NAME.npz)')
    ev = common.HARTREE2EV

    old = np.load(os.path.join(RUN_DIR, 'soc_results.npz'), allow_pickle=True)
    new = np.load(os.path.join(RUN_DIR, f'casci_pdft{etag}.npz'),
                  allow_pickle=True)
    ref = np.load(os.path.join(RUN_DIR, 'casci_pdft.npz'), allow_pickle=True)
    mult, e_new = new['mult'], new['e_pdft']
    no_hole, core_hole = new['no_hole'].tolist(), new['core_hole'].tolist()
    h_soc, v_old = old['h_soc'], old['vecs']
    basis = [(int(i), ms) for i, ms in old['basis']]

    # h_soc is only reusable if both energy sets label the same CI vectors.
    if not (np.array_equal(mult, old['mult'])
            and np.array_equal(new['e_casci'], ref['e_casci'])
            and np.array_equal(ref['e_pdft'], old['e_pdft'])):
        raise SystemExit(f'casci_pdft{etag}.npz does not hold the states '
                         'soc_results.npz was built from')

    # stored energies must give back the stored spin-orbit spectrum
    w0, _, _ = soc.diagonalize(old['e_pdft'], mult, h_soc, basis)
    err = np.abs(w0 - old['e_so']).max()
    print(f'stored energies + stored h_soc vs stored e_so: {err:.2e} Ha '
          f'({"PASS" if err < 1e-9 else "FAIL"})')
    if err >= 1e-9:
        raise SystemExit('soc_results.npz is not self-consistent')

    w, v, herm = soc.diagonalize(e_new, mult, h_soc, basis)
    print(f'{etag[1:]} energies: Hermiticity residual {herm:.2e}')
    e0 = w.min()
    print(f'  ground SO state at {e0:.8f} Ha; SOC stabilizes the ground '
          f'state by {(e_new[no_hole[0]] - e0) * ev * 1000:.1f} meV')

    # Dipoles: old eigenbasis -> (state, Ms) basis -> new eigenbasis.
    u = v_old.conj().T @ v
    mu_so = np.stack([u.conj().T @ old['mu_so_c'][:, :, x] @ u
                      for x in range(3)], axis=-1)

    is_core = np.zeros(len(mult), bool)
    is_core[core_hole] = True
    basis_state = np.array([i for i, _ in basis])
    basis_mult = np.round(mult[basis_state]).astype(int)
    pop = np.abs(v) ** 2
    core_weight = pop[is_core[basis_state]].sum(axis=0)
    mult_gs = int(old['mult_gs'])
    spin_forbidden_weight = pop[basis_mult != mult_gs].sum(axis=0)
    quartet_weight = pop[basis_mult == 4].sum(axis=0)

    out = os.path.join(RUN_DIR, f'soc_results{etag}.npz')
    np.savez(out, e_so=w, vecs=v, mu_so=np.abs(mu_so), mu_so_c=mu_so,
             basis=old['basis'], core_weight=core_weight,
             quartet_weight=quartet_weight,
             spin_forbidden_weight=spin_forbidden_weight,
             mult_gs=mult_gs, h_soc=h_soc,
             e_pdft=e_new, mult=mult, e_ground=e0)
    print(f'\nSaved {len(w)} spin-orbit states to {os.path.relpath(out)}')

    val = np.flatnonzero(core_weight <= 0.5)
    print(f'\n  lowest valence SO states, {etag[1:]} vs soc_results.npz (eV):')
    for k in val[:24:2]:
        print(f'  {k:4d} {(w[k] - e0) * ev:8.4f} '
              f'{(old["e_so"][k] - old["e_so"].min()) * ev:8.4f}')
    core = np.flatnonzero(core_weight > 0.5)
    print(f'  core-excited manifold: {(w[core].min() - e0) * ev:.3f} - '
          f'{(w[core].max() - e0) * ev:.3f} eV '
          f'(was {(old["e_so"][core].min() - old["e_so"].min()) * ev:.3f} - '
          f'{(old["e_so"][core].max() - old["e_so"].min()) * ev:.3f})')
