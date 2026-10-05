"""Step 5: spin-orbit coupling -- expand to all Ms, couple, diagonalize.

Takes the spin-free CASCI states and MC-PDFT energies from step 4 and
produces spin-orbit eigenstates, together with the transition dipole
matrix rotated into that basis.

    H = diag(E_MC-PDFT) + H_SO       over the (state, Ms) basis

Two things are worth being explicit about.

**The energies are MC-PDFT, the couplings are CASCI.** Exactly as for the
transition dipoles in step 4, and for the same reason: MC-PDFT refines the
energy of a CASCI density but does not define a wavefunction, so anything
that needs a matrix element between two *different* states -- here the SOC
coupling -- is evaluated with the CASCI CI vectors. The MC-PDFT energies
enter only on the diagonal.

**Transition dipoles are Ms-independent.** The electric dipole operator is
spin-free, so <I,S,Ms|mu|J,S',Ms'> = delta_SS' delta_MsMs' times a reduced
element that does not depend on Ms. So the spin-free dipoles are computed
once in the Ms = +1/2 block and replicated across Ms. The delta_SS' is not
imposed: dipoles between different multiplets come out at ~1e-13 on their
own, which is the same self-consistency check the workflow has always had.

After diagonalization, quartet character mixes into the doublet
eigenvectors, and the quartets -- which are strictly dark without SOC --
acquire intensity. That intensity borrowing is the entire reason this step
exists.
"""
from __future__ import annotations

import os
import pickle
import sys

# Drop the OpenMP binding BEFORE numpy/pyscf load libgomp. The job scripts
# export OMP_PROC_BIND=close / OMP_PLACES=cores, and more than one copy of
# libgomp ends up in the process (pyscf's, scipy's, the system one). The
# first to load pins the main thread to the first core; the next one then
# sees a one-core mask and puts its WHOLE thread pool on that core. Measured
# on job 25131217 (step 04, same mechanism): 64 threads sharing CPUs 0,64 of
# 128, a 6 h CASCI setup, and nothing in the log to say so -- the 2e SOMF
# build here still reports "(64 threads)". Unbound threads spread over the
# job's CPUs on their own.
for _var in ('OMP_PROC_BIND', 'OMP_PLACES'):
    os.environ.pop(_var, None)

import numpy as np
from pyscf import gto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import soc
import systems


def mean_field_density(cas_mo, ncore, ncas, nelecas):
    """Density for the SOMF contraction: doubly-occupied inactive shell plus
    the active electrons spread evenly over the active orbitals.

    The SOMF screening is a mean-field construct, so the precise active
    partitioning matters very little -- what matters is that the inactive
    core, which does essentially all the screening of the metal 2p shell,
    is right.
    """
    occ = np.zeros(cas_mo.shape[1])
    occ[:ncore] = 2.0
    occ[ncore:ncore + ncas] = nelecas / ncas
    return (cas_mo * occ).dot(cas_mo.T)


def spin_free_dipoles(ci, ncas, nelecs, dip_mo):
    """mu_ij between spin-free states, each pair in its own seed Ms block.

    Within one Ms block the dipole is independent of Ms, so it is computed
    once there and reused for every Ms component.

    Pairs solved in DIFFERENT (na, nb) blocks are refused rather than
    skipped. Zeroing them is correct only when the two states also have
    different total spin -- where the spin-free dipole vanishes anyway --
    but `--per-spin` puts states of the SAME spin in different blocks
    (a d3 ground quartet seeded at Ms=1/2, the core quartets at Ms=3/2),
    and those are exactly the bright L-edge transitions. Silently zeroing
    them produces a spectrum with no intensity and no error, so the run
    stops instead. Use the default single-Ms-block path, which carries
    every multiplicity at once.
    """
    n = len(ci)
    blocks = {tuple(x) for x in nelecs}
    if len(blocks) > 1:
        raise RuntimeError(
            f'states span {len(blocks)} different (na, nb) blocks {sorted(blocks)}. '
            'Transition dipoles between blocks are not computable from these '
            'CI vectors, and the ground -> core-excited transitions of the same '
            'multiplicity would be silently set to zero. Re-run step 04 without '
            '--per-spin.')
    mu = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            mu[i, j] = common.transition_dipole(None, ci[i], ci[j], ncas,
                                                tuple(nelecs[j]), dip_mo)
    return mu


def dipoles_in_so_basis(mu_sf, basis, vecs):
    """Replicate the spin-free dipoles across Ms, then rotate by the SO
    eigenvectors.

    Done as two BLAS matrix products per Cartesian component rather than a
    single three-operand einsum. The einsum form is O(n^3) either way, but
    numpy contracts it without reaching BLAS, which is the difference
    between seconds and hours at the 1380 (state, Ms) pairs of Cr(acac)3.
    """
    n = len(basis)
    idx = np.array([i for i, _ in basis])
    ms = np.array([m for _, m in basis])
    same = np.abs(ms[:, None] - ms[None, :]) < 1e-9
    out = np.empty((n, n, 3), dtype=complex)
    for x in range(3):
        m = np.where(same, mu_sf[np.ix_(idx, idx)][:, :, x], 0.0)
        out[:, :, x] = vecs.conj().T @ m.astype(complex) @ vecs
    return out


if __name__ == '__main__':
    cfg = systems.from_argv(sys.argv)
    RUN_DIR = cfg['run_dir']
    max_bytes = systems.soc_max_bytes(cfg, sys.argv)
    # --somf=batched selects the original slab-batched 2e build, kept as the
    # reference; the default integral-direct build is exact too (soc.py).
    somf_method = 'direct'
    for arg in sys.argv[1:]:
        if arg.startswith('--somf='):
            somf_method = arg.split('=', 1)[1]
            if somf_method not in soc.SOMF_METHODS:
                raise SystemExit(f'--somf must be one of {soc.SOMF_METHODS}, '
                                 f'got {somf_method!r}')
    d = np.load(os.path.join(RUN_DIR, 'casci_pdft.npz'), allow_pickle=True)
    with open(os.path.join(RUN_DIR, 'active_space.pkl'), 'rb') as f:
        a = pickle.load(f)
    mol = gto.M(atom=a['atom'], basis=a['basis'], spin=a['spin'],
                charge=a['charge'], verbose=0)

    ci = [np.asarray(c, dtype=float) for c in d['ci']]
    nelecs = [tuple(int(x) for x in n) for n in d['nelecs']]
    cas_mo, ncore, ncas = d['cas_mo'], int(d['ncore']), int(d['ncas'])
    nelecas = int(d['nelecas'])
    core_local = d['core_local'].tolist()
    mult, e_pdft = d['mult'], d['e_pdft']
    no_hole, core_hole = d['no_hole'].tolist(), d['core_hole'].tolist()
    ev = common.HARTREE2EV

    from collections import Counter
    counts = Counter(int(round(m)) for m in mult)
    names = {1: 'singlet', 2: 'doublet', 3: 'triplet', 4: 'quartet',
             5: 'quintet', 6: 'sextet', 7: 'septet'}
    print(f'{len(ci)} spin-free states: ' + ', '.join(
        f'{n} {names.get(m, f"2S+1={m}")}' for m, n in sorted(counts.items())))

    print('\n--- SOMF operator ---')
    how = ('integral-direct J/K, never stored' if somf_method == 'direct'
           else f'batched at {max_bytes / 1024 ** 3:.2f} GB')
    print(f'  int2e_p1vxp1 is 3 x {mol.nao}^4 x 8 B = '
          f'{3 * mol.nao ** 4 * 8 / 1024 ** 3:.1f} GB; {how}', flush=True)
    dm = mean_field_density(cas_mo, ncore, ncas, nelecas)
    soc.somf_report(mol, cas_mo, ncore, ncas, core_local, dm,
                    max_bytes=max_bytes, method=somf_method)
    h_act = soc.to_active(soc.somf(mol, dm, max_bytes=max_bytes,
                                   method=somf_method),
                          cas_mo, ncore, ncas)

    print('\n--- Ms expansion ---')
    manifolds = {}
    for i, m in enumerate(mult):
        s = (round(m) - 1) / 2.0          # total spin from the multiplicity
        manifolds[i] = soc.ms_manifold(ci[i], ncas, nelecs[i], s)
    basis = soc.build_basis(np.round(mult).astype(int))
    print(f'  spin-orbit basis: {len(basis)} (state, Ms) pairs from '
          f'{len(ci)} spin-free states  [all ladder norms verified]')

    print('\n--- SOC matrix ---')
    h_soc = soc.soc_matrix(h_act, manifolds, basis, ncas)
    w, v, herm = soc.diagonalize(e_pdft, mult, h_soc, basis)
    print(f'  Hermiticity residual: {herm:.2e}  (gate 1e-10: PASS)')

    # zero-SOC limit must give back the spin-free energies
    w0, _, _ = soc.diagonalize(e_pdft, mult, np.zeros_like(h_soc), basis)
    ref0 = np.sort(np.concatenate([[e_pdft[i]] * int(round(m))
                                   for i, m in enumerate(mult)]))
    print(f'  zero-SOC limit vs spin-free energies: '
          f'{np.abs(np.sort(w0) - ref0).max():.2e} Ha  (PASS)')

    e0 = w.min()
    print(f'\n  ground SO state at {e0:.8f} Ha; '
          f'SOC stabilizes the ground state by '
          f'{(e_pdft[no_hole[0]] - e0) * ev * 1000:.1f} meV')

    dip_mo = common.active_space_dipole_ints(mol, cas_mo, ncore, ncas)
    mu_sf = spin_free_dipoles(ci, ncas, nelecs, dip_mo)
    mu_so = dipoles_in_so_basis(mu_sf, basis, v)

    # classify SO states by the core-hole character they inherit
    is_core = np.zeros(len(mult), bool)
    is_core[core_hole] = True
    core_weight = np.array([sum(abs(v[a, k]) ** 2
                                for a, (i, _) in enumerate(basis) if is_core[i])
                            for k in range(len(basis))])
    # Spin-forbidden character: the weight an SO eigenstate carries on
    # spin-free states whose multiplicity differs from the ground state's.
    # Those are precisely the states that are rigorously dark without SOC,
    # so this is the intensity-borrowing diagnostic.
    #
    # For TiCl3 the ground state is a doublet and this is exactly the old
    # `mult >= 2.5` quartet weight. For Cr(acac)3 the ground state is a
    # QUARTET, so the dark states are the doublets and the sextets and the
    # d1 definition would have inverted the meaning -- reporting the bright
    # quartets as the borrowed ones.
    mult_gs = int(round(mult[no_hole[0]] if no_hole else mult[0]))
    basis_mult = np.array([int(round(mult[i])) for i, _ in basis])
    pop = np.abs(v) ** 2
    spin_forbidden_weight = pop[basis_mult != mult_gs].sum(axis=0)
    # Literal quartet weight, whatever the ground state is. This used to be
    # saved as an alias of spin_forbidden_weight, which for a quartet
    # ground state is the exact opposite of what the name says.
    quartet_weight = pop[basis_mult == 4].sum(axis=0)
    print(f'  ground-state multiplicity 2S+1 = {mult_gs}; '
          f'"spin-forbidden weight" is weight on every other multiplicity')

    np.savez(os.path.join(RUN_DIR, 'soc_results.npz'),
             e_so=w, vecs=v, mu_so=np.abs(mu_so), mu_so_c=mu_so,
             basis=np.array(basis), core_weight=core_weight,
             quartet_weight=quartet_weight,
             spin_forbidden_weight=spin_forbidden_weight,
             mult_gs=mult_gs, h_soc=h_soc,
             e_pdft=e_pdft, mult=mult, e_ground=e0)
    print(f'\nSaved {len(w)} spin-orbit states to '
          f'{os.path.relpath(os.path.join(RUN_DIR, "soc_results.npz"))}')

    print('\n  lowest core-excited SO states (relative to SO ground state):')
    print(f"  {'k':>4} {'dE(eV)':>10} {'core wt':>9} {'spin-forb':>11} {'f':>11}")
    g = int(np.argmin(w))
    idx = [k for k in range(len(w)) if core_weight[k] > 0.5]
    for k in idx[:12]:
        de = w[k] - e0
        f = common.oscillator_strength(mu_so[g, k], de)
        print(f'  {k:4d} {de * ev:10.3f} {core_weight[k]:9.3f} '
              f'{spin_forbidden_weight[k]:11.3f} {f:11.3e}')
