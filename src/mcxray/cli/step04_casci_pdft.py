"""Step 4: CASCI over the combined 2p+3d space, with MC-PDFT state energies.

Spin blocks
------------
The L-edge problem needs more than one multiplicity. For a d1 ion the
2p^5 3d^2 core-excited manifold carries both doublets and quartets, and
with SA-CASSCF orbitals the QUARTETS come out as the LOWEST core-excited
states (Hund's rule). They are rigorously dark without spin-orbit coupling,
which is why the previous version of this workflow discarded them; with
SOC they borrow intensity and become a real part of the spectrum, so they
have to be kept.

For d3 (Cr(III)) there is a third multiplicity. The valence 2p^6 3d^3
manifold is 10 quartets + 40 doublets, and the core-excited 2p^5 3d^4
manifold is 285 doublets + 150 quartets + 15 SEXTETS -- the sextets being
2p^5(2P) x 3d^4(5D) -> 6P + 6D + 6F. The ground state is a quartet, so it
is the core quartets that are dipole-bright; the doublets and sextets are
dark until SOC mixes them in.

Two ways to get them, both available here:

* **Default (`Ms = 1/2`).** One plain diagonalization in the minimum-|Ms|
  block. That block is a complete seed for every multiplet the system has,
  since a state of total spin S has Ms components from -S to +S. Doublets,
  quartets and sextets come out together, already spin-pure. Measured for
  Cr(acac)3: the Ms=1/2 CVS(nhole=1) space is exactly 450 determinants and
  one solve returns all 450 core states -- 285/150/15 by multiplicity,
  spin-pure to 1.2e-14, in 11 s. Requesting nroots equal to the full space
  size is fine; no dense solver or spin adaptation is needed.

* **`--per-spin`.** One plain diagonalization per total spin S, each in its
  own Ms = S block. At Ms = 3/2 no doublet exists at all, so that block
  yields pure quartets directly -- and much faster (2.9 s vs 13.7 s for 14
  roots here), because the Davidson is not also converging the doublets.
  Use this when you need many roots of one specific multiplicity.

No S^2 penalty is used in either path. `fci.addons.fix_spin_` was tried and
removed: plain solves already return spin-pure roots (see below), while the
penalty cost 9x at Ms=3/2 (26.3 s vs 2.9 s) and failed to converge at all
at Ms=1/2. It solves a problem this calculation does not have.

Spin purity and the unconverged tail
-------------------------------------
`direct_spin1` conserves Sz but not S^2, so roots are *not* guaranteed to
be spin eigenstates. Measured on a 30-root solve here, roots 0-23 are pure
to 1e-9 or better in 2S+1, while roots 24-29 -- the top of the Davidson
window, which is simply not converged -- deviate by 6e-4 to 0.46.

This matters because step 5 builds an Ms ladder with the exact operators
S+/S-, whose normalization sqrt(S(S+1)-Ms(Ms±1)) holds only for pure spin
eigenstates. Contaminated roots break SOC outright -- though not silently:
`soc.ms_manifold` checks every ladder norm and raises, which is what found
this in the first place. So the unconverged tail is filtered off here, and
the count of discarded roots is reported rather than hidden.

Dense diagonalization (`--dense`)
----------------------------------
With ligand orbitals in the valence space the CVS spaces are no longer the
pure d manifolds, and a Davidson for hundreds of roots stops converging:
for [Cr(ddpd)2]3+ CAS(7,7) the one-hole space is 5880 determinants, and a
450-root solve left roots 302-449 unconverged, which cut the core manifold
off 7.6 eV above its onset. Spaces of this size are trivially small for a
dense eigensolver, so `--dense` builds H and S^2 in each CVS space
explicitly (one contraction per determinant), diagonalizes S^2 first and
then H inside each spin block. Every root comes out converged and exactly
spin-pure, and the active-space integrals are built once for both
manifolds instead of once per solve.

The complete solution goes to casci_dense.npz and a table of root and
spin-orbit-state counts per energy window is printed. `--window=EV` then
keeps the core-excited roots within EV (CASCI) of the lowest one and
carries on to MC-PDFT; without it the script stops after the table, which
is how the window is chosen. A later run with `--window=` reuses
casci_dense.npz.

Energies vs. wavefunctions
---------------------------
Unchanged, and deliberate: transition dipoles come from the CASCI CI
vectors, state energies from MC-PDFT. MC-PDFT refines the energy of a given
CASCI density; it does not define a rigorous transition density between two
on-top-functional states. See common.py.

MC-PDFT variant
----------------
Default is **state-specific (SS-PDFT)** with the **tPBE** on-top functional:
each CASCI root's own density is passed through the functional
independently, so there is no state interaction at all.

`--ms=xms|cms|lpdft` switches to **multi-state** MC-PDFT, which builds and
diagonalizes an effective Hamiltonian over a model space. That matters here
because the core-excited manifold is dense -- 19 states within ~6 eV for
this small example -- and SS-PDFT lets near-degenerate states cross without
interacting.

**XMS is the right default for core multiplets**, and the reason is
measurable rather than aesthetic. TiCl3 is near-C3v, so the core-excited
quartets come in degenerate E pairs. Splitting within those pairs, on the
8-root quartet block:

    CASCI (reference)   ~1-3 meV   (the geometry is not exactly C3v)
    XMS-PDFT            0-1 meV    degeneracy preserved
    CMS-PDFT           12-61 meV   degeneracy broken

This is not a CMS bug -- it is CMS working as designed. Its objective
maximizes the sum of state-specific classical Coulomb energies, i.e. it
deliberately selects *charge-localized* intermediate states, which for a
symmetry-degenerate pair means picking a localized combination over a
symmetry-adapted one. XMS instead diagonalizes the state-averaged Fock
operator, which commutes with the spatial symmetry operations and is
invariant to rotations within a degenerate set. CMS is the better-tested
variant for the problems it was built for (conical intersections, charge
transfer); a symmetric multiplet manifold is not one of them.

L-PDFT is also wired up. Note this pyscf-forge build has a bug in it: the
object returned by `linear_multi_state` reads `self.chk_ci`, which is never
set, and lacks `get_ci_adiabats`. Both are worked around here (see
`_ci_adiabats`).
"""
from __future__ import annotations

import os
import pickle
import sys

# The CPUs this job may use, captured BEFORE numpy/pyscf load libgomp. With
# OMP_PROC_BIND set (the job scripts set it, for step 05's threaded
# integrals), libgomp pins this process's main thread to the first core the
# moment pyscf is imported, and os.sched_getaffinity(0) from then on reports
# that one core rather than the allocation. See _pdft_worker_init.
JOB_CPUS = (frozenset(os.sched_getaffinity(0))
            if hasattr(os, 'sched_getaffinity') else None)

import numpy as np
from pyscf import gto, mcpdft, mcscf, scf
from pyscf.fci import addons, cistring, direct_spin1, spin_op
from pyscf.mcpdft import lpdft, mcpdft as mcpdft_mod, mspdft

from mcxray import common
from mcxray import cvs_fci
from mcxray import systems

# Max |2S+1 - nearest integer| for a root to be trusted as a spin
# eigenstate. Converged roots sit at 1e-9 or below; the unconverged tail is
# 1e-3 or worse, so anything in between is comfortably separated.
PURITY_TOL = 1e-6
# Purity tolerance for states that have been through an MS-PDFT rotation.
# Looser than PURITY_TOL because the rotation carries its own round-off:
# measured, the last root of a 16-state model space comes back at 1e-6. The
# failure this guards against -- the effective Hamiltonian mixing doublets
# with quartets -- would show up at O(1), not O(1e-6), so the two are still
# separated by four orders of magnitude.
MS_PURITY_TOL = 1e-4
OTFNAL = 'tPBE'


def make_reference_mf(mol, cas_mo, ncore, ncas, nelecas):
    """A fully-populated mean-field object for CASCI/MC-PDFT. It is used
    to hold the orbitals and integrals, but is not itself solved. It is 
    required because pyscf's CASCI and MC-PDFT objects are built on top 
    of a mean-field object, which is where they get their integrals from.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The molecule for which to create the mean-field object.
    cas_mo : numpy.ndarray
        The CAS orbitals.
    ncore : int
        The number of core orbitals.
    ncas : int
        The number of active orbitals.
    nelecas : int
        The number of electrons in the active space.

    Returns
    -------
    mf : pyscf.scf.ROHF
        The mean-field object.
    """
    mf = scf.ROHF(mol)
    mf.mo_coeff = cas_mo
    occ = np.zeros(cas_mo.shape[1])
    occ[:ncore] = 2
    n_act_pairs = nelecas // 2
    occ[ncore:ncore + n_act_pairs] = 2
    if nelecas % 2:
        occ[ncore + n_act_pairs] = 1
    mf.mo_occ = occ
    mf.mo_energy = np.zeros(cas_mo.shape[1])
    mf.converged = True
    return mf


def nelec_for_ms(nelecas, ms):
    """(na, nb) for a given Ms."""
    na = int(round((nelecas + 2 * ms) / 2))
    nb = nelecas - na
    if nb < 0 or na < 0 or abs((na - nb) / 2 - ms) > 1e-9:
        raise ValueError(f'Ms={ms} unreachable with {nelecas} electrons')
    return (na, nb)


def run_casci(mf, cas_mo, ncore, ncas, nelec, nroots, core_local=None,
              nhole=None):
    """Plain CASCI in one Ms block; `nhole` switches on the CVS restriction."""
    mc = mcscf.CASCI(mf, ncas, nelec)
    mc.ncore = ncore
    if nhole is not None:
        cvs_fci.attach_cvs(mc, core_local, nhole=nhole, nroots=nroots)
        nres, nfull = mc.fcisolver.space_size(ncas, nelec)
        print(f'    CVS(nhole={nhole}): {nres} of {nfull} determinants '
              f'({100 * nres / nfull:.1f}%)')
    mc.fcisolver.nroots = nroots
    mc.kernel(cas_mo)
    roots = list(mc.ci) if isinstance(mc.ci, (list, tuple)) else [mc.ci]
    return roots, np.atleast_1d(mc.e_tot)


def filter_spin_pure(ci, e, ncas, nelec, tol=PURITY_TOL, label=''):
    """Drop roots that are not spin eigenstates (the unconverged tail)."""
    ss = np.array([spin_op.spin_square(c, ncas, nelec)[1] for c in ci])
    keep = np.abs(ss - np.round(ss)) < tol
    if not keep.all():
        drop = np.where(~keep)[0]
        print(f'    {label}discarded {len(drop)} unconverged root(s) '
              f'{drop.tolist()}: 2S+1 off by up to '
              f'{np.abs(ss - np.round(ss))[drop].max():.1e}')
    idx = np.where(keep)[0]
    return ([ci[i] for i in idx], e[idx], np.round(ss[idx]),
            [tuple(nelec)] * len(idx))


def solve_manifold(mf, cas_mo, ncore, ncas, nelecas, nroots, core_local,
                   per_spin, spins, nhole=None):
    """All spin-free states, from one Ms block or one per spin."""
    ci, e, mult, nelecs = [], [], [], []
    ms_list = spins if per_spin else [0.5 if nelecas % 2 else 0.0]
    for ms in ms_list:
        nelec = nelec_for_ms(nelecas, ms)
        n = nroots[ms] if isinstance(nroots, dict) else nroots
        print(f'  Ms={ms} block, nelec={nelec}, nroots={n}')
        r, en = run_casci(mf, cas_mo, ncore, ncas, nelec, n, core_local,
                          nhole=nhole)
        r, en, ss, ne = filter_spin_pure(r, en, ncas, nelec)
        if per_spin:
            # An Ms = S block holds every multiplet with S' >= S, not just
            # S: the Ms=3/2 core block of Cr(acac)3 returns 150 quartets
            # AND the 15 sextets. Without this filter those higher spins
            # are solved again in the next block up and the manifold is
            # silently double-counted -- which then double-counts their
            # intensity in the spectrum.
            keep = np.abs(ss - (2 * ms + 1)) < 0.5
            if not keep.all():
                print(f'    dropped {int((~keep).sum())} root(s) belonging to '
                      f'higher spin blocks (2S+1 != {int(2 * ms + 1)})')
            r = [r[i] for i in np.where(keep)[0]]
            en, ss = en[keep], ss[keep]
            ne = [ne[i] for i in np.where(keep)[0]]
        print(f'    kept {len(r)} spin-pure roots; '
              f'2S+1 present: {sorted(set(ss.tolist()))}')
        ci += r
        e += en.tolist()
        mult += ss.tolist()
        nelecs += ne
    order = np.argsort(e)
    return ([ci[i] for i in order], np.array(e)[order],
            np.array(mult)[order], [nelecs[i] for i in order])


def solve_manifold_cvs(mf, cas_mo, ncore, ncas, nelecas, core_local, per_spin,
                       spins, n_valence, n_core):
    """Two CVS solves -- no core hole, then one core hole -- concatenated.

    A single `nhole=1` solve gives the core-excited manifold but contains no
    ground state, so there is nothing to reference excitation energies to.
    The valence manifold therefore comes from its own `nhole=0` solve, which
    is cheap: restricting to a filled core block is exactly the valence CI.
    The two spaces are orthogonal by construction (they differ in core
    occupation) so their roots simply concatenate, and since both use the
    same orbitals and integrals their energies are directly comparable.
    """
    print('  --- valence manifold (no core hole) ---')
    cv, ev_, mv, nv = solve_manifold(mf, cas_mo, ncore, ncas, nelecas,
                                     n_valence, core_local, False, spins,
                                     nhole=0)
    print('  --- core-excited manifold (one core hole) ---')
    cc, ec, mc_, nc = solve_manifold(mf, cas_mo, ncore, ncas, nelecas,
                                     n_core, core_local, per_spin, spins,
                                     nhole=1)
    return (list(cv) + list(cc), np.concatenate([ev_, ec]),
            np.concatenate([mv, mc_]), list(nv) + list(nc))


def _dense_spin_blocks(solver, h2e, norb, nelec):
    """All eigenstates of H in one CVS space, spin block by spin block.

    Returns (e, vecs, mult, addr): electronic energies (no core energy),
    eigenvectors as columns over the determinants `addr` of the full
    (na, nb) FCI array, and 2S+1 of each.
    """
    mask = solver.determinant_mask(norb, nelec)
    addr = np.flatnonzero(mask.ravel())
    n = len(addr)
    link = solver.gen_linkstr(norb, nelec, True)
    h = np.empty((n, n))
    s2 = np.empty((n, n))
    x = np.zeros(mask.shape)
    for j, a in enumerate(addr):
        x.flat[a] = 1.0
        h[:, j] = solver.contract_2e(h2e, x, norb, nelec, link).ravel()[addr]
        s2[:, j] = spin_op.contract_ss(x, norb, nelec).ravel()[addr]
        x.flat[a] = 0.0
    comm = np.abs(h @ s2 - s2 @ h).max()
    print(f'    dense H and S^2 over {n} determinants: |H - H^T| = '
          f'{np.abs(h - h.T).max():.1e}, |[H, S^2]| = {comm:.1e}')
    if comm > 1e-8:
        raise RuntimeError('H and S^2 do not commute in this space')
    h = 0.5 * (h + h.T)
    w, u = np.linalg.eigh(0.5 * (s2 + s2.T))
    two_s1 = np.sqrt(4 * w + 1)
    if np.abs(two_s1 - np.round(two_s1)).max() > PURITY_TOL:
        raise RuntimeError('S^2 eigenvalues are not S(S+1)')
    two_s1 = np.round(two_s1)
    e, vecs, mult = [], [], []
    for m in np.unique(two_s1):
        ub = u[:, two_s1 == m]
        eb, cb = np.linalg.eigh(ub.T @ h @ ub)
        e.append(eb)
        vecs.append(ub @ cb)
        mult.append(np.full(len(eb), m))
    e, mult = np.concatenate(e), np.concatenate(mult)
    order = np.argsort(e, kind='stable')
    return e[order], np.hstack(vecs)[:, order], mult[order], addr


def solve_dense(mf, cas_mo, ncore, ncas, nelecas, core_local, cache_path):
    """Complete valence and one-core-hole CVS manifolds, saved to
    `cache_path` and reused from there if it exists."""
    import time
    if os.path.exists(cache_path):
        print(f'  dense solution read from {os.path.relpath(cache_path)}')
        return dict(np.load(cache_path))
    nelec = nelec_for_ms(nelecas, 0.5 if nelecas % 2 else 0.0)
    mc = mcscf.CASCI(mf, ncas, nelec)
    mc.ncore = ncore
    t0 = time.time()
    h1, ecore = mc.get_h1eff(cas_mo)
    eri = mc.get_h2eff(cas_mo)
    print(f'  active-space integrals: {(time.time() - t0) / 60:.1f} min',
          flush=True)
    out = dict(nelec=np.array(nelec), ecore=ecore)
    for key, nhole in (('val', 0), ('core', 1)):
        print(f'  --- {"valence" if nhole == 0 else "core-excited"} manifold '
              f'({nhole} core hole{"" if nhole == 1 else "s"}) ---')
        t0 = time.time()
        solver = cvs_fci.CVSFCISolver(mf.mol)
        solver.core_idx, solver.nhole = list(core_local), nhole
        h2e = solver.absorb_h1e(h1, eri, ncas, nelec, .5)
        e, v, m, addr = _dense_spin_blocks(solver, h2e, ncas, nelec)
        print(f'    {len(e)} roots in {time.time() - t0:.0f} s; by 2S+1: '
              + ', '.join(f'{int(k)}: {int((m == k).sum())}'
                          for k in np.unique(m)), flush=True)
        out.update({f'e_{key}': e + ecore, f'vec_{key}': v,
                    f'mult_{key}': m, f'addr_{key}': addr})
    np.savez(cache_path, **out)
    print(f'  dense solution saved to {os.path.relpath(cache_path)}')
    return out


def print_window_table(e, mult, n_valence_so):
    """Roots and spin-orbit states per CASCI energy window."""
    rel = (e - e[0]) * common.HARTREE2EV
    print('\n  core-excited roots within a CASCI window above the lowest '
          'one\n  (SOC dim includes the valence states; SOC matrix is '
          '16 B x dim^2):')
    print(f'  {"window(eV)":>10} {"roots":>6} {"SOC dim":>8} {"h_soc(GB)":>10}')
    for w in [4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, np.inf]:
        k = rel <= w
        dim = int(mult[k].sum()) + n_valence_so
        print(f'  {"all" if np.isinf(w) else f"{w:g}":>10} {int(k.sum()):6d} '
              f'{dim:8d} {16 * dim ** 2 / 1e9:10.2f}')
        if k.all():
            break


def solve_manifold_dense(mf, cas_mo, ncore, ncas, nelecas, core_local,
                         n_valence, window_ev, cache_path):
    """Lowest `n_valence` valence roots plus the core-excited roots within
    `window_ev` of the lowest one, from the dense solution. With
    `window_ev=None` only the window table is printed."""
    d = solve_dense(mf, cas_mo, ncore, ncas, nelecas, core_local, cache_path)
    nelec = tuple(int(x) for x in d['nelec'])
    shape = tuple(cistring.num_strings(ncas, n) for n in nelec)
    n_valence = min(n_valence, len(d['e_val']))
    print_window_table(d['e_core'], d['mult_core'],
                       int(d['mult_val'][:n_valence].sum()))
    if window_ev is None:
        return None
    rel = (d['e_core'] - d['e_core'][0]) * common.HARTREE2EV
    n_core = int((rel <= window_ev).sum())
    print(f'\n  --window={window_ev:g}: {n_core} of {len(rel)} core-excited '
          f'roots, {n_valence} of {len(d["e_val"])} valence roots')
    ci, e, mult = [], [], []
    for key, n in (('val', n_valence), ('core', n_core)):
        for k in range(n):
            c = np.zeros(shape[0] * shape[1])
            c[d[f'addr_{key}']] = d[f'vec_{key}'][:, k]
            ci.append(c.reshape(shape))
        e.append(d[f'e_{key}'][:n])
        mult.append(d[f'mult_{key}'][:n])
    return ci, np.concatenate(e), np.concatenate(mult), [nelec] * len(ci)


def _ci_adiabats(mc):
    """Final (adiabatic) CI vectors of an MS-PDFT object.

    `mspdft` objects expose `get_ci_adiabats`; the L-PDFT state-average
    wrapper in this pyscf-forge build does not (it only sets `si_pdft`, and
    its own accessor is defined on a class the factory does not return), so
    the rotation is applied by hand. Both are the same operation:
    ci_adiabatic = si^T . ci_model.
    """
    if hasattr(mc, 'get_ci_adiabats'):
        return mc.get_ci_adiabats(uci='MSPDFT')
    si = getattr(mc, 'si_pdft', None)
    if si is None:
        raise RuntimeError('MS-PDFT object exposes neither get_ci_adiabats '
                           'nor si_pdft; cannot recover the final states.')
    return list(np.tensordot(np.asarray(si).T, np.asarray(mc.ci), axes=1))


def multi_state_pdft(mf, cas_mo, ncore, ncas, nelec, nroots, core_local,
                     nhole, variant, spin=None, otfnal=OTFNAL):
    """Multi-state MC-PDFT over one (manifold, spin) block.

    Returns (energies, adiabatic CI vectors).

    Why the model space is one manifold, and why both spins in it are safe
    -----------------------------------------------------------------------
    * **One manifold, which REQUIRES the CVS solver.** `multi_state` calls
      the underlying state-averaged CASCI, which returns the LOWEST nroots.
      For the core-excited manifold those are the valence states, ~460 eV
      away and completely irrelevant. Restricting the determinant space to
      one core hole makes the core-excited states the lowest roots, so
      `--ms` implies `--cvs`. Averaging a model space across the 460 eV
      core/valence gap would in any case be meaningless.
    * **Both multiplicities in one model space.** The worry is that an
      MS-PDFT rotation mixes doublets with quartets and destroys the spin
      purity step 5's S+/S- ladder depends on. It does not: the XMS
      rotation diagonalizes the state-averaged Fock operator, which is
      spin-free, and the MS-PDFT effective Hamiltonian takes its
      off-diagonal elements from the CASCI Hamiltonian, also spin-free.
      Both therefore vanish identically between different S, so the
      rotation is block-diagonal in spin. `solve_manifold_mspdft` measures
      <S^2> of every rotated state and raises rather than assuming this.

      Selecting one multiplicity instead, with an S^2 penalty, was tried
      and does NOT work here: pyscf's `mspdft` kernel rebuilds the
      model-space Hamiltonian without the penalty and then asserts its
      eigenvalues match the penalized solver's energies, which trips its
      "Sanity fault" check.
    """
    mc = mcpdft.CASCI(mf, otfnal, ncas, nelec)
    mc.ncore = ncore
    mc.verbose = 0
    cvs_fci.attach_cvs(mc, core_local, nhole=nhole, nroots=nroots)
    weights = [1.0 / nroots] * nroots
    mc = mcscf.state_average_(mc, weights)
    mc.chk_ci = False                      # upstream lpdft.py reads this
    if variant.upper() == 'LPDFT':
        mc = lpdft.linear_multi_state(mc, weights=weights)
    else:
        mc = mspdft.multi_state(mc, weights=weights,
                                diabatization=variant.upper())
    mc.chk_ci = False
    mc.kernel(cas_mo)
    return np.asarray(mc.e_states), _ci_adiabats(mc)


def solve_manifold_mspdft(mf, cas_mo, ncore, ncas, nelecas, core_local,
                          spins, n_valence, n_core, variant):
    """Multi-state MC-PDFT over every (manifold, spin) block."""
    ci, e, mult, nelecs = [], [], [], []
    ms = 0.5 if nelecas % 2 else 0.0
    nelec = nelec_for_ms(nelecas, ms)
    for nhole, n, label in ((0, n_valence, 'valence'),
                            (1, n_core, 'core-excited')):
        en, roots = multi_state_pdft(mf, cas_mo, ncore, ncas, nelec, n,
                                     core_local, nhole, variant)
        ss = np.array([spin_op.spin_square(c, ncas, nelec)[1] for c in roots])
        bad = [(k, round(x, 8)) for k, x in enumerate(ss)
               if abs(x - round(x)) > MS_PURITY_TOL]
        if bad:
            raise RuntimeError(
                f'{variant.upper()}-PDFT rotation broke spin purity in the '
                f'{label} manifold: roots {bad} are not spin eigenstates. '
                'The effective Hamiltonian should be block-diagonal in S; '
                'check the model space.')
        print(f'  {label}: {len(roots)} states, 2S+1 present '
              f'{sorted(set(np.round(ss).tolist()))}, max spin impurity '
              f'{np.abs(ss - np.round(ss)).max():.1e}')
        ci += list(roots)
        e += en.tolist()
        mult += np.round(ss).tolist()
        nelecs += [nelec] * len(roots)
    order = np.argsort(e)
    return ([ci[i] for i in order], np.array(e)[order],
            np.array(mult)[order], [nelecs[i] for i in order])


# ---------------------------------------------------------------------------
# Core Coulomb cache for SS-PDFT
# ---------------------------------------------------------------------------
# pyscf's `energy_mcwfn` builds J[dm_core + dm_active] in the AO basis for
# every state. With frozen orbitals dm_core is the same for all of them, so
# that is the same nao^4-scaling build repeated once per state -- half of
# each state's cost at 430 AOs and nearly all of it at 1613. Split the
# density instead, D = Dc + Da with Da = C_a g C_a^T (g the active 1-RDM):
#
#   Vnn + Tr(h D) + 1/2 (D|D)
#     =  Vnn + Tr(h Dc) + 1/2 (Dc|Dc)                 one number, once
#      + sum_pq [C_a^T (h + J[Dc]) C_a]_pq g_pq       ncas x ncas, once
#      + 1/2 sum_pqrs g_pq (pq|rs) g_rs               active ERIs, once
#
# This is an identity, not an approximation: one J[Dc] build and one
# active-space integral transformation replace every per-state J build, and
# each state's wave-function energy becomes two small contractions.
#
# It covers what `energy_mcwfn` computes for a pure (non-hybrid) on-top
# functional, which is what this workflow uses. For a hybrid functional the
# cache is not installed and pyscf's own path runs. `--no-jcache` forces
# pyscf's path; and whenever the cache is used, one state is recomputed
# through pyscf's `energy_mcwfn` and the run aborts if they disagree.
JCACHE_TOL = 1e-7           # Ha, cached vs pyscf energy_mcwfn


def _is_pure_functional(mcp):
    ot = mcp.otfnal
    hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, mcp.mol.spin)[2]
    return all(abs(x) < 1e-10 for x in hyb)


def build_mcwfn_cache(mf, cas_mo, ncore, ncas, nelec, otfnal=OTFNAL):
    """The three state-independent pieces above, or None for a hybrid
    functional. Runs in the parent, threaded."""
    from pyscf import ao2mo
    mcp = mcpdft.CASCI(mf, otfnal, ncas, tuple(nelec))
    mcp.ncore = ncore
    mcp.mo_coeff = cas_mo
    mcp.verbose = 0
    if not _is_pure_functional(mcp):
        return None
    mo_core = cas_mo[:, :ncore]
    mo_act = cas_mo[:, ncore:ncore + ncas]
    dm_core = 2 * mo_core @ mo_core.T
    h = mf.get_hcore()
    vj_core = mf.get_j(dm=dm_core)
    e_core = (mf.energy_nuc() + np.tensordot(h, dm_core)
              + np.tensordot(vj_core, dm_core) / 2)
    h_act = mo_act.T @ (h + vj_core) @ mo_act
    eri_act = ao2mo.restore(1, mcp.get_h2eff(cas_mo), ncas)
    return dict(e_core=float(e_core), h_act=h_act, eri_act=eri_act)


def install_mcwfn_cache(mcp, cache):
    """Make `energy_tot` use the cache: it calls `mc.energy_mcwfn` if the
    object has one, so an instance attribute is all it takes."""
    if cache is None:
        return mcp

    def energy_mcwfn(ot=None, mo_coeff=None, casdm1s=None, casdm2=None,
                     verbose=None, **kwargs):
        g = np.asarray(casdm1s[0]) + np.asarray(casdm1s[1])
        return (cache['e_core'] + np.tensordot(cache['h_act'], g)
                + np.einsum('pq,pqrs,rs->', g, cache['eri_act'], g) / 2)
    mcp.energy_mcwfn = energy_mcwfn
    return mcp


def check_mcwfn_cache(mf, cas_mo, ncore, ncas, nelec, ci, cache,
                      otfnal=OTFNAL):
    """Cached vs pyscf's own `energy_mcwfn` for one state; fatal on a
    mismatch. Costs one full J build, once."""
    mcp = mcpdft.CASCI(mf, otfnal, ncas, tuple(nelec))
    mcp.ncore = ncore
    mcp.mo_coeff = cas_mo
    mcp.verbose = 0
    ci = np.asarray(ci, dtype=float)
    casdm1s = mcp.make_one_casdm1s(ci, state=0)
    casdm2 = mcp.make_one_casdm2(ci, state=0)
    ref = mcpdft_mod.energy_mcwfn(mcp, mo_coeff=cas_mo, casdm1s=casdm1s,
                                  casdm2=casdm2)
    got = install_mcwfn_cache(mcp, cache).energy_mcwfn(casdm1s=casdm1s)
    if abs(got - ref) > JCACHE_TOL:
        raise RuntimeError(
            f'core Coulomb cache disagrees with pyscf energy_mcwfn: '
            f'{got:.10f} vs {ref:.10f} Ha (diff {got - ref:.2e} > '
            f'{JCACHE_TOL:.0e}). Rerun with --no-jcache.')
    return float(got - ref)


# ---------------------------------------------------------------------------
# Parallel SS-PDFT: one worker process per state, not one thread per state
# ---------------------------------------------------------------------------
# Threading barely helps a single MC-PDFT energy. Measured on Cr(acac)3
# (430 AOs, 43 atoms), `energy_tot` for one state:
#
#     threads    4       8      14
#     s/state   36.4    31.6    28.7
#
# An Amdahl fit gives serial 25.6 s + parallel 43.1 s, i.e. ~89% serial at
# 14 threads and a floor near 25.6 s/state no matter how many cores. A
# 40-core node measured 22.4 s/state -- BELOW that floor, so the gain there
# came from faster cores, not from parallelism. cProfile says why: 15.4 s
# of the 30.3 s is a single direct Coulomb build (`get_j`) and 14.6 s is
# the on-top functional, neither of which scales well at this size.
#
# The loop over roots, by contrast, is a pure map: each state's energy is an
# independent function of its own CI vector, with no shared mutable state.
# So N single-threaded worker PROCESSES buy ~Nx where N threads in one
# process buy almost nothing.
#
# Why 'spawn' and not the default fork: libgomp is not fork-safe, and by the
# time this runs the parent has already executed threaded OpenMP regions in
# the CASCI solve. Forking there can deadlock. Spawn starts each worker as a
# fresh interpreter, which costs a few seconds of import + Mole construction
# per worker, once -- negligible against minutes of work per worker.
_W = {}


def _pdft_worker_init(mol_kwargs, cas_mo, ncore, ncas, nelecas, otfnal,
                      job_cpus, jcache=None):
    """Rebuild the read-only context once per worker process.

    Everything here is identical across states, so it is sent once at
    startup rather than pickled with every task. Only (index, ci, nelec)
    travels per task -- a CI vector is ~31 kB against cas_mo's 1.5 MB.

    The affinity reset is what makes the pool parallel at all under the job
    scripts' OMP_PROC_BIND=close. libgomp pins the parent's main thread to
    one core when pyscf loads, and a spawned child inherits that mask --
    clearing OMP_PROC_BIND for the children does not undo it (measured: 14
    job CPUs, parent pinned to 2, every child on the same 2). Cluster job
    24838352 ran 60 workers that way and took 506 min for 500 states, ~55x
    the single-threaded rate. Each worker is therefore handed back the whole
    allocation, captured before libgomp loaded (JOB_CPUS).
    """
    if job_cpus is not None:
        os.sched_setaffinity(0, job_cpus)
    from pyscf import gto, lib
    lib.num_threads(1)          # the pool provides the parallelism
    mol = gto.M(**mol_kwargs)
    _W.update(mf=make_reference_mf(mol, cas_mo, ncore, ncas, nelecas),
              cas_mo=cas_mo, ncore=ncore, ncas=ncas, otfnal=otfnal,
              jcache=jcache, solvers={})


def _pdft_worker_one(task):
    k, ci, nelec = task
    key = tuple(int(x) for x in nelec)
    mcp = _W['solvers'].get(key)
    if mcp is None:
        mcp = mcpdft.CASCI(_W['mf'], _W['otfnal'], _W['ncas'], key)
        mcp.ncore = _W['ncore']
        mcp.mo_coeff = _W['cas_mo']
        mcp.verbose = 0
        install_mcwfn_cache(mcp, _W['jcache'])
        _W['solvers'][key] = mcp
    e = mcpdft_mod.energy_tot(mcp, mo_coeff=_W['cas_mo'],
                              ci=np.asarray(ci, dtype=float), state=0)[0]
    aff = (len(os.sched_getaffinity(0))
           if hasattr(os, 'sched_getaffinity') else None)
    return k, float(e), aff


def _state_specific_pdft_parallel(mf, cas_mo, ncore, ncas, nelecas, nelecs,
                                  ci_list, nproc, otfnal=OTFNAL, jcache=None):
    """`state_specific_pdft` over a process pool. Same arithmetic, same
    inputs, one worker per state instead of one thread per integral."""
    import concurrent.futures as cf
    import multiprocessing
    import time

    mol = mf.mol
    # Each worker must believe it owns only its share of the node, or 40 of
    # them will each try to pick an incore algorithm sized for the whole
    # machine.
    mol_kwargs = dict(atom=mol.atom, basis=mol.basis, spin=mol.spin,
                      charge=mol.charge, verbose=0,
                      max_memory=max(1000, mol.max_memory / nproc))

    # Spawned children inherit os.environ, so this is how they are pinned to
    # one thread each. The parent's own already-loaded BLAS is unaffected,
    # which is fine because the parent only waits from here on.
    # OMP_PROC_BIND / OMP_PLACES are removed rather than set: with them, a
    # single-threaded worker's libgomp binds it to the first place in its
    # mask, which is the same core for every worker.
    threads = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
    binding = ('OMP_PROC_BIND', 'OMP_PLACES')
    saved = {k: os.environ.get(k) for k in threads + binding}
    os.environ.update({k: '1' for k in threads})
    for k in binding:
        os.environ.pop(k, None)

    # os.cpu_count() reports the MACHINE, not the allocation. Under a SLURM
    # cgroup that grants fewer CPUs than were asked for, spawning nproc
    # workers anyway just oversubscribes: N workers contending for C cores
    # shows up as ~C*100% total CPU and N/C times the expected wall time,
    # with nothing in the log to say so. The affinity mask is the number
    # that actually binds -- taken from JOB_CPUS, because by now the
    # parent's own mask may be the single core libgomp pinned it to.
    if JOB_CPUS is not None:
        avail = len(JOB_CPUS)
        now = len(os.sched_getaffinity(0))
        if now < avail:
            print(f'    note: this process is pinned to {now} of the job\'s '
                  f'{avail} CPUs (OMP_PROC_BIND); workers are reset to all '
                  f'{avail}', flush=True)
    else:                                       # not Linux
        avail = os.cpu_count() or nproc
    print(f'    {nproc} worker processes, 1 thread each; '
          f'{avail} CPUs available to this job', flush=True)
    if nproc > avail:
        print(f'    WARNING: --nproc={nproc} exceeds the {avail} CPUs this '
              f'job can actually use. The workers will oversubscribe and the '
              f'run will be ~{nproc / avail:.1f}x slower than {avail} workers. '
              f'Check --cpus-per-task and the cgroup.', flush=True)

    out = np.empty(len(ci_list))
    t0 = time.time()
    try:
        with cf.ProcessPoolExecutor(
                max_workers=nproc,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_pdft_worker_init,
                initargs=(mol_kwargs, cas_mo, ncore, ncas, nelecas,
                          otfnal, JOB_CPUS, jcache)) as pool:
            futures = [pool.submit(_pdft_worker_one, (k, ci, nelec))
                       for k, (ci, nelec) in enumerate(zip(ci_list, nelecs))]
            done = 0
            for fut in cf.as_completed(futures):
                k, e, aff = fut.result()
                # Fatal, not a warning: a pool that is pinned still returns
                # correct energies, just ~nproc times late, so the only
                # symptom is a walltime that runs out. Checked per result
                # because every worker must pass, not just the first.
                if aff is not None and nproc > 1 and aff < min(nproc, avail):
                    raise RuntimeError(
                        f'SS-PDFT worker is confined to {aff} CPU(s) of '
                        f'{avail}; the pool would share them instead of '
                        'running in parallel. Check OMP_PROC_BIND / '
                        'OMP_PLACES and the job\'s cgroup.')
                out[k] = e
                done += 1
                if len(ci_list) > 20 and done % 25 == 0:
                    el = time.time() - t0
                    print(f'    SS-PDFT {done}/{len(ci_list)} states '
                          f'({nproc} workers), {el / done:.2f} s/state wall, '
                          f'{el / 60:.1f} min elapsed, '
                          f'~{el / done * (len(ci_list) - done) / 60:.0f} min '
                          f'left', flush=True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def state_specific_pdft(mf, cas_mo, ncore, ncas, nelecs, ci_list,
                        otfnal=OTFNAL, nproc=1, nelecas=None, jcache=True):
    """State-specific MC-PDFT energy for each CASCI root's own density.

    States from different spin blocks live in different (na, nb) FCI
    spaces, so one MC-PDFT object is built per distinct block. The on-top
    energy depends only on the density, so the block affects how the
    density is assembled, not the functional evaluation.

    `nproc > 1` evaluates the states over a process pool instead. The
    serial path is the default and stays the reference; see the comment
    above `_pdft_worker_init` for why processes and not threads.

    `jcache=True` builds the core Coulomb cache once (see
    `build_mcwfn_cache`) and checks it against pyscf on the first state;
    `jcache=False` is pyscf's own path, one J build per state.
    """
    import time
    cache = None
    if jcache and len(ci_list):
        t0 = time.time()
        cache = build_mcwfn_cache(mf, cas_mo, ncore, ncas, nelecs[0], otfnal)
        if cache is None:
            print('    core Coulomb cache: not used (hybrid on-top '
                  'functional)', flush=True)
        else:
            t1 = time.time()
            diff = check_mcwfn_cache(mf, cas_mo, ncore, ncas, nelecs[0],
                                     ci_list[0], cache, otfnal)
            print(f'    core Coulomb cache: built in {t1 - t0:.0f} s; vs '
                  f'pyscf energy_mcwfn on state 0: diff {diff:.1e} Ha '
                  f'(check took {time.time() - t1:.0f} s)', flush=True)
    if nproc > 1:
        if nelecas is None:
            raise ValueError('nproc > 1 needs nelecas to rebuild the '
                             'reference mean-field in each worker')
        return _state_specific_pdft_parallel(mf, cas_mo, ncore, ncas, nelecas,
                                             nelecs, ci_list, nproc, otfnal,
                                             jcache=cache)
    solvers, out = {}, np.empty(len(ci_list))
    t0 = time.time()
    for k, (ci, nelec) in enumerate(zip(ci_list, nelecs)):
        key = tuple(nelec)
        if key not in solvers:
            mcp = mcpdft.CASCI(mf, otfnal, ncas, key)
            mcp.ncore = ncore
            mcp.mo_coeff = cas_mo
            solvers[key] = install_mcwfn_cache(mcp, cache)
        out[k] = mcpdft_mod.energy_tot(solvers[key], mo_coeff=cas_mo, ci=ci,
                                       state=0)[0]
        # This is the wall-clock bottleneck of the whole workflow once the
        # system is real: ~28 s/state at 430 AOs / 43 atoms on 14 cores,
        # so 500 states is ~4 h. Report progress rather than going silent.
        if len(ci_list) > 20 and (k + 1) % 25 == 0:
            el = time.time() - t0
            print(f'    SS-PDFT {k + 1}/{len(ci_list)} states, '
                  f'{el / (k + 1):.1f} s/state, '
                  f'{el / 60:.1f} min elapsed, '
                  f'~{el / (k + 1) * (len(ci_list) - k - 1) / 60:.0f} min left',
                  flush=True)
    return out


def core_populations(ci_list, nelecs, ncas, core_local):
    """Core-shell population of each state's active-space 1-RDM."""
    pops = []
    for ci, nelec in zip(ci_list, nelecs):
        dm = direct_spin1.make_rdm1(ci, ncas, tuple(nelec))
        pops.append(float(np.real(np.trace(dm[np.ix_(core_local, core_local)]))))
    return np.array(pops)


if __name__ == '__main__':
    cfg = systems.from_argv(sys.argv)
    RUN_DIR = cfg['run_dir']
    cvs = '--cvs' in sys.argv
    per_spin = '--per-spin' in sys.argv
    jcache = '--no-jcache' not in sys.argv
    dense = '--dense' in sys.argv
    window = None
    ms = None
    nroots = cfg['nroots_core']
    n_valence = cfg['n_valence_roots']
    spins = list(cfg['spins'])
    nproc = 1
    for a in sys.argv[1:]:
        if a.startswith('--nproc='):
            v = a.split('=', 1)[1]
            nproc = os.cpu_count() or 1 if v == 'auto' else int(v)
        if a.startswith('--ms='):
            ms = a.split('=')[1]
            if ms.upper() not in ('XMS', 'CMS', 'LPDFT'):
                raise SystemExit(f'--ms must be xms, cms or lpdft, got {ms!r}')
        if a.startswith('--window='):
            window = float(a.split('=')[1])
        if a.startswith('--nroots='):
            nroots = int(a.split('=')[1])
        elif a.startswith('--nvalence='):
            n_valence = int(a.split('=')[1])
        elif a.startswith('--spins='):
            spins = [float(x) for x in a.split('=')[1].split(',')]
    if per_spin and not spins:
        raise SystemExit('--per-spin needs the core-manifold spins: set '
                         '`spins` in the system config or pass --spins=0.5,1.5')

    with open(os.path.join(RUN_DIR, 'active_space.pkl'), 'rb') as f:
        d = pickle.load(f)
    mol = gto.M(atom=d['atom'], basis=d['basis'], spin=d['spin'],
                charge=d['charge'], verbose=0)
    cas_mo, ncore, ncas = d['cas_mo'], d['ncore'], d['ncas']
    nelecas, core_local = d['nelecas'], d['cas_core_local']
    n_core_orb = d['n_core_orb']

    if dense and (ms or per_spin):
        raise SystemExit('--dense is the single-Ms-block SS-PDFT path; it '
                         'does not combine with --ms or --per-spin')
    if window is not None and not dense:
        raise SystemExit('--window= needs --dense')
    if ms or dense:
        # MS-PDFT needs the target manifold to be the lowest roots, which is
        # exactly what the CVS restriction provides; and it needs a
        # single-multiplicity model space to preserve spin purity.
        cvs = True
    print(f'CASCI({nelecas}e,{ncas}o)  energies: '
          f'{ms.upper() + "-PDFT (multi-state)" if ms else "SS-PDFT (state-specific)"}'
          f'  |  {"per-spin" if per_spin else "single Ms block"}'
          f'{", CVS" if cvs else ""}{", dense" if dense else ""}'
          f'{f", {nproc} worker processes" if nproc > 1 and not ms else ""}')
    if nproc > 1 and ms:
        print('  note: --nproc only applies to state-specific PDFT; the '
              'multi-state path runs serially.')
    mf = make_reference_mf(mol, cas_mo, ncore, ncas, nelecas)
    if ms:
        ci, e_pdft_ms, mult, nelecs = solve_manifold_mspdft(
            mf, cas_mo, ncore, ncas, nelecas, core_local, spins,
            n_valence, nroots, ms)
        e_casci = e_pdft_ms          # CASCI energies are not separated out
    elif dense:
        res = solve_manifold_dense(
            mf, cas_mo, ncore, ncas, nelecas, core_local, n_valence, window,
            os.path.join(RUN_DIR, 'casci_dense.npz'))
        if res is None:
            print('\nNo --window= given: stopping after the window table.')
            sys.exit(0)
        ci, e_casci, mult, nelecs = res
    elif cvs:
        ci, e_casci, mult, nelecs = solve_manifold_cvs(
            mf, cas_mo, ncore, ncas, nelecas, core_local, per_spin, spins,
            n_valence, nroots)
    else:
        ci, e_casci, mult, nelecs = solve_manifold(
            mf, cas_mo, ncore, ncas, nelecas, nroots, core_local, per_spin,
            spins)
    print(f'  {len(ci)} spin-pure states total')

    pops = core_populations(ci, nelecs, ncas, core_local)
    full = 2 * n_core_orb
    no_hole = [i for i, p in enumerate(pops) if full - p <= 0.5]
    core_hole = [i for i, p in enumerate(pops) if full - p > 0.5]
    print(f'  no-hole: {len(no_hole)} states, core-hole: {len(core_hole)}')

    e_pdft = (e_pdft_ms if ms else
              state_specific_pdft(mf, cas_mo, ncore, ncas, nelecs, ci,
                                  nproc=nproc, nelecas=nelecas,
                                  jcache=jcache))
    ev = common.HARTREE2EV
    gs = no_hole[0] if no_hole else 0

    print(f'\n  {"root":>4} {"2p pop":>7} {"2S+1":>5} {"E_CASCI(eV)":>12} '
          f'{"E_MCPDFT(eV)":>13}')
    for i in range(min(len(ci), 14)):
        print(f'  {i:4d} {pops[i]:7.3f} {mult[i]:5.1f} '
              f'{(e_casci[i] - e_casci[gs]) * ev:12.3f} '
              f'{(e_pdft[i] - e_pdft[gs]) * ev:13.3f}')

    # A 1-D object array of float arrays. Building it elementwise is
    # deliberate: np.array(list_of_equal_shaped_arrays, dtype=object) makes a
    # 3-D OBJECT array instead, and pyscf's trans_rdm1 silently returns zeros
    # for object dtype rather than raising.
    ci_store = np.empty(len(ci), dtype=object)
    for i, c in enumerate(ci):
        ci_store[i] = np.asarray(c, dtype=float)

    np.savez(os.path.join(RUN_DIR, 'casci_pdft.npz'),
             ci=ci_store, nelecs=np.array(nelecs),
             e_casci=e_casci, e_pdft=e_pdft, mult=mult, core_pop=pops,
             no_hole=np.array(no_hole), core_hole=np.array(core_hole),
             cas_mo=cas_mo, ncore=ncore, ncas=ncas, nelecas=nelecas,
             core_local=np.array(core_local), cvs=cvs)
    print(f'\nSaved {len(ci)} states to '
          f'{os.path.relpath(os.path.join(RUN_DIR, "casci_pdft.npz"))}')
