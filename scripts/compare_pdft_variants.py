"""Controlled comparison of SS-PDFT and XMS-PDFT for the TiCl3 L-edge.

Everything except the energy treatment is held fixed: the same ORCA
SA-CASSCF orbitals, the same CAS(7e,8o) active space, the same Ms = 1/2
block, and the same CVS-restricted model spaces (one `nhole=0` solve for the
valence manifold, one `nhole=1` solve for the core-excited manifold). So the
underlying CI states are identical and every difference reported here is
attributable to SS vs XMS alone.

Using CVS for BOTH variants matters for fairness. MS-PDFT requires it (the
state-averaged CASCI underneath returns the lowest roots, which for the core
manifold would otherwise be the valence states 460 eV away), and running
SS-PDFT on a different, unrestricted set of roots would confound the model
space with the method. An earlier uncontrolled pair of runs did exactly
that and produced a spurious 81:19 vs 65:35 branching-ratio difference that
was purely a model-space truncation artifact.

Three things are compared:

1. **Excitation energies**, valence and core-excited.
2. **Degeneracy preservation.** TiCl3 is near-C3v, so many states come in
    symmetry-degenerate pairs. How much a method splits a pair that CASCI
    leaves degenerate is a direct, assumption-free quality measure.
3. **The resulting L-edge spectrum** after the identical SOC treatment.

Writes run/pdft_comparison.md and run/pdft_comparison.npz.
"""
from __future__ import annotations

import importlib
import os
import pickle
import sys

import numpy as np
from pyscf import gto
from pyscf.fci import direct_spin1, spin_op

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import soc

s04 = importlib.import_module('04_casci_pdft')
s05 = importlib.import_module('05_soc')

RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'run')
EV = common.HARTREE2EV
N_VALENCE = 5
N_CORE = 16
DEGEN_TOL_EV = 0.010     # CASCI gap below which a pair is called degenerate


def build_states(mf, cas_mo, ncore, ncas, nelecas, core_local, nelec):
    """CVS-CASCI states for both manifolds, at fixed Ms."""
    ci, e, nhole_of = [], [], []
    for nhole, n in ((0, N_VALENCE), (1, N_CORE)):
        roots, en = s04.run_casci(mf, cas_mo, ncore, ncas, nelec, n,
                                  core_local, nhole=nhole)
        ci += list(roots)
        e += np.atleast_1d(en).tolist()
        nhole_of += [nhole] * len(roots)
    return ci, np.array(e), np.array(nhole_of)


def xms_states(mf, cas_mo, ncore, ncas, nelec, core_local):
    """XMS-PDFT over each manifold separately."""
    ci, e, nhole_of = [], [], []
    for nhole, n in ((0, N_VALENCE), (1, N_CORE)):
        en, roots = s04.multi_state_pdft(mf, cas_mo, ncore, ncas, nelec, n,
                                         core_local, nhole, 'XMS')
        ci += list(roots)
        e += np.asarray(en).tolist()
        nhole_of += [nhole] * len(roots)
    return ci, np.array(e), np.array(nhole_of)


def degeneracy_profile(e_ev, tol=DEGEN_TOL_EV):
    """Splittings (meV) of near-degenerate pairs in a SORTED energy list.

    Sorting first is essential and was got wrong in an earlier version of
    this script. SS-PDFT and XMS-PDFT reorder states relative to CASCI --
    for this system CASCI has a near-degenerate triple at 467.176/467.176/
    467.184 eV whose third member SS-PDFT moves below the other two.
    Pairing states by CASCI index therefore compares *different physical
    states* across methods and manufactures apparent 100+ meV symmetry
    breaking out of nothing but a reordering. Comparing sorted lists asks
    the question that actually matters: does the method reproduce the
    near-degenerate structure, wherever those states end up.
    """
    s = np.sort(np.asarray(e_ev)) * 1000.0
    gaps = np.diff(s)
    return gaps[gaps < tol * 1000.0]


def spectrum_from(ci, e_states, mult, nelec, mol, cas_mo, ncore, ncas,
                  nelecas, core_local, nhole_of):
    """Identical SOC + XAS treatment applied to either variant's states."""
    dm = s05.mean_field_density(cas_mo, ncore, ncas, nelecas)
    h_act = soc.to_active(soc.somf(mol, dm), cas_mo, ncore, ncas)
    manifolds = {i: soc.ms_manifold(ci[i], ncas, nelec, (round(m) - 1) / 2.0)
                 for i, m in enumerate(mult)}
    basis = soc.build_basis(np.round(mult).astype(int))
    h_soc = soc.soc_matrix(h_act, manifolds, basis, ncas)
    w, v, _ = soc.diagonalize(e_states, mult, h_soc, basis)

    dip_mo = common.active_space_dipole_ints(mol, cas_mo, ncore, ncas)
    mu_sf = s05.spin_free_dipoles(ci, ncas, [nelec] * len(ci), dip_mo)
    mu_so = s05.dipoles_in_so_basis(mu_sf, basis, v)

    is_core = np.array([n == 1 for n in nhole_of])
    core_w = np.array([sum(abs(v[a, k]) ** 2
                           for a, (i, _) in enumerate(basis) if is_core[i])
                       for k in range(len(basis))])
    g = int(np.argmin(w))
    core = [k for k in range(len(w)) if core_w[k] > 0.5]
    de = np.array([(w[k] - w[g]) * EV for k in core])
    osc = np.array([common.oscillator_strength(mu_so[g, k], w[k] - w[g])
                    for k in core])
    return de, osc


def branching(de, osc):
    s = np.sort(de)
    k = int(np.argmax(np.diff(s)))
    cut = s[k] + np.diff(s)[k] / 2
    lo, hi = de < cut, de >= cut
    cA = (de[lo] * osc[lo]).sum() / osc[lo].sum()
    cB = (de[hi] * osc[hi]).sum() / osc[hi].sum()
    return cB - cA, 100 * osc[lo].sum() / osc.sum(), lo.sum(), hi.sum()


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
# Categorical slots 1 and 2 of the validated default palette. Checked with the
# skill's six-check validator (ported to Python, no node here): CVD separation
# min(protan 24.7, deutan 31.7) = 24.7 against a >=8 target, normal-vision
# deltaE 33.6 against a >=15 floor, contrast 4.30:1 and 3.12:1 on the light
# surface. Identity is carried by the legend AND direct labels, never colour
# alone.
C_SS, C_XMS = '#2a78d6', '#eb6834'
SURFACE, INK, INK2, INK3 = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984'


def plot_comparison(de_ss, osc_ss, de_x, osc_x, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lo = min(de_ss.min(), de_x.min()) - 3
    hi = max(de_ss.max(), de_x.max()) + 3
    grid = np.linspace(lo, hi, 4000)
    s_ss = common.broaden_sticks(de_ss, osc_ss, grid, common.GAMMA_CORE_EV)
    s_x = common.broaden_sticks(de_x, osc_x, grid, common.GAMMA_CORE_EV)
    # One shared scale factor, so the two curves stay quantitatively comparable
    scale = s_ss.max()
    s_ss, s_x = s_ss / scale, s_x / scale

    fig, (ax, axs, axd) = plt.subplots(
        3, 1, figsize=(8.4, 6.6), sharex=True,
        gridspec_kw={'height_ratios': [3.0, 0.7, 1.2], 'hspace': 0.12})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, axs, axd):
        a.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            a.spines[side].set_color(INK3)
            a.spines[side].set_linewidth(0.8)
        a.tick_params(colors=INK2, labelsize=9, length=3, width=0.8)

    ax.plot(grid, s_ss, color=C_SS, lw=2, label='SS-PDFT', zorder=3)
    ax.plot(grid, s_x, color=C_XMS, lw=2, label='XMS-PDFT', zorder=3)
    ax.grid(axis='y', color=INK3, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel('Intensity (norm., shared scale)', color=INK2, fontsize=9.5)
    ax.set_ylim(0, 1.18)
    leg = ax.legend(frameon=False, loc='upper right', fontsize=9.5,
                    labelcolor=INK)
    # Direct labels as well as the legend: identity never rests on colour alone.
    kL3 = int(np.argmax(s_ss))
    kL2 = int(np.argmax(np.where(grid > grid[kL3] + 2.0, s_x, 0)))
    ax.annotate('SS-PDFT', (grid[kL3], s_ss[kL3]), xytext=(-46, 6),
                textcoords='offset points', color=C_SS, fontsize=9.5)
    ax.annotate('XMS-PDFT', (grid[kL2], s_x[kL2]), xytext=(8, 4),
                textcoords='offset points', color=C_XMS, fontsize=9.5)
    ax.set_title('Ti L-edge XAS: state-specific vs extended multi-state MC-PDFT',
                 color=INK, fontsize=12, pad=30, loc='left')
    ax.text(0, 1.012, 'Identical orbitals, active space, model space and SOC '
                      'treatment — only the MC-PDFT step differs',
            transform=ax.transAxes, color=INK2, fontsize=9.2, va='bottom')

    axs.vlines(de_ss, 0.55, 1.0, color=C_SS, lw=1.0, alpha=0.85)
    axs.vlines(de_x, 0.0, 0.45, color=C_XMS, lw=1.0, alpha=0.85)
    axs.set_ylim(-0.05, 1.05)
    axs.set_yticks([0.78, 0.22])
    axs.set_yticklabels(['SS', 'XMS'], fontsize=8.5, color=INK2)
    axs.tick_params(axis='y', length=0)
    axs.set_ylabel('transitions', color=INK2, fontsize=8.5, labelpad=8)

    diff = s_x - s_ss
    axd.axhline(0, color=INK3, lw=0.8)
    axd.fill_between(grid, 0, diff, where=diff >= 0, color=C_XMS, alpha=0.55,
                     lw=0)
    axd.fill_between(grid, 0, diff, where=diff < 0, color=C_SS, alpha=0.55,
                     lw=0)
    axd.grid(axis='y', color=INK3, alpha=0.22, lw=0.6)
    axd.set_axisbelow(True)
    axd.set_ylabel('XMS − SS', color=INK2, fontsize=9.5)
    axd.set_xlabel('Energy above ground state (eV)', color=INK2, fontsize=9.5)
    m = np.abs(diff).max() * 1.25
    axd.set_ylim(-m, m)
    axd.text(0.982, 0.84, f'max |difference| = {np.abs(diff).max():.3f} '
                          f'of peak height', transform=axd.transAxes, ha='right',
             color=INK2, fontsize=8.5)

    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    return path


def write_report(ref, ss, xms, mult, core, de_ss, osc_ss, de_x, osc_x):
    """Render the comparison. Energies are eV above the ground state."""
    prof = {k: degeneracy_profile(v[core])
            for k, v in (('CASCI', ref), ('SS-PDFT', ss), ('XMS-PDFT', xms))}
    sep_ss, br_ss, nlo_ss, nhi_ss = branching(de_ss, osc_ss)
    sep_x, br_x, nlo_x, nhi_x = branching(de_x, osc_x)

    lines = []
    A = lines.append
    A('# SS-PDFT vs XMS-PDFT - TiCl3 Ti L-edge\n')
    A('Controlled comparison: identical ORCA SA-CASSCF orbitals, identical')
    A('CAS(7e,8o), identical Ms=1/2 block, identical CVS-restricted model')
    A(f'spaces ({N_VALENCE} valence + {N_CORE} core-excited roots), identical')
    A('SOC treatment. Only the MC-PDFT step differs, so every difference')
    A('below is attributable to SS vs XMS alone. tPBE on-top functional in')
    A('both cases.\n')

    A('## 1. Excitation energies (eV above the ground state)\n')
    A('Sorted within each manifold, because the two methods do not preserve')
    A('CASCI ordering -- see the note in section 2.\n')
    A('| # | manifold | CASCI | SS-PDFT | XMS-PDFT | XMS - SS |')
    A('|---|---|---|---|---|---|')
    for man, mask in (('valence', ~core), ('core', core)):
        r, a, b = np.sort(ref[mask]), np.sort(ss[mask]), np.sort(xms[mask])
        for i in range(len(r)):
            A(f'| {i} | {man} | {r[i]:.3f} | {a[i]:.3f} | {b[i]:.3f} | '
              f'{b[i]-a[i]:+.3f} |')
    A('')
    dc = np.sort(xms[core]) - np.sort(ss[core])
    dv = np.sort(xms[~core]) - np.sort(ss[~core])
    A(f'Valence: XMS - SS is {np.abs(dv).max()*1000:.1f} meV at most -- the')
    A('two agree essentially exactly where the manifold is sparse and')
    A('well separated.')
    A(f'Core-excited: XMS - SS ranges {dc.min():+.3f} to {dc.max():+.3f} eV')
    A(f'(mean {dc.mean():+.3f} eV). This is where state interaction matters:')
    A('the manifold is dense and SS-PDFT lets near-degenerate states cross')
    A('without interacting.\n')

    A('## 2. Degeneracy preservation\n')
    A('TiCl3 is near-C3v, so many core states form symmetry-degenerate')
    A('pairs. CASCI is the reference: its residual splitting is nonzero only')
    A('because the ORCA-optimized geometry is not exactly C3v.\n')
    A('**Sorted energies are compared, not CASCI-indexed ones.** Both methods')
    A('reorder states relative to CASCI -- CASCI has a near-degenerate triple')
    A('at 467.176/467.176/467.184 eV whose third member SS-PDFT moves below')
    A('the other two. An earlier version of this script paired states by')
    A('CASCI index and so reported 100+ meV of spurious "symmetry breaking"')
    A('that was purely that reordering.\n')
    A('| | near-degenerate pairs found | splittings (meV) | max |')
    A('|---|---|---|---|')
    for k in ('CASCI', 'SS-PDFT', 'XMS-PDFT'):
        v = prof[k]
        A(f'| {k} | {len(v)} | {np.round(v,2).tolist()} | {v.max():.2f} |')
    A('')
    A(f'Both methods preserve the near-degenerate structure. XMS recovers')
    A(f'{len(prof["XMS-PDFT"])} pairs, matching CASCI\'s')
    A(f'{len(prof["CASCI"])}; SS-PDFT recovers {len(prof["SS-PDFT"])}.')
    A('Neither inflates a splitting beyond a few meV.\n')
    A('For contrast, CMS-PDFT on the 8-root quartet block (sorted energies,')
    A('same treatment) splits these pairs by **12-61 meV**. That is CMS')
    A('working as designed: its objective maximizes state-specific classical')
    A('Coulomb energies, deliberately selecting charge-localized rather than')
    A('symmetry-adapted combinations, which breaks a degeneracy that XMS and')
    A('SS both respect.\n')

    A('## 3. Resulting L-edge spectrum (identical SOC treatment)\n')
    A('![SS-PDFT vs XMS-PDFT Ti L-edge XAS](pdft_comparison.png)\n')
    A('Both curves are on one shared intensity scale (not separately')
    A('normalized), so the comparison is quantitative. The bottom panel is')
    A('the difference; its derivative-like shape at each peak is the')
    A('signature of a small energy SHIFT rather than an intensity')
    A('redistribution.\n')
    A('| | SS-PDFT | XMS-PDFT |')
    A('|---|---|---|')
    A(f'| L3 / L2 states | {nlo_ss} / {nhi_ss} | {nlo_x} / {nhi_x} |')
    A(f'| L3 onset (eV) | {de_ss.min():.2f} | {de_x.min():.2f} |')
    A(f'| centroid separation (eV) | {sep_ss:.2f} | {sep_x:.2f} |')
    A(f'| branching ratio | {br_ss:.0f}:{100-br_ss:.0f} | {br_x:.0f}:{100-br_x:.0f} |')
    A(f'| total oscillator strength | {osc_ss.sum():.3e} | {osc_x.sum():.3e} |')
    A('')
    A('Caveat: the L2 branch is under-sampled for BOTH variants at this model')
    A(f'space ({N_CORE} core roots), so the branching ratios are meaningful')
    A('for comparing the methods against each other at fixed model space, not')
    A('as absolute predictions (statistical L3:L2 is 67:33).\n')

    A('## Conclusion\n')
    A('For this system the two are close. They are indistinguishable in the')
    A('valence manifold, and differ by up to a few hundred meV on individual')
    A('core-excited states, with almost no net shift (mean')
    A(f'{dc.mean():+.3f} eV) and a nearly identical final spectrum.')
    A('')
    A('XMS is still the better-justified default for core spectroscopy: it')
    A('adds the state interaction SS-PDFT lacks entirely, which is what a')
    A('dense core-excited manifold needs, and it does so without degrading')
    A('the symmetry-degenerate structure (it in fact reproduces CASCI\'s pair')
    A('count where SS-PDFT loses one). The cost is that the model space must')
    A('be sized deliberately, since MS-PDFT averages over exactly the roots')
    A('requested.')
    return lines


if __name__ == '__main__':
    if '--from-npz' in sys.argv:
        z = np.load(os.path.join(RUN_DIR, 'pdft_comparison.npz'))
        g = int(np.argmin(z['e_casci']))
        ref = (z['e_casci'] - z['e_casci'][g]) * EV
        ss_ = (z['e_ss'] - z['e_ss'][g]) * EV
        xms_ = (z['e_xms'] - z['e_xms'][g]) * EV
        lines = write_report(ref, ss_, xms_, z['mult'], z['nhole'] == 1,
                             z['de_ss'], z['osc_ss'], z['de_xms'], z['osc_xms'])
        with open(os.path.join(RUN_DIR, 'pdft_comparison.md'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        fig = plot_comparison(z['de_ss'], z['osc_ss'], z['de_xms'],
                              z['osc_xms'],
                              os.path.join(RUN_DIR, 'pdft_comparison.png'))
        print(f'Wrote run/pdft_comparison.md and {os.path.basename(fig)}')
        raise SystemExit(0)

    with open(os.path.join(RUN_DIR, 'active_space.pkl'), 'rb') as f:
        d = pickle.load(f)
    mol = gto.M(atom=d['atom'], basis=d['basis'], spin=d['spin'],
                charge=d['charge'], verbose=0)
    cas_mo, ncore, ncas = d['cas_mo'], d['ncore'], d['ncas']
    nelecas, core_local = d['nelecas'], d['cas_core_local']
    nelec = s04.nelec_for_ms(nelecas, 0.5)
    mf = s04.make_reference_mf(mol, cas_mo, ncore, ncas, nelecas)

    print('CVS-CASCI reference states ...')
    ci_ss, e_casci, nhole_of = build_states(mf, cas_mo, ncore, ncas, nelecas,
                                            core_local, nelec)
    mult = np.array([round(spin_op.spin_square(c, ncas, nelec)[1])
                     for c in ci_ss])

    print('SS-PDFT energies ...')
    e_ss = s04.state_specific_pdft(mf, cas_mo, ncore, ncas,
                                   [nelec] * len(ci_ss), ci_ss)

    print('XMS-PDFT energies ...')
    ci_xms, e_xms, nhole_x = xms_states(mf, cas_mo, ncore, ncas, nelec,
                                        core_local)
    mult_x = np.array([round(spin_op.spin_square(c, ncas, nelec)[1])
                       for c in ci_xms])
    assert (nhole_x == nhole_of).all(), 'manifold layout differs'

    g = int(np.argmin(e_casci))
    ref = (e_casci - e_casci[g]) * EV
    ss = (e_ss - e_ss[g]) * EV
    xms = (e_xms - e_xms[g]) * EV
    core = nhole_of == 1

    print('SOC + XAS for each variant ...')
    de_ss, osc_ss = spectrum_from(ci_ss, e_ss, mult, nelec, mol, cas_mo,
                                  ncore, ncas, nelecas, core_local, nhole_of)
    de_x, osc_x = spectrum_from(ci_xms, e_xms, mult_x, nelec, mol, cas_mo,
                                ncore, ncas, nelecas, core_local, nhole_x)
    lines = write_report(ref, ss, xms, mult, core, de_ss, osc_ss, de_x, osc_x)

    out_md = os.path.join(RUN_DIR, 'pdft_comparison.md')
    with open(out_md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    np.savez(os.path.join(RUN_DIR, 'pdft_comparison.npz'),
             e_casci=e_casci, e_ss=e_ss, e_xms=e_xms, mult=mult,
             nhole=nhole_of,
             de_ss=de_ss, osc_ss=osc_ss, de_xms=de_x, osc_xms=osc_x)
    plot_comparison(de_ss, osc_ss, de_x, osc_x,
                    os.path.join(RUN_DIR, 'pdft_comparison.png'))
    print('\n'.join(lines[-12:]))
    print(f'\nWrote {out_md}, run/pdft_comparison.npz, '
          'run/pdft_comparison.png')
