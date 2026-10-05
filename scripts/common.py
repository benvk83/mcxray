"""
Shared utilities for the L-edge XAS / RIXS workflow:

    ORCA orbitals  -->  PySCF active-space selection  -->  CASCI  -->  MC-PDFT
                                                                     |
                                                    transition dipoles (CASCI level)
                                                                     |
                                                    XAS stick spectrum / RIXS map

Design notes (read before extending this file)
------------------------------------------------
* MC-PDFT gives a refined *energy* for a given CASCI/CASSCF state's density;
  it does not define a new wavefunction or a rigorous transition density
  between two different on-top-functional states. The standard, defensible
  approach (used throughout the RASSCF/RASPT2 and MC-PDFT X-ray literature)
  is therefore a hybrid:
      - transition dipole moments  -> computed from the CASCI wavefunctions
        (well-defined multiconfigurational transition density matrices)
      - state energies (resonance positions, KH denominators) -> computed
        at the MC-PDFT level (state-specific "SS-PDFT" energy of each
        CASCI density)
  This module keeps that separation explicit: functions that touch `ci`
  vectors are wavefunction-level; functions that touch `e_tot` are
  MC-PDFT-level.

* No spin-orbit coupling is included. Electric-dipole transitions are only
  evaluated between states of identical total spin (Delta S = 0); this is
  enforced structurally by only ever calling trans_rdm1 on CI vectors drawn
  from the same fixed-spin FCI solver.

* RIXS orientation averaging is polarization-summed (powder): the full 3x3
  Cartesian scattering tensor is built and every component is squared and
  summed, weighting the three rotational invariants equally. It is NOT
  polarization-resolved -- a specific parallel/perpendicular scattering
  geometry weights those invariants differently and needs them computed
  separately. Note in particular that the tensor must not be contracted to
  its trace before squaring; kramers_heisenberg_map's docstring explains
  why that shortcut silently deletes the inelastic d-d channels.
"""
from __future__ import annotations

import numpy as np
from pyscf import lib
from pyscf.fci import direct_spin1

# ---------------------------------------------------------------------------
# Physical constants and default broadenings
# ---------------------------------------------------------------------------
HARTREE2EV = 27.211386245988

# Core-hole lifetime broadening (HWHM) for the intermediate states, and the
# final-state/instrumental broadening for the RIXS energy-loss axis. Both
# live here rather than being redefined in each driver script, which is how
# they previously drifted apart. ~0.4 eV is a typical 3d-metal L-edge value;
# adjust per system and edge.
GAMMA_CORE_EV = 0.4
GAMMA_FINAL_EV = 0.15


# ---------------------------------------------------------------------------
# Orbital bookkeeping / active space
# ---------------------------------------------------------------------------
def ao_character(mol, mo_coeff, ao_labels_substr):
    """Return, for every MO, the fraction of its Mulliken-like AO weight
    (sum of C_{mu,i}^2 over normalized AOs, i.e. assuming an orthogonal-ish
    minimal-overlap estimate) that sits on AOs whose label contains
    `ao_labels_substr` (e.g. 'Ti 2p', 'Ti 3d').

    This is a quick-look diagnostic for picking active-space orbitals by
    hand; it is deliberately simple (no full Mulliken partitioning with S)
    -- always eyeball the printed table before trusting the selection.
    """
    s = mol.intor_symmetric('int1e_ovlp')
    labels = mol.ao_labels()
    mask = np.array([ao_labels_substr in lab for lab in labels])
    # weight of AO mu in MO i, properly using the overlap metric:
    # population_mu(i) = C_mu,i * (S C)_mu,i
    sc = s.dot(mo_coeff)
    pop = mo_coeff * sc  # elementwise, shape (nao, nmo)
    frac = pop[mask, :].sum(axis=0) / pop.sum(axis=0)
    return frac


def print_orbital_table(mol, mo_energy, mo_coeff, mo_occ, metal_label,
                         indices=None):
    """Human-checkable printout: index, energy (eV), occupation, and the
    dominant AO character near `metal_label` (2p / 3d). Use this to confirm
    or override the automatically-guessed active-space indices -- automatic
    core/valence assignment is a heuristic, not a guarantee.

    `mo_energy` may be None, which is the normal case for orbitals read
    from a CASSCF .gbw: those are natural orbitals, which carry occupation
    numbers rather than canonical orbital eigenvalues. The energy column is
    then printed as '--' instead of a meaningless number.
    """
    frac_2p = ao_character(mol, mo_coeff, f'{metal_label} 2p')
    frac_3d = ao_character(mol, mo_coeff, f'{metal_label} 3d')
    idx = range(len(mo_occ)) if indices is None else indices
    print(f"{'idx':>4} {'E(eV)':>12} {'occ':>5} {'%2p':>7} {'%3d':>7}")
    for i in idx:
        e = '--'.rjust(12) if mo_energy is None else f'{mo_energy[i]*HARTREE2EV:12.3f}'
        print(f"{i:4d} {e} {mo_occ[i]:5.2f} "
              f"{100*frac_2p[i]:7.1f} {100*frac_3d[i]:7.1f}")


# ---------------------------------------------------------------------------
# State classification (core-hole vs. no-hole) from active-space density
# ---------------------------------------------------------------------------
def active_1rdm(fcisolver, ci_vec, ncas, nelecas):
    """Spin-summed active-space 1-RDM <ci|E_pq|ci> for one CI vector."""
    dm1 = fcisolver.make_rdm1(ci_vec, ncas, nelecas)
    return dm1


def core_hole_population(fcisolver, ci_vec, ncas, nelecas, core_ao_idx):
    """Population (trace of the active 1-RDM) on the given active-space
    orbital indices that represent the metal core (2p) shell. For a
    "no-hole" state this should be ~ 2*len(core_ao_idx); for a state with
    one 2p hole it drops by ~1.
    """
    dm1 = active_1rdm(fcisolver, ci_vec, ncas, nelecas)
    return np.real(np.trace(dm1[np.ix_(core_ao_idx, core_ao_idx)]))


def classify_states(fcisolver, ci_list, ncas, nelecas, core_idx,
                     full_core_pop=None, hole_threshold=0.5):
    """Split CI roots into 'no_hole' (ground + valence/ligand-field excited
    states) and 'core_hole' (XAS/RIXS intermediate states) by comparing the
    2p-shell population to the fully-occupied value.

    Returns dict(no_hole=[indices...], core_hole=[indices...], pop=[...]).
    """
    if full_core_pop is None:
        full_core_pop = 2 * len(core_idx)
    pops = [core_hole_population(fcisolver, ci, ncas, nelecas, core_idx)
            for ci in ci_list]
    no_hole, core_hole = [], []
    for i, p in enumerate(pops):
        if full_core_pop - p > hole_threshold:
            core_hole.append(i)
        else:
            no_hole.append(i)
    return {'no_hole': no_hole, 'core_hole': core_hole, 'pop': pops}


# ---------------------------------------------------------------------------
# Transition dipoles (CASCI wavefunction level)
# ---------------------------------------------------------------------------
def active_space_dipole_ints(mol, mo_coeff, ncore, ncas, origin=None):
    """AO dipole integrals transformed into the active-orbital MO block.

    Returns array of shape (3, ncas, ncas). The gauge origin defaults to
    the nuclear charge centroid (a molecule-fixed, well-defined choice;
    absolute value of a *transition* dipole between orthogonal states is
    gauge-origin independent, but we fix one for reproducibility/logging).
    """
    if origin is None:
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        origin = np.einsum('i,ix->x', charges, coords) / charges.sum()
    with mol.with_common_origin(origin):
        dip_ao = mol.intor('int1e_r', comp=3)
    mo_cas = mo_coeff[:, ncore:ncore + ncas]
    dip_mo = np.einsum('xuv,up,vq->xpq', dip_ao, mo_cas, mo_cas)
    return dip_mo


def transition_dipole(fcisolver, ci_bra, ci_ket, ncas, nelecas, dip_mo):
    """Electronic transition dipole moment <bra| r |ket> restricted to the
    active space (core orbitals are doubly occupied in both bra and ket by
    construction of CASCI, so they contribute nothing to an off-diagonal
    transition density).

    `nelecas` must be identical for bra and ket (same fixed-spin FCI
    problem) -- this enforces Delta S = 0 structurally.
    """
    # Guard against object-dtype CI vectors. `trans_rdm1` accepts them and
    # silently returns ZEROS rather than raising, so an object array -- which
    # is exactly what `np.array(list_of_arrays, dtype=object)` produces when
    # round-tripping states through .npz -- yields a spectrum with every
    # oscillator strength identically zero and no error anywhere.
    for name, vec in (('bra', ci_bra), ('ket', ci_ket)):
        if not np.issubdtype(np.asarray(vec).dtype, np.floating):
            raise TypeError(
                f'{name} CI vector has dtype '
                f'{np.asarray(vec).dtype}, expected float. trans_rdm1 returns '
                'zeros for non-float input instead of raising; cast with '
                'np.asarray(ci, dtype=float) when loading.')
    tdm1 = direct_spin1.trans_rdm1(ci_bra, ci_ket, ncas, nelecas)
    # pyscf convention: trans_rdm1[p,q] = <bra| p^+ q |ket>
    mu = np.einsum('xpq,qp->x', dip_mo, tdm1)
    return np.real(mu)


def oscillator_strength(mu_vec, de_hartree):
    """f = (2/3) * dE * |mu|^2 in atomic units (isotropic average over the
    three Cartesian components already implicit since mu_vec is the full
    3-vector).

    `mu_vec` may be complex, which it is between spin-orbit eigenstates:
    the SOC eigenvectors are complex, so transition dipoles in that basis
    are too. |mu|^2 is therefore sum_x mu_x* mu_x, not sum_x mu_x^2 -- the
    latter is not even real for a complex dipole.
    """
    return float((2.0 / 3.0) * de_hartree * np.vdot(mu_vec, mu_vec).real)


# ---------------------------------------------------------------------------
# Broadening / spectra
# ---------------------------------------------------------------------------
def lorentzian(x, x0, gamma):
    return (gamma / np.pi) / ((x - x0) ** 2 + gamma ** 2)


def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def broaden_sticks(energies_ev, intensities, grid_ev, gamma_ev,
                    lineshape='lorentzian'):
    """Sum of broadened lines on `grid_ev`."""
    fn = lorentzian if lineshape == 'lorentzian' else gaussian
    spec = np.zeros_like(grid_ev)
    for e, i in zip(energies_ev, intensities):
        spec += i * fn(grid_ev, e, gamma_ev)
    return spec


# ---------------------------------------------------------------------------
# Kramers-Heisenberg RIXS map
# ---------------------------------------------------------------------------
def kramers_heisenberg_map(omega_in_grid, energy_loss_grid,
                            e_g, e_n_list, e_f_list,
                            mu_gn_list, mu_nf_matrix,
                            gamma_n_ev, gamma_f_ev):
    """Direct Kramers-Heisenberg RIXS intensity map, powder /
    polarization-summed (incident and scattered polarizations are not
    resolved -- see module docstring and the note on invariants below).

    The scattering amplitude is a rank-2 Cartesian TENSOR in the
    (scattered, incident) polarization indices:

      F^{beta alpha}_fg(Omega)
          = sum_n  <f|mu_beta|n><n|mu_alpha|g>
                   / (Omega - (E_n-E_g) + i*Gamma_n)

    and the polarization-summed intensity keeps all nine components:

      I(Omega, omega) = sum_f  (sum_{alpha,beta} |F^{beta alpha}_fg|^2)
                               * L(omega - (E_f-E_g), Gamma_f)

    Do NOT contract this to sum_alpha F^{alpha alpha} (the tensor trace)
    before squaring. Under orientation averaging the intensity splits into
    three rotational invariants of F -- isotropic (|Tr F|^2/3),
    antisymmetric, and symmetric-traceless -- and the trace retains only
    the first. For a RIXS loss feature between two states of different
    ligand-field symmetry the trace can, and here does, vanish exactly:
    the d-d channels of the TiCl3 example carry |Tr F|^2 ~ 1e-13 while
    sum_{alpha,beta}|F|^2 ~ 1e-3, so a trace-only average annihilates
    precisely the inelastic features RIXS is measured for. Summing all
    nine components weights the three invariants equally, which is the
    standard polarization-summed powder measure; a specific
    parallel/perpendicular scattering geometry would weight them
    differently and needs the invariants computed separately.

    Parameters
    ----------
    omega_in_grid : (nO,) incident photon energy grid, eV, relative to E_g
    energy_loss_grid : (nE,) energy-transfer grid, eV (Omega - omega)
    e_g : float, ground-state energy (eV)
    e_n_list, e_f_list : intermediate / final state energies (eV)
    mu_gn_list : list of 3-vectors, ground->intermediate transition dipoles
    mu_nf_matrix : array (n_n, n_f, 3), intermediate->final transition dipoles
    gamma_n_ev, gamma_f_ev : core-hole lifetime and final-state broadening (eV)

    Returns
    -------
    map_ : (nE, nO) RIXS intensity
    """
    n_n = len(e_n_list)
    n_f = len(e_f_list)
    nO = len(omega_in_grid)
    # F[f, beta, alpha, Omega]: the full Cartesian scattering tensor, summed
    # over intermediate states n. Kept uncontracted until after squaring.
    F = np.zeros((n_f, 3, 3, nO), dtype=complex)
    for n in range(n_n):
        denom = (omega_in_grid - (e_n_list[n] - e_g)) + 1j * gamma_n_ev  # (nO,)
        amp_alpha = mu_gn_list[n]        # (3,)   <n|mu_alpha|g>
        for f in range(n_f):
            amp_beta = mu_nf_matrix[n, f]  # (3,) <f|mu_beta|n>
            coupling = np.outer(amp_beta, amp_alpha)  # (3,3), no contraction
            F[f] += coupling[:, :, None] / denom

    nE = len(energy_loss_grid)
    map_ = np.zeros((nE, nO))
    for f in range(n_f):
        weight = np.abs(F[f]) ** 2         # (3, 3, nO)
        weight = weight.sum(axis=(0, 1))   # polarization sum -> (nO,)
        center = e_f_list[f] - e_g
        line = lorentzian(energy_loss_grid, center, gamma_f_ev)  # (nE,)
        map_ += np.outer(line, weight)
    return map_


def scattering_tensor(omega_in_grid, e_i, e_n, mu_ni, mu_fn, gamma_n_ev):
    """F[f, beta, alpha, Omega] of kramers_heisenberg_map as one matrix
    product over the intermediates.

    e_n : (n_n,) eV;  mu_ni : (n_n, 3) <n|mu_alpha|i>;
    mu_fn : (n_n, n_f, 3) <f|mu_beta|n>.
    """
    e_n = np.asarray(e_n)
    inv = 1.0 / ((omega_in_grid[None, :] - (e_n - e_i)[:, None])
                 + 1j * gamma_n_ev)
    a = np.asarray(mu_ni)[:, :, None] * inv[:, None, :]      # (n, 3, nO)
    b = np.asarray(mu_fn)                                    # (n, f, 3)
    n = len(e_n)
    F = b.reshape(n, -1).T @ a.reshape(n, -1)
    return F.reshape(b.shape[1], 3, 3, len(omega_in_grid))


def rotational_invariants(F):
    """Split sum_{alpha,beta}|F|^2 into its three rotational invariants.

    F : (n_f, 3, 3, nO). Returns (G0, G1, G2), each (n_f, nO): isotropic
    |Tr F|^2/3, antisymmetric sum|(F - F^T)/2|^2 and symmetric-traceless.
    G0 + G1 + G2 is the nine-component sum kramers_heisenberg_map uses.
    """
    tr = F[:, 0, 0] + F[:, 1, 1] + F[:, 2, 2]
    Ft = F.transpose(0, 2, 1, 3)
    sym = 0.5 * (F + Ft)
    for x in range(3):
        sym[:, x, x] -= tr / 3
    g0 = np.abs(tr) ** 2 / 3
    g1 = (np.abs(0.5 * (F - Ft)) ** 2).sum(axis=(1, 2))
    g2 = (np.abs(sym) ** 2).sum(axis=(1, 2))
    return g0, g1, g2


def geometry_weights(pol, theta_deg):
    """Weights (w0, w1, w2) of the invariants for a randomly oriented
    sample, linearly polarized incident light and no polarization
    analysis of the scattered beam.

    For real unit polarization vectors e (in) and e' (out) the
    orientation average is

      <|e'.F.e|^2> = b + (a + c) (e'.e)^2,
      b     = G1/6 + G2/10,
      a + c = G0/3 - G1/6 + G2/30,

    i.e. G0/3 + 2 G2/15 for parallel and G1/6 + G2/10 for crossed
    polarizations. Summing e' over the two directions transverse to the
    outgoing beam gives 2b + (a + c) s with s = sum (e'.e)^2:

      pol='pi'    (e in the scattering plane):       s = cos^2(theta)
      pol='sigma' (e normal to the scattering plane): s = 1

    theta is the angle between incident and scattered beams. At
    theta = 90 with pi polarization e lies along the outgoing beam, s = 0,
    and the isotropic invariant G0 does not scatter at all.
    """
    if pol == 'pi':
        s = np.cos(np.radians(theta_deg)) ** 2
    elif pol == 'sigma':
        s = 1.0
    else:
        raise ValueError(f"pol must be 'pi' or 'sigma', not {pol!r}")
    return s / 3, 1 / 3 - s / 6, 1 / 5 + s / 30


def geometry_from_argv(argv):
    """`--pol=pi|sigma` and `--theta=DEG` (default 90).

    Returns (weights, tag, description), or None when --pol is absent and
    the caller should keep the plain nine-component sum.
    """
    pol, theta = None, 90.0
    for a in argv[1:]:
        if a.startswith('--pol='):
            pol = a.split('=', 1)[1]
        elif a.startswith('--theta='):
            theta = float(a.split('=', 1)[1])
    if pol is None:
        return None
    return (geometry_weights(pol, theta), f'{pol}{theta:g}',
            f'randomly oriented, {pol}-polarized in, {theta:g} deg, '
            'no polarization analysis')


def tag_from_argv(argv):
    """`--tag=NAME` selects an alternative set of state energies: steps 6
    and 7 then read soc_results_NAME.npz (05b_soc_swap_energies.py) and
    put _NAME in their output names. Returns '' or '_NAME'.
    """
    for a in argv[1:]:
        if a.startswith('--tag='):
            return '_' + a.split('=', 1)[1]
    return ''
