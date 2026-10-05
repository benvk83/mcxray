"""Breit-Pauli spin-orbit coupling for the L-edge workflow.

Without SOC an L-edge calculation has no L3/L2 structure at all, and the
quartet 2p^5 3d^2 states -- which for TiCl3 with SA-CASSCF orbitals are the
LOWEST core-excited states -- are strictly dark. SOC is what makes them
visible, so it is not a refinement here, it is the physics.

There is no SOMF/AMFI implementation in pyscf (`pyscf.prop` is an empty
namespace package), so this module builds one from the raw integrals
`int1e_prinvxp` and `int2e_p1vxp1`.

The operator
-------------
Breit-Pauli, in atomic units:

    H_SO = (alpha^2/2) [ sum_i sum_A Z_A (r_iA x p_i).s_i / r_iA^3
                       - sum_i/=j (r_ij x p_i).(s_i + 2 s_j) / r_ij^3 ]

The two-electron part is folded into an effective one-electron operator by
the standard spin-orbit mean-field (SOMF) construction of Hess, Marian,
Wahlgren and Gropen -- the same approximation ORCA's RI-SOMF and
OpenMolcas' AMFI make:

    h_ij = h1e_ij + sum_kl D_kl [ (ij|kl) - 3/2 (il|kj) - 3/2 (kj|il) ]

Calibration: the raw one-electron term alone gives zeta(Ti 2p) = 4.05 eV.
The accepted atomic value is ~3.7 eV, and 2e screening is what closes that
gap -- so a SOMF result slightly BELOW the 1e number is the expected
behaviour and is checked for explicitly (`somf_report`).

Spin blocks and the Ms ladder
------------------------------
The CI is solved once at Ms = +1/2, which contains every half-integer
multiplet the system has. The other Ms components that SOC couples are
generated here with the exact ladder operators

    S- = sum_p a+_{p beta} a_{p alpha},   S+ = sum_p a+_{p alpha} a_{p beta}

normalized by S-|S,Ms> = sqrt(S(S+1) - Ms(Ms-1)) |S,Ms-1>.

Generating the components rather than diagonalizing each Ms block
separately matters for more than convenience: it fixes the relative phases
within a multiplet, which is exactly what makes the assembled SOC matrix
come out Hermitian with no Wigner-Eckart phase bookkeeping. Hermiticity is
therefore a real check on the construction, and is asserted.

Matrix elements
----------------
Decomposing h.s = h_z s_z + (h+ s- + h- s+)/2, with h+- = h_x +- i h_y:

    <I,Ms  |H_SO|J,Ms>   = 1/2 sum_pq h_z,pq (tdm_alpha - tdm_beta)_pq
    <I,Ms+1|H_SO|J,Ms>   = 1/2 sum_pq h-,pq  <I|a+_{p alpha} a_{q beta}|J>

and dMs = -1 follows by Hermitian conjugation. The dMs = 0 part uses
`direct_spin1.trans_rdm1s`; the dMs = +-1 part is built with
`fci.addons.des_b` followed by `cre_a`, which is the only way to reach
across Ms blocks.
"""
from __future__ import annotations

import time

import numpy as np
from pyscf import lib
from pyscf.data import nist
from pyscf.scf import jk
from pyscf.fci import addons, direct_spin1

HARTREE2EV = 27.211386245988
ALPHA2_HALF = nist.ALPHA ** 2 / 2

# Tolerance on the S+/S- ladder normalization. Raw CI roots are spin-pure to
# ~1e-9, but states that have been through an MS-PDFT rotation carry the
# rotation's round-off (~1e-6 measured), so this must accommodate them. A
# genuinely spin-contaminated state -- the thing this check exists to catch
# -- misses by O(1e-2) or worse.
LADDER_RTOL = 1e-5

# Above this many integral batches, hso_2e_somf reports a per-batch
# heartbeat. Small systems finish in seconds and stay quiet; large ones
# would otherwise print nothing for hours.
PROGRESS_MIN_BATCHES = 20

# Likewise for the (state, Ms) basis in soc_matrix: 70 pairs for TiCl3
# (quiet), 1380 for Cr(acac)3 (reports every 100 columns).
PROGRESS_MIN_BASIS = 200


# ---------------------------------------------------------------------------
# Integrals
# ---------------------------------------------------------------------------
def hso_1e(mol):
    """One-electron Breit-Pauli SOC integrals, real and antisymmetric.

    The physical operator is `1j * ALPHA2_HALF * hso_1e` (i times a real
    antisymmetric matrix is Hermitian).
    """
    h = np.zeros((3, mol.nao, mol.nao))
    for iat in range(mol.natm):
        z = mol.atom_charge(iat)
        with mol.with_rinv_origin(mol.atom_coord(iat)):
            h += -z * mol.intor('int1e_prinvxp', comp=3)
    return h


# The two-electron SOC integral is the expensive part of this module, and
# the one place where system size genuinely bites. `int2e_p1vxp1` is
# 3 x nao^4 doubles:
#
#     TiCl3      85 AOs      1.25 GB
#     Cr(acac)3 430 AOs    764.20 GB
#
# Only its contraction with one density is ever needed, which is a J/K
# build, so the default (`method='direct'`) hands it to pyscf's integral-
# direct J/K driver: the (ij|kl) antisymmetry in ij ('a4ij') cuts the
# integral count 4x, negligible shell quartets are screened out, the
# contraction happens in C as integrals are made, and nothing nao^4 is
# ever stored. Measured against the batched path below:
#
#                         batched s1 + einsum        direct (jk.get_jk)
#     TiCl3, 14 cores     3.4 s                      2.4 s, max|diff| 4.5e-13
#     Cr(acac)3           53.3 min (60-core node,    5.1 min (14-core laptop,
#                         16 GB batches)             0.3 GB peak)
#
# and on Cr(acac)3 both give zeta(Cr 2p, SOMF) = 5.300 eV. The gain grows
# with molecule size, since screening removes more of the tensor.
#
# `method='batched'` keeps the original path -- the full s1 tensor built in
# shell slabs over the leading AO index and contracted with numpy -- as the
# reference. Its peak memory is set by `max_bytes`; one AO of that index
# carries 3 x nao^3 x 8 B (1.78 GB at 430 AOs) and a batch is never smaller
# than one shell. Per-system values live in systems.py (`soc_max_bytes`).
#
# Contracted results are memoized so it is built once per (molecule, density).
_SOMF_CACHE = {}
SOMF_METHODS = ('direct', 'batched')


def _shell_batches(mol, max_bytes):
    """Group shells so one batch of int2e_p1vxp1 stays under `max_bytes`.

    The int() casts are load-bearing, not decoration. `mol.ao_loc_nr()`
    returns an INT32 array, so `n * per_ao` is evaluated in int32 and
    silently overflows once per_ao approaches 2^31 -- which it does exactly
    when batching starts to matter. At 430 AOs per_ao is 3 x 430^3 x 8 =
    1.9e9, so n = 2 already wraps negative, the size test never fires, and
    the function returns a SINGLE batch spanning every shell: a 764 GB
    allocation and an instant OOM. TiCl3 never tripped it because its
    per_ao is 1.5e7 and the batch closes at n ~ 13, four orders of
    magnitude below the wrap.
    """
    ao_loc = [int(x) for x in mol.ao_loc_nr()]
    nao = int(mol.nao)
    per_ao = 3 * nao ** 3 * 8          # bytes per AO of the batched index
    batches, start = [], 0
    for sh in range(1, mol.nbas + 1):
        n = ao_loc[sh] - ao_loc[start]
        at_end = sh == mol.nbas
        if n * per_ao > max_bytes or at_end:
            stop = sh if at_end else max(sh - 1, start + 1)
            batches.append((start, stop))
            start = stop
            if start >= mol.nbas:
                break
    return batches


def hso_2e_somf(mol, dm, max_bytes=200 * 1024 ** 2, method='direct'):
    """Two-electron SOC (spin-same-orbit + spin-other-orbit) contracted
    against a mean-field density into an effective one-electron operator,

        h_ij = sum_kl D_kl [ (ij|kl) - 3/2 (il|kj) - 3/2 (kj|il) ].

    `method='direct'` (default) builds this with the integral-direct J/K
    driver; `method='batched'` with the original slab-batched tensor, whose
    peak memory is `max_bytes`. Both are exact; see the comment above
    `_SOMF_CACHE` for the measured comparison. `max_bytes` only applies to
    the batched path.
    """
    if method not in SOMF_METHODS:
        raise ValueError(f'method must be one of {SOMF_METHODS}, got {method!r}')
    key = (id(mol), mol.nao, hash(np.asarray(dm).tobytes()), method)
    if key in _SOMF_CACHE:
        return _SOMF_CACHE[key]
    out = (_hso_2e_somf_direct(mol, dm) if method == 'direct'
           else _hso_2e_somf_batched(mol, dm, max_bytes))
    _SOMF_CACHE.clear()          # only ever one density in flight
    _SOMF_CACHE[key] = out
    return out


def _hso_2e_somf_direct(mol, dm):
    """The SOMF 2e term as three J/K-type contractions, integral-direct.

    The scripts are exactly the three einsums of the batched path. 'a4ij'
    declares (ij|kl) antisymmetric in i<->j and symmetric in k<->l, which is
    the permutational symmetry of int2e_p1vxp1 (the orbit operator acts on
    electron 1); a wrong declaration here would not raise, which is why it
    was checked against the s1 tensor rather than assumed (max|diff| 4.5e-13
    on TiCl3, and zeta 5.300 eV on Cr(acac)3 either way).

    One call, so no per-batch heartbeat: it prints start and finish only.
    """
    t0 = time.time()
    print(f'      2e SOC: integral-direct J/K build over {mol.nao} AOs '
          f'({lib.num_threads()} threads)...', flush=True)
    vj, vk1, vk2 = jk.get_jk(mol, [dm, dm, dm],
                             ['ijkl,lk->ij', 'ijkl,jk->il', 'ijkl,li->kj'],
                             intor='int2e_p1vxp1', aosym='a4ij', comp=3)
    print(f'      2e SOC: done in {(time.time() - t0) / 60:.1f} min',
          flush=True)
    return vj - 1.5 * (vk1 + vk2)


def _hso_2e_somf_batched(mol, dm, max_bytes):
    """Reference path: the full s1 tensor in shell slabs, contracted by numpy.

    The full `int2e_p1vxp1` tensor is 3 x nao^4 -- 1.25 GB already at the
    85 AOs of this small demo, before einsum temporaries -- which is enough
    to start swapping on an ordinary machine. It is therefore built and
    contracted in batches over the first AO index, so peak memory is set by
    `max_bytes` rather than by nao^4. The result is identical; only the
    memory profile changes.

    Progress is reported once the batch count exceeds `PROGRESS_MIN_BATCHES`.
    That threshold exists so the small demo stays quiet while a real system
    does not go silent for hours: Cr(acac)3 at a 16 GB cap is 59 batches
    over 53 min on a 60-core node, and without a heartbeat there is no way
    to tell a running job from a wedged one. TiCl3 is ~7 batches and prints
    nothing.
    """
    nao, nbas = mol.nao, mol.nbas
    ao_loc = mol.ao_loc_nr()
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    batches = _shell_batches(mol, max_bytes)
    report = len(batches) > PROGRESS_MIN_BATCHES
    t0 = time.time()
    for n, (sh0, sh1) in enumerate(batches):
        i0, i1 = ao_loc[sh0], ao_loc[sh1]
        blk = mol.intor('int2e_p1vxp1', comp=3, aosym='s1',
                        shls_slice=(sh0, sh1, 0, nbas, 0, nbas, 0, nbas))
        blk = blk.reshape(3, i1 - i0, nao, nao, nao)
        vj[:, i0:i1, :] += np.einsum('xijkl,lk->xij', blk, dm)
        vk[:, i0:i1, :] += np.einsum('xijkl,jk->xil', blk, dm)
        vk += np.einsum('xijkl,li->xkj', blk, dm[:, i0:i1])
        del blk
        if report:
            el = time.time() - t0
            print(f'      2e SOC batch {n + 1}/{len(batches)} '
                  f'(AOs {i0}-{i1} of {nao}), {el / 60:.1f} min elapsed, '
                  f'~{el / (n + 1) * (len(batches) - n - 1) / 60:.0f} min left',
                  flush=True)
    return vj - 1.5 * vk


def somf(mol, dm, include_2e=True, max_bytes=200 * 1024 ** 2,
         method='direct'):
    """Full SOMF operator in the AO basis, already scaled by alpha^2/2.

    Returns a real antisymmetric (3, nao, nao) array `h`; the Hermitian
    operator components are `1j * h`.
    """
    h = hso_1e(mol)
    if include_2e:
        h = h + hso_2e_somf(mol, dm, max_bytes=max_bytes, method=method)
    return ALPHA2_HALF * h


def to_active(h_ao, cas_mo, ncore, ncas):
    """AO -> active-space MO block."""
    c = cas_mo[:, ncore:ncore + ncas]
    return np.einsum('xuv,up,vq->xpq', h_ao, c, c)


def core_shell_zeta(h_act, core_local):
    """The p-shell SOC constant zeta, read off the active-space operator.

    Within a p shell the SOC operator is zeta * l.s, so in a canonical
    (px, py, pz) basis the 3x3 core block of component x is
    h[x]_pq = zeta * eps_xpq. Contracting that with itself,

        sum_x sum_pq h[x]_pq^2 = 6 * zeta^2

    and the Levi-Civita symbol is an isotropic tensor, so this reading is
    invariant under a simultaneous rotation of the Cartesian index and the
    orbital indices -- i.e. under rotating the molecule.

    That invariance is the point. The obvious shortcut, max|h_z| over the
    core block, is only equal to zeta when the two orbitals it connects are
    the in-plane pair for the LAB z axis. TiCl3 is built with z along its
    C3 axis, so there the shortcut happens to be right (4.050 vs 4.051 eV)
    and the error was invisible. Cr(acac)3 sits at an arbitrary
    orientation, where the same shortcut reads 4.424 eV against a true
    zeta of 6.009 eV -- a 26% underestimate of a quantity that sets the
    L3-L2 splitting. The operator itself was never wrong; only this
    diagnostic was.
    """
    b = h_act[:, core_local][:, :, core_local]
    return float(np.sqrt((b ** 2).sum() / 6.0))


def somf_report(mol, cas_mo, ncore, ncas, core_local, dm,
                max_bytes=200 * 1024 ** 2, method='direct'):
    """Print the core-shell SOC constant with and without 2e screening.

    Screening must REDUCE zeta; a SOMF value above the 1e value means the
    density or the integral conventions are wrong, and is called out.

    The 2e integrals are memoized, so the `include_2e=True` pass here costs
    nothing extra beyond what the operator itself needs.
    """
    out = {}
    for tag, twoe in (('1e only', False), ('SOMF (1e+2e)', True)):
        h = to_active(somf(mol, dm, include_2e=twoe, max_bytes=max_bytes,
                           method=method),
                      cas_mo, ncore, ncas)
        zeta = core_shell_zeta(h, core_local)
        out[tag] = zeta
        print(f'    zeta(core 2p), {tag:13s}: {zeta * HARTREE2EV:6.3f} eV'
              f'   -> L3-L2 = 1.5*zeta = {1.5 * zeta * HARTREE2EV:6.3f} eV')
    if out['SOMF (1e+2e)'] >= out['1e only']:
        print('    WARNING: 2e screening did not reduce zeta -- check the '
              'SOMF density or integral conventions.')
    return out


# ---------------------------------------------------------------------------
# Spin ladder
# ---------------------------------------------------------------------------
def spin_lower(ci, norb, nelec):
    """S- |ci>, returning the vector in the (na-1, nb+1) block (unnormalized)."""
    na, nb = nelec
    out = None
    for p in range(norb):
        t = addons.des_a(ci, norb, (na, nb), p)
        t = addons.cre_b(t, norb, (na - 1, nb), p)
        out = t if out is None else out + t
    return out


def spin_raise(ci, norb, nelec):
    """S+ |ci>, returning the vector in the (na+1, nb-1) block (unnormalized)."""
    na, nb = nelec
    out = None
    for p in range(norb):
        t = addons.des_b(ci, norb, (na, nb), p)
        t = addons.cre_a(t, norb, (na, nb - 1), p)
        out = t if out is None else out + t
    return out


def ms_manifold(ci, norb, nelec, s):
    """All Ms components of one multiplet, from its Ms = (na-nb)/2 component.

    Returns {Ms: (ci_vector, (na, nb))}, with each component normalized.
    The ladder normalization S-|S,Ms> = sqrt(S(S+1)-Ms(Ms-1))|S,Ms-1> is
    used as a check: the norm of the raw ladder result must match, which
    verifies both that `s` is the true spin of the state and that the
    creation/annihilation phase conventions compose correctly.
    """
    na, nb = nelec
    ms0 = (na - nb) / 2.0
    out = {ms0: (np.asarray(ci), (na, nb))}

    cur, cur_nelec, ms = np.asarray(ci), (na, nb), ms0
    while ms - 1 >= -s - 1e-9:
        expect = np.sqrt(s * (s + 1) - ms * (ms - 1))
        raw = spin_lower(cur, norb, cur_nelec)
        got = np.linalg.norm(raw)
        if not np.isclose(got, expect, rtol=LADDER_RTOL, atol=1e-8):
            raise RuntimeError(
                f'S- normalization mismatch at Ms={ms}: got {got:.8f}, '
                f'expected {expect:.8f} for S={s}. The state is probably '
                'not a pure spin eigenstate.')
        cur_nelec = (cur_nelec[0] - 1, cur_nelec[1] + 1)
        cur, ms = raw / got, ms - 1
        out[ms] = (cur, cur_nelec)

    cur, cur_nelec, ms = np.asarray(ci), (na, nb), ms0
    while ms + 1 <= s + 1e-9:
        expect = np.sqrt(s * (s + 1) - ms * (ms + 1))
        raw = spin_raise(cur, norb, cur_nelec)
        got = np.linalg.norm(raw)
        if not np.isclose(got, expect, rtol=LADDER_RTOL, atol=1e-8):
            raise RuntimeError(
                f'S+ normalization mismatch at Ms={ms}: got {got:.8f}, '
                f'expected {expect:.8f} for S={s}.')
        cur_nelec = (cur_nelec[0] + 1, cur_nelec[1] - 1)
        cur, ms = raw / got, ms + 1
        out[ms] = (cur, cur_nelec)
    return out


# ---------------------------------------------------------------------------
# Transition densities across Ms blocks
# ---------------------------------------------------------------------------
def spinflip_basis(ci_ket, nelec_ket, norb):
    """All a+_{p alpha} a_{q beta} |ket> vectors, flattened to (norb^2, dim).

    Built once per ket and reused against every bra: the ladder operators
    are the expensive part, the overlaps are a single matrix-vector product.
    """
    na, nb = nelec_ket
    out = np.empty((norb, norb), dtype=object)
    for q in range(norb):
        u = addons.des_b(ci_ket, norb, (na, nb), q)
        for p in range(norb):
            out[p, q] = np.asarray(
                addons.cre_a(u, norb, (na, nb - 1), p)).ravel()
    dim = out[0, 0].size
    return np.stack([out[p, q] for p in range(norb)
                     for q in range(norb)]).reshape(norb * norb, dim)


def spinflip_tdm(ci_bra, nelec_bra, ci_ket, nelec_ket, norb, basis=None):
    """<bra| a+_{p alpha} a_{q beta} |ket>, bra in (na+1, nb-1) of ket."""
    na, nb = nelec_ket
    if tuple(nelec_bra) != (na + 1, nb - 1):
        raise ValueError(f'block mismatch: bra {nelec_bra} vs ket {nelec_ket}')
    if basis is None:
        basis = spinflip_basis(ci_ket, nelec_ket, norb)
    return basis.dot(np.asarray(ci_bra).ravel()).reshape(norb, norb)


# ---------------------------------------------------------------------------
# The SOC matrix over (state, Ms) pairs
# ---------------------------------------------------------------------------
def build_basis(mult, spins=None):
    """Enumerate (state_index, Ms) pairs for the spin-orbit basis."""
    basis = []
    for i, m in enumerate(mult):
        s = (m - 1) / 2.0
        ms = -s
        while ms <= s + 1e-9:
            basis.append((i, round(ms * 2) / 2))
            ms += 1
    return basis


def soc_matrix(h_act, manifolds, basis, norb):
    """Assemble <I,Ms_I| H_SO |J,Ms_J> over the whole (state, Ms) basis.

    `h_act` is the (3, ncas, ncas) real antisymmetric active-space SOMF
    operator; the Hermitian components are 1j*h_act.
    `manifolds[i][ms] = (ci, nelec)` from `ms_manifold`.
    """
    hx, hy, hz = 1j * h_act[0], 1j * h_act[1], 1j * h_act[2]
    hplus, hminus = hx + 1j * hy, hx - 1j * hy

    n = len(basis)
    h = np.zeros((n, n), dtype=complex)
    flip_cache = {}
    # Pure-Python double loop over n^2 pairs -- single-threaded whatever the
    # node, and the second-longest silent stretch in step 05 after the
    # integrals (1380 pairs for Cr(acac)3 against 70 for TiCl3).
    report = n > PROGRESS_MIN_BASIS
    t0 = time.time()
    for b, (j, msj) in enumerate(basis):
        if report and b and b % 100 == 0:
            el = time.time() - t0
            print(f'      SOC matrix column {b}/{n}, {el / 60:.1f} min '
                  f'elapsed, ~{el / b * (n - b) / 60:.0f} min left', flush=True)
        ci_j, ne_j = manifolds[j][msj]
        for a, (i, msi) in enumerate(basis):
            if msi < msj:          # fill one triangle, conjugate the rest
                continue
            ci_i, ne_i = manifolds[i][msi]
            if msi == msj:
                dma, dmb = direct_spin1.trans_rdm1s(ci_i, ci_j, norb, ne_j)
                val = 0.5 * np.einsum('pq,qp->', hz, dma - dmb)
            elif abs(msi - msj - 1) < 1e-9:
                if (j, msj) not in flip_cache:
                    flip_cache[(j, msj)] = spinflip_basis(ci_j, ne_j, norb)
                t = spinflip_tdm(ci_i, ne_i, ci_j, ne_j, norb,
                                 basis=flip_cache[(j, msj)])
                val = 0.5 * np.einsum('pq,pq->', hminus, t)
            else:
                continue
            h[a, b] = val
            h[b, a] = np.conj(val)
    return h


def diagonalize(e_states, mult, h_soc, basis):
    """H = diag(E_state) + H_SO over the (state, Ms) basis -> eigenpairs."""
    diag = np.array([e_states[i] for i, _ in basis])
    h = h_soc + np.diag(diag).astype(complex)
    herm = np.abs(h - h.conj().T).max()
    if herm > 1e-10:
        raise RuntimeError(f'spin-orbit Hamiltonian is not Hermitian '
                           f'(max asymmetry {herm:.2e}) -- the Ms ladder or '
                           'the matrix-element phases are inconsistent.')
    w, v = np.linalg.eigh(h)
    return w, v, herm
