"""ORCA -> PySCF orbital transfer for the L-edge workflow.

Why MOKIT, and not ORCA's molden export
----------------------------------------
Earlier versions of this workflow tried `pyscf.tools.molden.load()` on
ORCA's `orca_2mkl` molden output and abandoned it, correctly, for d/f-block
elements. That conclusion has since been sharpened, and the molden route is
now a documented dead end rather than a "best effort" utility:

  * Raw `molden.load` on TiCl3/def2-SVP gives an orthonormality deviation
    `max|C^T S C - I|` of **1.58**.
  * The primitive-normalization correction that the old code applied (see
    `load_orca_molden` below) makes it **worse**, 5.55 -- because def2-SVP
    on Ti carries f functions, which that fix mishandles. It is only
    correct for the s/p-only main-group case it was validated on.
  * An exhaustive search over *every* d and f sign convention (2^5 x 2^7 =
    4096 patterns, applied to both the raw and the normalization-corrected
    basis) bottoms out at a residual of **0.83**. So this is not a phase
    convention that can be patched -- the radial normalization of the
    higher-l shells genuinely disagrees.

MOKIT handles the ORCA -> Gaussian -> PySCF convention chain properly, so
that is the supported route:

    .gbw --orca_2mkl--> .mkl --mkl2fch--> .fch --fch2py--> mo_coeff

MOKIT's conda packages stop at python 3.12 while this workflow runs on
3.13, so `fch2py` is executed in a separate env behind
`scripts/_mokit_bridge.py`. The main env never imports MOKIT.

Setup (once):

    conda create -y -n mokit-bridge -c mokit -c conda-forge python=3.12 mokit
    conda run -n mokit-bridge pip install pyscf     # fch2py needs it

Validation
-----------
`load_orca_orbitals` is deliberately fail-fast. An orbital import that is
silently slightly wrong is far more dangerous than one that raises: the old
molden path converged TiCl3's SCF into a basin 5 Hartree off the true
ground state, which looks like a converged answer. Every import therefore
passes through `assert_orthonormal`, and callers should additionally check
that a CASCI in the imported active space reproduces ORCA's own CASSCF
energy (see `check_casscf_energy`).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from pyscf import gto

MOKIT_ENV = os.environ.get('MOKIT_ENV', 'mokit-bridge')
ORCA_2MKL = os.environ.get('ORCA_2MKL', shutil.which('orca_2mkl')
                           or '/home/benvk/orca_6_1_1/orca_2mkl')
_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# The MOKIT chain
# ---------------------------------------------------------------------------
def _run(cmd, cwd=None, what=''):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f'{what} failed (exit {p.returncode})\n'
                           f'  cmd: {" ".join(cmd)}\n'
                           f'  stdout: {p.stdout.strip()}\n'
                           f'  stderr: {p.stderr.strip()}')
    return p.stdout


def _mokit_bin(name):
    """Absolute path to a MOKIT executable inside the bridge env."""
    root = _run(['conda', 'run', '-n', MOKIT_ENV, 'python', '-c',
                 'import sys, os; print(os.path.dirname(sys.executable))'],
                what='locating mokit env').strip().splitlines()[-1]
    path = os.path.join(root, name)
    if not os.path.exists(path):
        raise RuntimeError(f'MOKIT executable {name!r} not found at {path}. '
                           f'Is the {MOKIT_ENV!r} conda env set up? See this '
                           f'module docstring.')
    return path


def gbw_to_fch(basename, run_dir, natural_orbitals=True):
    """`<basename>.gbw` -> `<basename>.fch`, via `.mkl`.

    `natural_orbitals=True` passes `-no` to `mkl2fch`, which is REQUIRED for
    a CASSCF .gbw: those store state-averaged natural orbitals with
    fractional occupations, and without `-no` mkl2fch's occupied-orbital
    count disagrees with the electron count and it aborts. Natural orbitals
    are in any case a fine active-space basis -- CASCI is invariant to
    rotations within the active space.
    """
    gbw = os.path.join(run_dir, basename + '.gbw')
    if not os.path.exists(gbw):
        raise FileNotFoundError(gbw)
    _run([ORCA_2MKL, basename], cwd=run_dir, what='orca_2mkl')
    fch = os.path.join(run_dir, basename + '.fch')
    cmd = [_mokit_bin('mkl2fch'), basename + '.mkl', basename + '.fch']
    if natural_orbitals:
        cmd.append('-no')
    _run(cmd, cwd=run_dir, what='mkl2fch')
    if not os.path.exists(fch):
        raise RuntimeError(f'mkl2fch reported success but {fch} is missing')
    return fch


def fch_to_mo(fch, ab='a'):
    """MO coefficients out of a `.fch`, in PySCF AO order, via MOKIT."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'mo.npy')
        _run(['conda', 'run', '-n', MOKIT_ENV, 'python',
              os.path.join(_HERE, '_mokit_bridge.py'), fch, out, ab],
             what='fch2py (mokit bridge)')
        return np.load(out)


def fch_occupations(fch):
    """Per-MO occupation numbers from the `.fch` 'Alpha Orbital Energies'
    slot, which `mkl2fch -no` fills with natural-orbital occupations.

    For a state-averaged CASSCF these identify the active space directly:
    inactive sit at 2.0, active at fractional values, virtual at 0.0.
    """
    vals, reading = [], False
    with open(fch) as f:
        for line in f:
            if line.startswith('Alpha Orbital Energies'):
                n = int(line.split()[-1]); reading = True; continue
            if reading:
                if re.match(r'^[A-Za-z]', line):
                    break
                vals.extend(float(x) for x in line.split())
                if len(vals) >= n:
                    break
    return np.array(vals)


# ---------------------------------------------------------------------------
# Validation gates
# ---------------------------------------------------------------------------
# The .fch format writes MO coefficients as E16.8 -- 9 significant digits,
# absolute granularity 1e-8 (literal '1.00000000E-07' / '-0.00000000E+00'
# entries appear in the file). Accumulated over ~85 AOs that puts a floor of
# ~1e-7 on max|C^T S C - I| for ANY import through a .fch, no matter how
# correct the conventions are. Measured for TiCl3/def2-SVP: 1.97e-07. The
# gate is set above that floor; anything materially larger is a real
# convention error (the molden route scores 0.83-5.55 for comparison).
FCH_ROUNDOFF_FLOOR = 1e-7


def orthonormality(mol, mo_coeff):
    s = mol.intor_symmetric('int1e_ovlp')
    return float(np.abs(mo_coeff.T @ s @ mo_coeff - np.eye(mo_coeff.shape[1])).max())


def symmetric_orthonormalize(mol, mo_coeff):
    """Lowdin (S^-1/2) cleanup of the .fch round-off.

    Returns (mo_clean, max_orbital_change). This is cosmetic by
    construction -- it removes a ~1e-7 deviation and so perturbs the
    orbitals by the same order -- but it lets everything downstream assume
    exactly orthonormal MOs. Always check the reported change is at the
    round-off scale: a large change would mean the input was not merely
    round-off-contaminated and the import is genuinely broken.

    The change is the largest overlap-metric norm of any one orbital's
    correction, sqrt(dC_i^T S dC_i), NOT the largest raw coefficient
    change. The raw number scales with the size of the coefficients
    themselves and so with how ill-conditioned the AO basis is: measured
    raw / metric,
        TiCl3      def2-SVP    85 AOs  max|C| =  2.6   2.0e-07 / 2.3e-07
        Cr(acac)3  def2-SVP   430 AOs  max|C| =  8.0   1.5e-06 / 7.4e-07
        Cr(ddpd)2  def2-TZVP 1613 AOs  max|C| = 29.2   2.5e-05 / 1.9e-06
    so a raw-coefficient gate fails a perfectly good triple-zeta import
    (min eig of S 1e-05) while the metric norm stays at the round-off scale.
    """
    s = mol.intor_symmetric('int1e_ovlp')
    m = mo_coeff.T @ s @ mo_coeff
    w, v = np.linalg.eigh(m)
    if w.min() <= 0:
        raise RuntimeError('MO overlap matrix is singular; import is broken.')
    mo_clean = mo_coeff @ (v * w**-0.5) @ v.T
    d = mo_clean - mo_coeff
    return mo_clean, float(np.sqrt(np.einsum('pi,pq,qi->i', d, s, d).max()))


def assert_orthonormal(mol, mo_coeff, tol=1e-6, what='imported orbitals'):
    dev = orthonormality(mol, mo_coeff)
    if dev > tol:
        raise RuntimeError(
            f'{what}: max|C^T S C - I| = {dev:.3e} exceeds tol {tol:.1e}.\n'
            'The orbital import is NOT trustworthy -- refusing to continue. '
            'Check that the PySCF Mole uses the same basis (and the same '
            'cart/spherical convention) as the ORCA calculation.')
    return dev


def check_casscf_energy(e_pyscf, e_orca, tol=1e-6, what='SA-CASSCF energy'):
    """Second import gate: energies, not just geometry of the MO space."""
    d = abs(e_pyscf - e_orca)
    if d > tol:
        raise RuntimeError(
            f'{what}: PySCF {e_pyscf:.10f} vs ORCA {e_orca:.10f} Ha '
            f'(diff {d:.3e} > tol {tol:.1e}). The imported orbitals do not '
            'reproduce ORCA\'s own energy, so the active space or the '
            'orbital transfer is wrong.')
    return d


def load_orca_orbitals(basename, run_dir, mol, natural_orbitals=True,
                       tol=1e-6, reorthonormalize=True, verbose=True):
    """Full import: `<basename>.gbw` -> validated PySCF MO coefficients.

    Returns (mo_coeff, occupations). Raises if the orthonormality gate fails.
    """
    fch = gbw_to_fch(basename, run_dir, natural_orbitals=natural_orbitals)
    mo = fch_to_mo(fch)
    occ = fch_occupations(fch)
    dev = assert_orthonormal(mol, mo, tol=tol,
                             what=f'MOKIT import of {basename}.gbw')
    msg = (f'  MOKIT import {basename}.gbw -> {mo.shape[1]} MOs, '
           f'max|C^T S C - I| = {dev:.2e} (gate {tol:.0e}: PASS')
    if reorthonormalize:
        mo, change = symmetric_orthonormalize(mol, mo)
        if change > 100 * FCH_ROUNDOFF_FLOOR:
            raise RuntimeError(
                f'Lowdin cleanup moved orbitals by {change:.2e}, far above '
                f'the {FCH_ROUNDOFF_FLOOR:.0e} .fch round-off scale -- the '
                'import is not merely round-off-contaminated.')
        msg += f'; Lowdin cleanup moved MOs by {change:.1e}'
    if verbose:
        print(msg + ')')
    return mo, occ


# ---------------------------------------------------------------------------
# ORCA output parsing
# ---------------------------------------------------------------------------
def geometry_from_orca_xyz(xyz_file):
    """Parse a standard .xyz (as ORCA's optimizer writes) into a PySCF atom
    string."""
    with open(xyz_file) as f:
        lines = f.readlines()
    natm = int(lines[0].split()[0])
    return '\n'.join(lines[2:2 + natm])


def energy_from_orca_out(out_file):
    """Final single-point energy ORCA reports."""
    pat = re.compile(r'FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)')
    e = None
    with open(out_file) as f:
        for line in f:
            m = pat.search(line)
            if m:
                e = float(m.group(1))
    return e


def casscf_blocks_from_orca_out(out_file):
    """Every 'CAS-SCF STATES FOR BLOCK' table, as [(mult, energies), ...].

    ORCA emits one block per multiplicity, so a d^3 reference run with
    `mult 4,2` produces two of them. Each block's table is re-emitted on
    every macro-iteration, so a block seen again REPLACES the earlier copy
    of that same multiplicity -- only the final one is kept.
    """
    head = re.compile(r'CAS-SCF STATES FOR BLOCK\s+(\d+)\s+MULT=\s*(\d+)\s+'
                      r'NROOTS=\s*(\d+)')
    root = re.compile(r'ROOT\s+(\d+):\s+E=\s+(-?\d+\.\d+)')
    blocks, cur = {}, None
    with open(out_file) as f:
        for line in f:
            m = head.search(line)
            if m:
                cur = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                blocks[cur[0]] = (cur[1], [])
                continue
            if cur is not None:
                r = root.search(line)
                if r:
                    e = blocks[cur[0]][1]
                    if len(e) < cur[2]:
                        e.append(float(r.group(2)))
                    if len(e) == cur[2]:
                        cur = None
    return [(m, np.array(e)) for m, e in blocks.values()]


def casscf_roots_from_orca_out(out_file):
    """All CASSCF roots (sorted, across every multiplicity block) plus the
    state-averaged energy.

    The state-averaged value is rebuilt from the blocks rather than taken
    from ORCA's 'Final CASSCF energy' line, because ORCA weights BLOCKS
    equally and then states equally within each block. For TiCl3 there is
    one block and this is just the mean; for Cr(acac)3's 10 quartets + 40
    doublets the plain mean over 50 roots is -2071.8514 Ha while the
    correctly weighted value is -2071.8825 Ha, which is what ORCA prints.
    """
    blocks = casscf_blocks_from_orca_out(out_file)
    if not blocks:
        return np.array([]), None
    roots = np.sort(np.concatenate([e for _, e in blocks]))
    sa = float(np.mean([e.mean() for _, e in blocks]))
    return roots, sa


# ---------------------------------------------------------------------------
# Deprecated: the molden route (kept only as a documented dead end)
# ---------------------------------------------------------------------------
def load_orca_molden(fname):
    """DEPRECATED -- do not use for d/f-block systems. See module docstring.

    Retained so the negative result stays discoverable rather than being
    rediscovered. The `gto_norm` division below cancels a real
    double-application of primitive normalization between ORCA's molden
    convention (normalized primitives) and PySCF's raw-list basis parser
    (which assumes unnormalized ones), and is exact for s/p-only main-group
    atoms -- but it does NOT fix, and for f-containing bases actively
    worsens, the transition-metal case this workflow exists for.
    """
    from pyscf.tools import molden
    mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = molden.load(fname)
    new_basis = {}
    for symb, shells in mol._basis.items():
        new_shells = []
        for shell in shells:
            l, prims = shell[0], shell[1:]
            es = np.array([p[0] for p in prims])
            norm = gto.mole.gto_norm(l, es)
            new_shells.append([l] + [[p[0], p[1] / n]
                                     for p, n in zip(prims, norm)])
        new_basis[symb] = new_shells
    mol2 = gto.M(atom=mol.atom, basis=new_basis, unit='Bohr', spin=mol.spin,
                 charge=mol.charge, cart=mol.cart, verbose=0)
    return mol2, mo_energy, mo_coeff, mo_occ
