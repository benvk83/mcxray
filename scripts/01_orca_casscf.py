"""Step 1: valence-only SA-CASSCF in ORCA.

This is the orbital-generating step. The active space here is the VALENCE
manifold only -- for TiCl3 that is the five Ti 3d orbitals, state-averaged
over the five d1 ligand-field roots. The 2p core is deliberately absent; it
is spliced into the active space later, on the PySCF side (step 03), for
the CASCI that actually produces the core-excited states.

That split is the whole point of the workflow:

  * Orbitals are optimized for the valence d-manifold, where CASSCF
    orbital relaxation matters a great deal. For TiCl3 the ligand-field
    states move from 1.56 / 3.06 eV (ROHF orbitals) to 0.32 / 1.64 eV
    (SA-CASSCF orbitals) -- a factor of ~2.
  * No attempt is made to variationally optimize orbitals for a core-hole
    state. Those orbitals are frozen, which is the standard
    frozen-orbital approximation for core spectroscopy and keeps every
    transition dipole well defined in one common orbital basis.

`ActOrbs DOrbs` is used rather than letting ORCA take whatever five
orbitals straddle the Fermi level: for TiCl3/def2-SVP the naive choice
grabs MO 37, which is only 1.2% Ti 3d.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orbital_io as oio
import systems

ORCA = os.environ.get('ORCA', shutil.which('orca') or '/home/benvk/orca_6_1_1/orca')


def run_orca(basename, force=False):
    """Run ORCA unless its output is already there."""
    inp = os.path.join(RUN_DIR, basename + '.inp')
    out = os.path.join(RUN_DIR, basename + '.out')
    if not os.path.exists(inp):
        shutil.copy(os.path.join(ORCA_DIR, basename + '.inp'), inp)
    if os.path.exists(out) and not force:
        print(f'  {basename}.out exists; reusing (pass --force to rerun)')
        return out
    print(f'  running ORCA on {basename}.inp ...')
    with open(out, 'w') as fh:
        p = subprocess.run([ORCA, basename + '.inp'], cwd=RUN_DIR,
                           stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f'ORCA failed (exit {p.returncode}); see {out}')
    return out


if __name__ == '__main__':
    cfg = systems.from_argv(sys.argv)
    RUN_DIR, ORCA_DIR = cfg['run_dir'], cfg['orca_dir']
    force = '--force' in sys.argv
    out = run_orca(cfg['orca_job'], force=force)

    roots, e_sa = oio.casscf_roots_from_orca_out(out)
    if len(roots) == 0:
        raise RuntimeError(f'no CAS-SCF roots parsed from {out}')
    ev = 27.211386245988
    print(f'\nORCA {cfg["orca_job"]}: {len(roots)} roots, '
          f'state-averaged E = {e_sa:.10f} Ha')
    print('  ligand-field excitation energies (eV):',
          np.array2string(np.round((roots - roots[0]) * ev, 4)))
    if cfg['name'] == 'ticl3':
        print(f'  (compare ROHF-orbital CASCI on the same active space: '
              f'0, 1.561, 1.563, 3.061, 3.064 eV -- CASSCF orbital relaxation '
              f'roughly halves these)')
    print(f'\nOrbitals for the next step are in '
          f'{os.path.relpath(RUN_DIR)}/{cfg["gbw"]}.gbw')
