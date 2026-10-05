"""mcxray -- multiconfigurational X-ray spectroscopy.

Transition-metal L-edge XAS and RIXS from valence SA-CASSCF orbitals:
core-valence-separated CI in PySCF, MC-PDFT state energies and
Breit-Pauli SOMF spin-orbit coupling. See README.md.

Importing this package imports nothing heavy. In particular it does not
import pyscf: step 04 has to record the job's CPU set before pyscf's
OpenMP runtime gets a chance to change it.
"""
from __future__ import annotations

import os
import subprocess

__version__ = '0.1.0'

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def git_revision():
    """Short commit hash of the source checkout, with '+' appended if the
    working tree differs from it; None if this is not a git checkout."""
    if not os.path.exists(os.path.join(_REPO, '.git')):
        return None
    try:
        def git(*args):
            return subprocess.run(('git', '-C', _REPO) + args, text=True,
                                  capture_output=True, timeout=30,
                                  check=True).stdout.strip()
        rev = git('rev-parse', '--short', 'HEAD')
        return rev + ('+' if git('status', '--porcelain') else '')
    except (OSError, subprocess.SubprocessError):
        return None


def version_string():
    """'mcxray 0.1.0 (git abc1234)' -- what every step prints first."""
    rev = git_revision()
    return f'mcxray {__version__}' + (f' (git {rev})' if rev else '')
