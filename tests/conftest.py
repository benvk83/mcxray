"""Shared fixtures. The TiCl3 fixture is the demo system from step 04 on:
its active_space.pkl (the output of steps 01-03, which need ORCA and
MOKIT) is stored in tests/data, so the tests need PySCF only."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
EXAMPLE = os.path.join(HERE, '..', 'examples', 'ticl3')


def run_step(step, system_dir, *args):
    """Run `mcxray STEP --system=system_dir ...` and return its stdout."""
    env = dict(os.environ)
    # See README: thread binding pins the whole job to one core here.
    env.pop('OMP_PROC_BIND', None)
    env.pop('OMP_PLACES', None)
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env.setdefault(var, '4')
    env['MPLBACKEND'] = 'Agg'
    p = subprocess.run([sys.executable, '-m', 'mcxray.cli', step,
                        f'--system={system_dir}', *args],
                       env=env, text=True, capture_output=True)
    assert p.returncode == 0, f'mcxray {step} failed:\n{p.stdout}\n{p.stderr}'
    return p.stdout


def ticl3_dir(path):
    """A fresh TiCl3 run directory, ready for step 04."""
    os.makedirs(path, exist_ok=True)
    for fn in ('system.toml', 'ticl3_opt.xyz'):
        shutil.copy(os.path.join(EXAMPLE, fn), path)
    shutil.copy(os.path.join(DATA, 'ticl3_active_space.pkl'),
                os.path.join(path, 'active_space.pkl'))
    return str(path)


@pytest.fixture(scope='session')
def reference():
    """Numbers produced by the pre-package scripts (commit ba4b0bd)."""
    return np.load(os.path.join(DATA, 'ticl3_reference.npz'))


@pytest.fixture(scope='session')
def ticl3_cvs_run(tmp_path_factory):
    """Steps 04 (--cvs, two workers) to 06, run once for the session."""
    d = ticl3_dir(tmp_path_factory.mktemp('ticl3_cvs'))
    logs = {'04': run_step('04', d, '--cvs', '--nproc=2'),
            '05': run_step('05', d),
            '06': run_step('06', d)}
    return d, logs


@pytest.fixture(scope='session')
def ticl3_dense_run(tmp_path_factory):
    """Step 04 --dense without a window: the CASCI only, no MC-PDFT."""
    d = ticl3_dir(tmp_path_factory.mktemp('ticl3_dense'))
    return d, run_step('04', d, '--dense')
