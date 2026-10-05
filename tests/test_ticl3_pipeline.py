"""TiCl3 from step 04 on, against numbers produced by the scripts as they
were before they became this package (commit ba4b0bd). Run with

    pytest              # everything, about three minutes
    pytest -m "not slow"   # seconds: no electronic structure at all
"""
from __future__ import annotations

import os

import numpy as np
import pytest

EV = 27.211386245988
pytestmark = pytest.mark.slow


def load(run_dir, name):
    return np.load(os.path.join(run_dir, name + '.npz'), allow_pickle=True)


def test_step04_cvs_energies(ticl3_cvs_run, reference):
    z = load(ticl3_cvs_run[0], 'casci_pdft')
    assert len(z['e_casci']) == 35
    assert len(z['no_hole']) == 5 and len(z['core_hole']) == 30
    assert bool(z['cvs'])
    order, ref_order = np.argsort(z['e_casci']), np.argsort(reference['cvs_e_casci'])
    np.testing.assert_allclose(z['e_casci'][order],
                               reference['cvs_e_casci'][ref_order], atol=1e-8)
    np.testing.assert_array_equal(z['mult'][order],
                                  reference['cvs_mult'][ref_order])
    # MC-PDFT is not variational, so it inherits the Davidson's ~1e-5
    # convergence of the CI vectors at first order (README: 1e-7 Ha scatter).
    np.testing.assert_allclose(np.sort(z['e_pdft']),
                               np.sort(reference['cvs_e_pdft']), atol=2e-6)


def test_step04_states_are_spin_pure_with_one_core_hole(ticl3_cvs_run):
    z = load(ticl3_cvs_run[0], 'casci_pdft')
    assert sorted(set(z['mult'].tolist())) == [2.0, 4.0]
    np.testing.assert_allclose(z['core_pop'][z['no_hole']], 6.0, atol=1e-10)
    np.testing.assert_allclose(z['core_pop'][z['core_hole']], 5.0, atol=1e-10)


def test_step04_log_names_the_code_version(ticl3_cvs_run):
    first = ticl3_cvs_run[1]['04'].splitlines()[0]
    assert first.startswith('mcxray ') and '| system ticl3 |' in first


def test_cvs_is_close_to_the_unrestricted_casci(ticl3_cvs_run, reference):
    """The README's claim: CVS costs ~10 meV on a ~469 eV excitation."""
    z = load(ticl3_cvs_run[0], 'casci_pdft')
    cvs = np.sort(z['e_casci'][z['core_hole']]) - z['e_casci'].min()
    hole = reference['exact_core_pop'] < 5.5
    exact = (np.sort(reference['exact_e_casci'][hole])
             - reference['exact_e_casci'].min())
    n = min(len(cvs), len(exact))
    assert n >= 15
    assert np.abs(cvs[:n] - exact[:n]).max() * EV < 0.015


def test_step05_spin_orbit_states(ticl3_cvs_run, reference):
    z = load(ticl3_cvs_run[0], 'soc_results')
    assert z['e_so'].shape == (104,)
    np.testing.assert_allclose(np.sort(z['e_so']),
                               np.sort(reference['soc_e_so']), atol=2e-6)
    h = z['h_soc']
    assert np.abs(h - h.conj().T).max() < 1e-12


def test_step06_spectra(ticl3_cvs_run, reference):
    z = load(ticl3_cvs_run[0], 'spectra')
    np.testing.assert_allclose(z['grid'], reference['xas_grid'])
    np.testing.assert_allclose(np.sort(z['de_ev']),
                               np.sort(reference['xas_de_ev']), atol=1e-4)
    # Individual oscillator strengths can be redistributed inside a
    # degenerate pair; their sum and the broadened spectrum cannot.
    np.testing.assert_allclose(z['osc'].sum(), reference['xas_osc'].sum(),
                               rtol=1e-4)
    scale = reference['xas_spectrum'].max()
    np.testing.assert_allclose(z['spectrum'], reference['xas_spectrum'],
                               atol=1e-4 * scale)
    np.testing.assert_allclose(z['rixs'][::6, ::6], reference['rixs'],
                               atol=1e-3 * reference['rixs'].max())
    for png in ('xas_spectrum.png', 'rixs_map.png'):
        assert os.path.getsize(os.path.join(ticl3_cvs_run[0], png)) > 10_000


def test_dense_solver_matches_reference_and_davidson(ticl3_dense_run,
                                                     ticl3_cvs_run, reference):
    d = load(ticl3_dense_run[0], 'casci_dense')
    assert d['e_val'].shape == (5,) and d['e_core'].shape == (105,)
    np.testing.assert_allclose(d['e_val'], reference['dense_e_val'], atol=1e-9)
    np.testing.assert_allclose(d['e_core'], reference['dense_e_core'],
                               atol=1e-9)
    np.testing.assert_array_equal(d['mult_core'], reference['dense_mult_core'])
    # 2p^5 3d^2 is 6 x C(10,2) = 270 microstates = 75 doublets + 30 quartets
    assert (d['mult_core'] == 2).sum() == 75
    assert (d['mult_core'] == 4).sum() == 30
    # without --window= the step stops after the table
    assert 'stopping after the window table' in ticl3_dense_run[1]
    assert not os.path.exists(os.path.join(ticl3_dense_run[0],
                                           'casci_pdft.npz'))
    # the Davidson's 30 core roots are the 30 lowest dense ones
    z = load(ticl3_cvs_run[0], 'casci_pdft')
    np.testing.assert_allclose(np.sort(z['e_casci'][z['core_hole']]),
                               d['e_core'][:30], atol=1e-7)
    np.testing.assert_allclose(np.sort(z['e_casci'][z['no_hole']]),
                               d['e_val'], atol=1e-7)
