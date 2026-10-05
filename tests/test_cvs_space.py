"""Sizes of the core-hole-restricted determinant spaces. These are the
counts the root numbers in a system.toml are derived from."""
from __future__ import annotations

import pytest

from mcxray import cvs_fci


def space(norb, nelec, nhole):
    solver = cvs_fci.CVSFCISolver()
    solver.core_idx, solver.nhole = [0, 1, 2], nhole
    return solver.space_size(norb, nelec)


@pytest.mark.parametrize('norb, nelec, nhole, restricted, full', [
    (8, (4, 3), 0, 5, 3920),         # TiCl3: 2p^6 3d^1
    (8, (4, 3), 1, 105, 3920),       # TiCl3: 2p^5 3d^2
    (8, (5, 4), 0, 50, 3920),        # Cr(III) 3d only: 2p^6 3d^3
    (8, (5, 4), 1, 450, 3920),       # Cr(III) 3d only: 2p^5 3d^4
    (10, (7, 6), 0, 1225, 25200),    # [Cr(ddpd)2]3+ CAS(7,7)
    (10, (7, 6), 1, 5880, 25200),
])
def test_space_size(norb, nelec, nhole, restricted, full):
    assert space(norb, nelec, nhole) == (restricted, full)


def test_mask_is_respected_by_the_contraction():
    import numpy as np
    solver = cvs_fci.CVSFCISolver()
    solver.core_idx, solver.nhole = [0, 1, 2], 1
    norb, nelec = 6, (4, 3)
    rng = np.random.default_rng(0)
    h1 = rng.standard_normal((norb, norb))
    h1 = h1 + h1.T
    eri = rng.standard_normal((norb,) * 4) * 0.1
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)
    h2e = solver.absorb_h1e(h1, eri, norb, nelec, .5)
    mask = solver.determinant_mask(norb, nelec)
    x = rng.standard_normal(mask.shape)
    hx = solver.contract_2e(h2e, x, norb, nelec)
    assert np.abs(hx[~mask]).max() == 0
    assert np.allclose(hx, solver.contract_2e(h2e, x * mask, norb, nelec))
