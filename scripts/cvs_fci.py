"""Optional core-valence-separated (CVS) FCI solver.

When you do NOT need this
--------------------------
With frozen MOs and the core orbitals inside the active space, an ordinary
CASCI is exact and the core-excited states are simply high-lying roots. For
the 3d-only active space of the TiCl3 demo -- CAS(7e,8o), with 2p in it --
the core-hole manifold starts at root 6, so `nroots=30` reaches it
comfortably and no approximation of any kind is required. **That is the
default path, and it stays the reference.**

When you do
------------
The number of valence (no-core-hole) roots below the core manifold grows
combinatorially with the valence active space, because every valence
state must be converged before the Davidson reaches the first core-excited
one:

    valence CAS          total CAS     roots to reach core manifold
    3d only               (7e,8o)                6
    3d + 4d              (7e,13o)               11
    3d + 3 Cl lone pairs (13e,11o)            3,921   (~72 GB of vectors)
    3d + 6 ligands       (19e,15o)          731,809

So the moment ligand orbitals enter the valence space -- which is the
normal thing to want, since that is where covalency and charge-transfer
satellites live -- brute-force diagonalization stops being an option. This
solver restricts the determinant space to exactly `nhole` holes in the core
block, which makes the core-excited states the LOWEST roots of the
restricted problem. For the demo case above that shrinks the space from
3920 determinants to 105.

What the approximation costs
-----------------------------
CVS drops the (small) mixing between core-hole determinants and the
2p-full determinants. Measured on TiCl3 CAS(7e,8o), where both the exact
and the restricted answer are computable:

    exact core-hole roots (eV): 468.8624 469.4258 469.4291 470.3896 ...
    CVS  core-hole roots  (eV): 468.8717 469.4340 469.4373 470.3990 ...
    max deviation             : 9.4 meV

i.e. ~10 meV against a ~469 eV excitation energy. The 2p population of the
CVS roots is exactly 5.0, and the spin sectors come out clean.

Implementation
---------------
`CVSFCISolver` subclasses `direct_spin1.FCISolver` and applies a
determinant mask on both sides of every Hamiltonian contraction. The
Davidson subspace then stays inside the restricted space: the initial guess
is built there, `contract_2e` projects, and the diagonal preconditioner is
elementwise so it cannot reintroduce excluded determinants.
"""
from __future__ import annotations

import numpy as np
from pyscf.fci import cistring, direct_spin1
from pyscf.fci.direct_spin1 import _unpack_nelec


def core_occupancy(norb, nelec_spin, core_idx):
    """Number of `core_idx` orbitals occupied in each FCI string of one spin."""
    strings = cistring.make_strings(range(norb), nelec_spin)
    return np.array([sum((s >> i) & 1 for i in core_idx) for s in strings])


class CVSFCISolver(direct_spin1.FCISolver):
    """FCI restricted to determinants with exactly `nhole` core holes.

    Attributes
    ----------
    core_idx : list[int]
        Indices, WITHIN the active space, of the core (e.g. metal 2p)
        orbitals. This is the `cas_2p_local` produced by step 03.
    nhole : int
        Number of core holes to enforce. 1 for one-photon XAS/RIXS.
    """

    core_idx = None
    nhole = 1

    def determinant_mask(self, norb, nelec):
        if not self.core_idx:
            raise RuntimeError('CVSFCISolver.core_idx must be set')
        na, nb = _unpack_nelec(nelec)
        pa = core_occupancy(norb, na, self.core_idx)
        pb = core_occupancy(norb, nb, self.core_idx)
        target = 2 * len(self.core_idx) - self.nhole
        return (pa[:, None] + pb[None, :]) == target

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        mask = self.determinant_mask(norb, nelec)
        ci = np.asarray(fcivec).reshape(mask.shape) * mask
        hc = super().contract_2e(eri, ci, norb, nelec, link_index, **kwargs)
        return np.asarray(hc).reshape(mask.shape) * mask

    def get_init_guess(self, norb, nelec, nroots, hdiag):
        """Lowest-diagonal determinants *within* the restricted space."""
        mask = self.determinant_mask(norb, nelec).ravel()
        idx = np.where(mask)[0]
        if len(idx) < nroots:
            raise RuntimeError(
                f'CVS space holds only {len(idx)} determinants but {nroots} '
                f'roots were requested (norb={norb}, nelec={nelec}, '
                f'nhole={self.nhole})')
        order = idx[np.argsort(hdiag[idx])]
        guess = []
        for i in range(nroots):
            x = np.zeros(hdiag.size)
            x[order[i]] = 1.0
            guess.append(x)
        return guess

    def space_size(self, norb, nelec):
        """(restricted, full) determinant counts -- useful for logging."""
        mask = self.determinant_mask(norb, nelec)
        return int(mask.sum()), int(mask.size)


def attach_cvs(mc, core_idx, nhole=1, nroots=1):
    """Swap a CASCI object's solver for the CVS-restricted one."""
    solver = CVSFCISolver(mc.mol)
    solver.core_idx = list(core_idx)
    solver.nhole = nhole
    solver.nroots = nroots
    mc.fcisolver = solver
    return mc
