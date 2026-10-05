"""Per-system configuration for the L-edge XAS/RIXS workflow.

Every driver script (01-06) takes `--system=NAME` and reads its parameters
from here. The default is `ticl3`, so the existing demo and everything
already in `run/` keep working untouched.

Adding a system means adding an entry here, not editing the drivers. What
an entry has to pin down is exactly the set of things that are NOT
discoverable from the files themselves:

  * `spin` / `charge`    -- the .gbw knows them but the PySCF Mole must be
                            built before the orbitals are read.
  * `nelec_valence` /    -- cross-checked in step 02 against the
    `norb_valence`          natural-orbital occupations in the .fch, which
                            is how a mismatched active space gets caught.
  * `n_valence_roots` /  -- how much of each manifold to solve for. These
    `nroots_core`           are counted from the determinant space, not
                            guessed; see the comments on each entry.
  * `spins`              -- which total spins the core-excited manifold
                            carries. Only used by `--per-spin`; the
                            default single-Ms-block path finds them all on
                            its own.

The `metal` label is what `common.ao_character` matches against AO labels
('Ti 2p', 'Cr 3d'), so it must be the element symbol as PySCF prints it.

Trying a system without editing this file
-----------------------------------------
`--system=` also accepts a path to a .toml or .json file, or to a
directory holding `system.toml` / `system.json`. The file carries the same
keys as an entry below; see `system.example.toml` next to this module.
In a file-defined system, relative paths resolve against the file's own
directory, `run_dir` defaults to that directory, and `orca_dir` defaults
to `run_dir` -- so a self-contained folder of .xyz / .gbw / .inp plus one
`system.toml` is all a new machine needs:

    python scripts/02_import_orbitals.py --system=/scratch/me/fecl4/
"""
from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))

REQUIRED = ('orca_job', 'gbw', 'xyz', 'basis', 'charge', 'spin', 'metal',
            'nelec_valence', 'norb_valence', 'n_valence_roots',
            'nroots_core')
OPTIONAL = dict(spins=None, soc_max_bytes=200 * 1024 ** 2)

SYSTEMS = {
    # -----------------------------------------------------------------
    # TiCl3 -- Ti(III), 3d^1. The shipped demo; runs in minutes.
    # -----------------------------------------------------------------
    'ticl3': dict(
        run_dir='run',
        orca_job='ticl3_casscf',       # step 01 runs orca/<orca_job>.inp
        gbw='ticl3_casscf',            # .gbw holding the CASSCF orbitals
        xyz='ticl3_opt.xyz',
        basis='def2-svp', charge=0, spin=1,
        metal='Ti',
        nelec_valence=1, norb_valence=5,
        # d^1 ligand-field manifold: 5 states, all doublets.
        n_valence_roots=5,
        # 2p^5 3d^2 is 60 microstates -> 6 quartets + 24 doublets = 30.
        nroots_core=30,
        spins=[0.5, 1.5],
        # 3 x 85^4 x 8 = 1.25 GB; 200 MB batches keep it off a laptop's swap.
        soc_max_bytes=200 * 1024 ** 2,
    ),

    # -----------------------------------------------------------------
    # Cr(acac)3 -- Cr(III), 3d^3, 43 atoms, 430 AOs in def2-SVP.
    # Orbitals come from an externally-run SA-CASSCF(3,5); step 01 only
    # re-diagonalizes in them to produce the gate-2 reference.
    # -----------------------------------------------------------------
    'cracac': dict(
        run_dir='run/cr/cracac',
        orca_job='cracac_casci_ref',
        gbw='cracac-cas35',
        xyz='cracac-r2scan3c-opt1.xyz',
        basis='def2-svp', charge=0, spin=3,
        metal='Cr',
        nelec_valence=3, norb_valence=5,
        # d^3 ligand-field manifold: C(10,3) = 120 microstates
        #   = 10 quartets (4F + 4P) + 40 doublets (2G 2H 2P 2D 2D 2F).
        # The Ms=1/2 CVS(nhole=0) space is exactly 50 determinants, so 50
        # roots IS the complete valence manifold, not a truncation.
        n_valence_roots=50,
        # 2p^5 3d^4: 6 x C(10,4) = 1260 microstates
        #   = 285 doublets + 150 quartets + 15 sextets = 450 states.
        # The Ms=1/2 CVS(nhole=1) space is exactly 450 determinants, so
        # again this is the complete manifold. Verified: one solve returns
        # 285/150/15 spin-pure to 1.2e-14 in 11 s.
        nroots_core=450,
        # 2p^5 3d^4 carries S = 1/2, 3/2 AND 5/2 -- the sextets are new
        # relative to the d^1 demo. All three appear in the Ms=1/2 block,
        # so --per-spin is never required; this list is for when it is used.
        spins=[0.5, 1.5, 2.5],
        # 3 x 430^4 x 8 = 764 GB total. Sized for a 512 GB node; drop it
        # with --soc-max-bytes= on a smaller machine.
        soc_max_bytes=16 * 1024 ** 3,
    ),

    # -----------------------------------------------------------------
    # [Cr(ddpd)2]3+ -- Cr(III), 3d^3, 79 atoms, 1613 AOs in def2-TZVP.
    # Same d^3 counting as Cr(acac)3, but a trication and a triple-zeta
    # basis: charge=3 and basis are read off the .gbw (CHAR_MULT 3 4;
    # Cr 6s4p4d1f, C/N 5s3p2d1f, H 3s1p), not assumed. The .gbw IS the
    # finished SA-CASSCF(3,5), so there is no step 01 for this system:
    # start at step 02, which only needs the 2p rotated in by step 03.
    # Step 02's ORCA-root comparison runs only if the original ORCA
    # output is dropped next to the .gbw as crddpd-cas35-1.out.
    # -----------------------------------------------------------------
    'crddpd': dict(
        run_dir='run/cr/crddpd',
        orca_job='crddpd-cas35-1',
        gbw='crddpd-cas35-1',
        xyz='crddpd-r2-opt1-reorient.xyz',
        basis='def2-tzvp', charge=3, spin=3,
        metal='Cr',
        nelec_valence=3, norb_valence=5,
        # d^3: 10 quartets + 40 doublets, complete (see cracac).
        n_valence_roots=50,
        # 2p^5 3d^4: 285 doublets + 150 quartets + 15 sextets, complete.
        nroots_core=450,
        spins=[0.5, 1.5, 2.5],
        # 3 x 1613^4 x 8 = 162 TB total, and one AO of the batched index
        # is already 3 x 1613^3 x 8 = 101 GB -- so --somf=batched is not
        # usable here; this cap only matters if someone asks for it.
        soc_max_bytes=16 * 1024 ** 3,
    ),

    # -----------------------------------------------------------------
    # [Cr(ddpd)2]3+ from SA-CASSCF(7,7) orbitals: the five Cr 3d plus two
    # doubly-occupied ligand orbitals. Same geometry, basis and charge as
    # 'crddpd'; only the .gbw and the active space differ. The .gbw is
    # again the finished CASSCF, so start at step 02. Combined with the
    # 2p this is CASCI(13e,10o), and --cvs is no longer optional.
    #
    # Unlike the 3d-only entries, the root counts below ARE truncations.
    # The Ms=1/2 CVS spaces are 1225 (no hole) and 5880 (one hole)
    # determinants, i.e. the d^3 / 2p^5 d^4 manifolds plus every LMCT
    # state on top of them. 50 / 450 are the sizes of the pure d
    # manifolds, kept so the result compares one-to-one with 'crddpd';
    # whether the lowest 50 / 450 roots really are those manifolds, or
    # LMCT states intrude, has to be read off the step 02 / 04 logs.
    # Override with --nvalence= / --nroots= on step 04.
    # -----------------------------------------------------------------
    'crddpd-cas77': dict(
        run_dir='run/cr/crddpd/cas77',
        orca_job='crddpd-cas77-1',
        gbw='crddpd-cas77-1',
        xyz='crddpd-r2-opt1-reorient.xyz',
        basis='def2-tzvp', charge=3, spin=3,
        metal='Cr',
        nelec_valence=7, norb_valence=7,
        n_valence_roots=50,
        nroots_core=450,
        spins=[0.5, 1.5, 2.5],
        # --somf=batched is not usable at 1613 AOs (see 'crddpd').
        soc_max_bytes=16 * 1024 ** 3,
    ),

    # -----------------------------------------------------------------
    # 'crddpd-cas77' again, for step 04 --dense --window=: the 450-root
    # Davidson there lost roots 302-449 and cut the core manifold off at
    # 7.6 eV. Same active_space.pkl (a symlink into ../cas77), separate
    # run_dir so those results stay. nroots_core is not used by --dense;
    # the core roots are whatever falls inside the window.
    # -----------------------------------------------------------------
    'crddpd-cas77-dense': dict(
        run_dir='run/cr/crddpd/cas77_dense',
        orca_job='crddpd-cas77-1',
        gbw='crddpd-cas77-1',
        xyz='crddpd-r2-opt1-reorient.xyz',
        basis='def2-tzvp', charge=3, spin=3,
        metal='Cr',
        nelec_valence=7, norb_valence=7,
        n_valence_roots=50,
        nroots_core=450,
        spins=[0.5, 1.5, 2.5],
        soc_max_bytes=16 * 1024 ** 3,
    ),
}

DEFAULT = 'ticl3'


def _parse_bytes(v):
    """A plain integer, or a string like '16g' / '200m' / '512k'."""
    if isinstance(v, (int, float)):
        return int(v)
    v = v.strip().lower()
    mult = {'g': 1024 ** 3, 'm': 1024 ** 2, 'k': 1024}.get(v[-1], 1)
    return int(float(v[:-1] if mult > 1 else v) * mult)


def _load_file(path):
    """Read a system definition from .toml / .json, or a directory with one."""
    if os.path.isdir(path):
        for fn in ('system.toml', 'system.json'):
            if os.path.isfile(os.path.join(path, fn)):
                path = os.path.join(path, fn)
                break
        else:
            raise SystemExit(f'--system={path!r} is a directory with no '
                             'system.toml or system.json in it')
    path = os.path.abspath(path)
    if path.endswith('.toml'):
        import tomllib  # stdlib from Python 3.11
        with open(path, 'rb') as fh:
            raw = tomllib.load(fh)
    else:
        with open(path) as fh:
            raw = json.load(fh)

    base = os.path.dirname(path)
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise SystemExit(f'{path}: missing required key(s) {missing}')
    unknown = set(raw) - set(REQUIRED) - set(OPTIONAL) - {'name', 'run_dir',
                                                          'orca_dir'}
    if unknown:
        raise SystemExit(f'{path}: unknown key(s) {sorted(unknown)}')

    cfg = {**OPTIONAL, **raw}
    cfg.setdefault('name', os.path.basename(base))
    cfg['run_dir'] = os.path.join(base, cfg.get('run_dir', '.'))
    cfg['orca_dir'] = os.path.join(base, cfg.get('orca_dir', cfg['run_dir']))
    return cfg


def from_argv(argv, default=DEFAULT):
    """Pull `--system=NAME|PATH` out of argv and return its config.

    `run_dir` comes back absolute, and `xyz_path` / `gbw` are resolved
    against it, so drivers never have to do path arithmetic.
    """
    name = default
    for a in argv[1:]:
        if a.startswith('--system='):
            name = a.split('=', 1)[1]
    if name in SYSTEMS:
        cfg = {**OPTIONAL, **SYSTEMS[name]}
        cfg['name'] = name
        cfg['run_dir'] = os.path.join(_ROOT, cfg['run_dir'])
        cfg['orca_dir'] = os.path.join(_ROOT, 'orca')
    elif os.path.exists(name):
        cfg = _load_file(name)
    else:
        raise SystemExit(f'unknown --system={name!r}: not a known system '
                         f'{sorted(SYSTEMS)} and not a path to a .toml/.json '
                         'file or a directory holding system.toml')
    cfg['run_dir'] = os.path.normpath(cfg['run_dir'])
    cfg['orca_dir'] = os.path.normpath(cfg['orca_dir'])
    cfg['spins'] = list(cfg['spins'] or [])
    cfg['soc_max_bytes'] = _parse_bytes(cfg['soc_max_bytes'])
    cfg['xyz_path'] = os.path.join(cfg['run_dir'], cfg['xyz'])
    return cfg


def soc_max_bytes(cfg, argv):
    """`--soc-max-bytes=` override, accepting a plain integer or NNg/NNm."""
    for a in argv[1:]:
        if a.startswith('--soc-max-bytes='):
            return _parse_bytes(a.split('=', 1)[1])
    return cfg['soc_max_bytes']
