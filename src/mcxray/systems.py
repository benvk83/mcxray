"""Per-system configuration for the L-edge XAS/RIXS workflow.

The code knows nothing about any particular calculation. A calculation is
a directory holding a `system.toml` (or `system.json`) next to its inputs
(.xyz, the .gbw with the CASSCF orbitals, optionally the ORCA .inp/.out),
and every step is pointed at that directory:

    mcxray 02 --system=/path/to/projects/fecl4
    cd /path/to/projects/fecl4 && mcxray 02        # same thing

`--system=` accepts a directory, a .toml/.json file, or the name of a
folder in this repository's `examples/` (so `--system=ticl3` is the
shipped demo). Without `--system=` the current directory is used.
Relative paths in the file resolve against the file's own directory,
`run_dir` defaults to that directory (outputs land there) and `orca_dir`
defaults to `run_dir`. `examples/system.example.toml` is a template.

What a system file has to pin down is exactly the set of things that are
NOT discoverable from the files themselves:

  * `spin` / `charge`    -- the .gbw knows them but the PySCF Mole must be
                            built before the orbitals are read.
  * `nelec_valence` /    -- cross-checked in step 02 against the
    `norb_valence`          natural-orbital occupations in the .fch, which
                            is how a mismatched active space gets caught.
  * `n_valence_roots` /  -- how much of each manifold to solve for. These
    `nroots_core`           are counted from the determinant space, not
                            guessed; see `examples/ticl3/system.toml`.
  * `spins`              -- which total spins the core-excited manifold
                            carries. Only used by `--per-spin`; the
                            default single-Ms-block path finds them all on
                            its own.

The `metal` label is what `common.ao_character` matches against AO labels
('Ti 2p', 'Cr 3d'), so it must be the element symbol as PySCF prints it.
"""
from __future__ import annotations

import json
import os

import mcxray

_HERE = os.path.dirname(os.path.abspath(__file__))
# <repo>/examples in a source checkout or an editable install.
EXAMPLES = os.path.abspath(os.path.join(_HERE, '..', '..', 'examples'))
_SYSTEM_FILES = ('system.toml', 'system.json')

REQUIRED = ('orca_job', 'gbw', 'xyz', 'basis', 'charge', 'spin', 'metal',
            'nelec_valence', 'norb_valence', 'n_valence_roots',
            'nroots_core')
OPTIONAL = dict(spins=None, soc_max_bytes=200 * 1024 ** 2)


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


def _has_system_file(path):
    return any(os.path.isfile(os.path.join(path, fn)) for fn in _SYSTEM_FILES)


def resolve(name=None):
    """Path of the system file or directory that `--system=NAME` means."""
    if name is None:
        if _has_system_file(os.getcwd()):
            return os.getcwd()
        raise SystemExit(
            'no --system= given and no system.toml in the current '
            'directory. Pass --system=DIR (a folder holding system.toml), '
            'or cd into it.')
    if os.path.exists(name):
        return name
    example = os.path.join(EXAMPLES, name)
    if os.path.isdir(example):
        return example
    shipped = (sorted(d for d in os.listdir(EXAMPLES)
                      if _has_system_file(os.path.join(EXAMPLES, d)))
               if os.path.isdir(EXAMPLES) else [])
    raise SystemExit(f'unknown --system={name!r}: not a path to a '
                     '.toml/.json file or a directory holding system.toml, '
                     f'and not a shipped example {shipped}')


def from_argv(argv, quiet=False):
    """Pull `--system=PATH` out of argv and return its config.

    `run_dir` comes back absolute, and `xyz_path` is resolved against it,
    so drivers never have to do path arithmetic. Unless `quiet`, one line
    naming the code version and the system is printed, so every job log
    records which code produced it.
    """
    name = None
    for a in argv[1:]:
        if a.startswith('--system='):
            name = a.split('=', 1)[1]
    cfg = _load_file(resolve(name))
    cfg['run_dir'] = os.path.normpath(cfg['run_dir'])
    cfg['orca_dir'] = os.path.normpath(cfg['orca_dir'])
    cfg['spins'] = list(cfg['spins'] or [])
    cfg['soc_max_bytes'] = _parse_bytes(cfg['soc_max_bytes'])
    cfg['xyz_path'] = os.path.join(cfg['run_dir'], cfg['xyz'])
    if not quiet:
        print(f'{mcxray.version_string()} | system {cfg["name"]} | '
              f'{cfg["run_dir"]}', flush=True)
    return cfg


def soc_max_bytes(cfg, argv):
    """`--soc-max-bytes=` override, accepting a plain integer or NNg/NNm."""
    for a in argv[1:]:
        if a.startswith('--soc-max-bytes='):
            return _parse_bytes(a.split('=', 1)[1])
    return cfg['soc_max_bytes']
