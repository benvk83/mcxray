from __future__ import annotations

import os

import pytest

from mcxray import systems
from mcxray.cli import STEPS

TOML = '''
orca_job = "job"
gbw = "job"
xyz = "geom.xyz"
basis = "def2-svp"
charge = 0
spin = 1
metal = "Ti"
nelec_valence = 1
norb_valence = 5
n_valence_roots = 5
nroots_core = 30
'''


def write(tmp_path, text=TOML):
    (tmp_path / 'system.toml').write_text(text)
    return str(tmp_path)


def test_shipped_example_by_name():
    cfg = systems.from_argv(['x', '--system=ticl3'], quiet=True)
    assert cfg['name'] == 'ticl3'
    assert cfg['run_dir'] == os.path.join(systems.EXAMPLES, 'ticl3')
    assert os.path.isfile(cfg['xyz_path'])
    assert cfg['soc_max_bytes'] == 200 * 1024 ** 2


def test_directory_and_defaults(tmp_path):
    cfg = systems.from_argv(['x', f'--system={write(tmp_path)}'], quiet=True)
    assert cfg['name'] == tmp_path.name
    assert cfg['run_dir'] == cfg['orca_dir'] == str(tmp_path)
    assert cfg['xyz_path'] == str(tmp_path / 'geom.xyz')
    assert cfg['spins'] == []


def test_current_directory_is_the_default(tmp_path, monkeypatch):
    write(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert systems.from_argv(['x'], quiet=True)['run_dir'] == str(tmp_path)


def test_no_system_file_is_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match='no system.toml'):
        systems.from_argv(['x'], quiet=True)
    with pytest.raises(SystemExit, match='unknown --system'):
        systems.from_argv(['x', '--system=no-such-system'], quiet=True)


def test_bad_keys_are_rejected(tmp_path):
    d = write(tmp_path, TOML + 'nroots = 3\n')
    with pytest.raises(SystemExit, match='unknown key'):
        systems.from_argv(['x', f'--system={d}'], quiet=True)
    d = write(tmp_path, TOML.replace('metal = "Ti"\n', ''))
    with pytest.raises(SystemExit, match='missing required'):
        systems.from_argv(['x', f'--system={d}'], quiet=True)


def test_banner_names_version_and_system(tmp_path, capsys):
    systems.from_argv(['x', f'--system={write(tmp_path)}'])
    out = capsys.readouterr().out
    assert out.startswith('mcxray ') and str(tmp_path) in out


@pytest.mark.parametrize('text, n', [('200m', 200 * 1024 ** 2),
                                     ('16g', 16 * 1024 ** 3), (123, 123)])
def test_parse_bytes(text, n):
    assert systems._parse_bytes(text) == n


def test_every_step_module_exists():
    import importlib.util
    for mod, _ in STEPS.values():
        assert importlib.util.find_spec(f'mcxray.cli.{mod}') is not None
