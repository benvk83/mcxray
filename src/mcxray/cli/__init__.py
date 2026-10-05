"""Command line: `mcxray STEP [options]`.

Each step is a module in this package and can equally be run as
`python -m mcxray.cli.<module>`; this dispatcher only saves the typing.
The step's own options (and its docstring) are in that module.
"""
from __future__ import annotations

import runpy
import sys

STEPS = {
    '01': ('step01_orca_casscf', 'ORCA valence SA-CASSCF'),
    '02': ('step02_import_orbitals', 'MOKIT orbital import + gates'),
    '03': ('step03_build_active_space', 'splice the metal 2p into the CAS'),
    '04': ('step04_casci_pdft', 'CASCI + MC-PDFT state energies'),
    '05': ('step05_soc', 'spin-orbit coupling'),
    '05b': ('step05b_soc_swap_energies', 'redo SOC with other state energies'),
    '06': ('step06_xas_rixs', 'XAS + RIXS'),
    '07': ('step07_rixs_by_initial_doublet', 'RIXS resolved by initial state'),
    'compare-pdft': ('compare_pdft_variants', 'SS-PDFT vs XMS-PDFT report'),
    'plot-xas': ('plot_xas_vs_experiment', 'calculated XAS vs a measurement'),
}


def usage():
    lines = ['usage: mcxray STEP [--system=DIR] [options]',
             '       mcxray --version', '', 'steps:']
    lines += [f'  {k:<13}{what:<38}mcxray.cli.{mod}'
              for k, (mod, what) in STEPS.items()]
    lines += ['', '--system= is a directory holding system.toml (default: '
              'the current directory).']
    return '\n'.join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ('-h', '--help'):
        print(usage())
        return 0
    if argv[0] == '--version':
        import mcxray
        print(mcxray.version_string())
        return 0
    if argv[0] not in STEPS:
        print(f'mcxray: unknown step {argv[0]!r}\n\n{usage()}', file=sys.stderr)
        return 2
    # alter_sys puts the step in sys.modules['__main__'] with its real
    # module spec, which is what lets step 04's spawned workers re-import it.
    sys.argv = [sys.argv[0]] + argv[1:]
    runpy.run_module(f'mcxray.cli.{STEPS[argv[0]][0]}', run_name='__main__',
                     alter_sys=True)
    return 0

