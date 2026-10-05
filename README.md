# mcxray — multiconfigurational X-ray spectroscopy

Transition-metal L-edge X-ray absorption (XAS) and resonant inelastic
X-ray scattering (RIXS) from multiconfigurational wavefunctions:

**ORCA** valence SA-CASSCF orbitals
→ **MOKIT** orbital transfer into PySCF
→ metal **2p core** spliced into the active space
→ core-valence-separated **CI**, every multiplicity the manifold carries
→ **MC-PDFT** state energies
→ **Breit-Pauli SOMF spin-orbit coupling**
→ XAS stick spectrum and Kramers-Heisenberg RIXS map.

Why each step is done the way it is, what was validated and what the
worked examples cost is in [`docs/method.md`](docs/method.md).

## Code here, calculations elsewhere

This repository holds code only. A calculation is a directory anywhere on
disk with a `system.toml` next to its inputs; every step reads from and
writes into that directory.

```
mcxray/                      this repository
  src/mcxray/                the package
    common.py                dipoles, broadening, Kramers-Heisenberg
    cvs_fci.py               core-hole-restricted FCI solver
    soc.py                   SOMF operator, spin ladder, SOC matrix
    orbital_io.py            MOKIT bridge and import validation
    systems.py               reads system.toml
    cli/                     one module per pipeline step
  tests/                     TiCl3 regression tests
  examples/ticl3/            the demo system; system.example.toml is a template
  docs/method.md             method notes, validation gates, measured costs
  notebooks/                 TiCl3 walk-through
  legacy/                    superseded ROHF-based scripts, unmaintained

projects/                    NOT in this repository
  crddpd/cas77_dense/
    system.toml  geometry  .gbw  submit scripts  →  outputs
```

## Install

```bash
pip install -e .            # into the environment that has pyscf + pyscf-forge
pip install -e '.[test]'    # the same, plus pytest
```

The install is editable, so the checkout is what runs, and every step
prints the code version and git commit as the first line of its log:

```
mcxray 0.1.0 (git ba4b0bd) | system crddpd-cas77-dense | /…/projects/crddpd/cas77_dense
```

A `+` after the hash means the checkout had uncommitted changes.

Steps 01–02 additionally need ORCA (`orca`, `orca_2mkl`) and a
`mokit-bridge` conda env; see "Orbital transfer" in `docs/method.md`.
Steps 03–07 need PySCF only.

## Running a calculation

```bash
mcxray 04 --system=/path/to/projects/fecl4 --cvs --nproc=64
cd /path/to/projects/fecl4 && mcxray 04 --cvs --nproc=64     # the same
python -m mcxray.cli 04 --system=…                            # the same
```

`--system=` takes a directory holding `system.toml`, the file itself, or
the name of a shipped example (`--system=ticl3`). Without it the current
directory is used. Run order is `01 → 06`; each step reads what the
previous one wrote.

| step | does | main options |
|---|---|---|
| `mcxray 01` | Run/parse the ORCA valence SA-CASSCF (`<orca_job>.inp`). | `--force` |
| `mcxray 02` | MOKIT orbital transfer and the two import gates. | `--reopt` |
| `mcxray 03` | Splice the metal 2p into the CASSCF active space. | |
| `mcxray 04` | CI, spin-purity filter, MC-PDFT energies. | `--cvs`, `--dense`, `--window=`, `--nroots=`, `--nvalence=`, `--ms=xms\|cms\|lpdft`, `--per-spin`, `--nproc=` |
| `mcxray 05` | SOMF operator, Ms ladder, SOC matrix, spin-orbit eigenstates. | `--somf=`, `--soc-max-bytes=` |
| `mcxray 05b` | Spin-orbit states for a second set of state energies. | `--tag=` |
| `mcxray 06` | XAS and RIXS from the spin-orbit states. | `--tag=`, `--pol=`, `--theta=` |
| `mcxray 07` | RIXS resolved by initial state. | `--tag=` |
| `mcxray compare-pdft` | SS-PDFT vs XMS-PDFT report (TiCl₃). | `--from-npz` |
| `mcxray plot-xas` | Calculated XAS against a measured spectrum. | `--exp=`, `--tags=`, `--gamma=` |

Each step's module docstring (`src/mcxray/cli/stepNN_*.py`) documents its
options in full.

### A new system

Copy `examples/system.example.toml` to `system.toml` in a new directory,
put the `.xyz` and the CASSCF `.gbw` beside it, and start at step 02 if
the `.gbw` is already the finished CASSCF (step 01 only runs ORCA).

### The shipped example

```bash
mcxray 01 --system=ticl3     # needs ORCA; then 02 … 06
```

TiCl₃ (Ti(III), 3d¹, 85 AOs) runs in minutes. Outputs land in
`examples/ticl3/` and are ignored by git.

## Tests

```bash
pytest                    # everything, about two minutes
pytest -m "not slow"      # seconds: no electronic structure
```

The slow tests run TiCl₃ from step 04 through step 06 (CVS, two worker
processes) and the dense solver, and compare energies, spin-orbit states
and spectra with reference numbers produced by the scripts as they were
before they became this package. `tests/data/ticl3_active_space.pkl` is
the output of steps 01–03, so the tests need neither ORCA nor MOKIT.

## Requirements

Python ≥ 3.11, `pyscf` 2.14, `pyscf-forge` (for `pyscf.mcpdft`), `numpy`,
`matplotlib`. No GPU. Memory and run-time figures for the worked examples
are in `docs/method.md`.
