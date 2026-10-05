# Method, validation and measured costs

Background for the `mcxray` pipeline: why each step is done the way it is,
what was checked, and what the two worked examples cost. For installation
and how to run a calculation see the top-level `README.md`. Paths such as
`cracac/submit_456.sh` refer to the calculation directories, which live
outside this repository.

A validated pipeline for transition-metal L-edge X-ray absorption (XAS) and
resonant inelastic X-ray scattering (RIXS):

**ORCA** valence-only SA-CASSCF (metal 3d, optionally + ligands)
→ **MOKIT** orbital transfer into PySCF
→ splice the metal **2p core** into the active space
→ **multi-spin CASCI** (every multiplicity the manifold carries)
→ **MC-PDFT** state energies
→ **Breit-Pauli SOMF spin-orbit coupling**
→ XAS stick spectrum + Kramers-Heisenberg RIXS map.

Two worked examples:

| system | | |
|---|---|---|
| **TiCl₃** | pyramidal Ti(III), 3d¹, 4 atoms, 85 AOs | runs in minutes; the shipped example |
| **Cr(acac)₃** | near-octahedral Cr(III), 3d³, 43 atoms, 430 AOs | ~6 h measured on 14 cores; a cluster job |

TiCl₃ is small enough to iterate on and open-shell enough to give genuine
multiplet structure. Cr(acac)₃ is the real-size case: a d³ ion whose ground
state is a *quartet*, so the valence manifold carries doublets **and**
quartets and the 2p⁵3d⁴ core-excited manifold carries doublets, quartets
**and sextets**.

## Why the orbitals come from a valence CASSCF

The 2p core is deliberately *absent* from the ORCA CASSCF. Orbitals are
optimized for the valence d-manifold, where CASSCF relaxation matters a
great deal, and the core is added afterwards for the CASCI that produces
the core-excited states. The difference is not subtle — the same active
space, the same geometry, the same basis, only the orbitals differ:

| Ti(III) ligand-field states | ROHF orbitals | **SA-CASSCF orbitals** |
|---|---|---|
| d-d excitation energies (eV) | 0, 1.561, 1.563, 3.061, 3.064 | **0, 0.315, 0.317, 1.636, 1.639** |

Orbital relaxation roughly halves the d-d energies.

MC-PDFT (tPBE) supplies the state energies throughout, avoiding the
intruder-state problems NEVPT2 suffers for core excitations.

## Orbital transfer: use MOKIT, not molden

ORCA's molden export **cannot** be made to work for d/f-block elements, and
this is now measured rather than asserted. `max|CᵀSC − I|` for TiCl₃/def2-SVP:

| route | deviation |
|---|---|
| raw `pyscf.tools.molden.load` | 1.58 |
| + primitive-normalization "fix" | **5.55** (worse — def2-SVP on Ti has f functions) |
| best over all 4096 d/f sign conventions | 0.83 |
| **MOKIT (`.gbw → .mkl → .fch → fch2py`)** | **1.97e-07** |

An exhaustive search over every d and f phase convention (2⁵ × 2⁷ patterns,
on both the raw and normalization-corrected basis) bottoms out at 0.83, so
this is not a fixable sign convention. The residual 1.97e-07 from MOKIT is
the `.fch` `E16.8` text-precision floor, not an error; a Löwdin cleanup
takes it to ~4e-15.

**Setup (once):**

```bash
conda create -y -n mokit-bridge -c mokit -c conda-forge python=3.12 mokit
conda run -n mokit-bridge pip install pyscf      # fch2py needs it
```

MOKIT's conda builds stop at python 3.12 while this workflow runs on 3.13,
so `fch2py` executes in that env behind `mcxray/_mokit_bridge.py`. The main
environment never imports MOKIT.

Note `mkl2fch -no` is **required** for a CASSCF `.gbw`: those store
state-averaged natural orbitals with fractional occupations, and without
`-no` mkl2fch's occupied-orbital count disagrees with the electron count and
it aborts.

## Method summary

1. **Active space.** `[inactive | 2p + valence active | virtual]`. For the
   demo that is CAS(7e,8o) with ncore=33. Both the no-hole manifold
   (2p⁶3d¹ — the RIXS *final* states) and the core-hole manifold (2p⁵3d² —
   the XAS/RIXS *intermediate* states) are roots of one CI problem, so every
   transition dipole is between vectors in the same space.
2. **Multiple spin blocks.** For a d¹ ion the 2p⁵3d² manifold carries both
   doublets and quartets, and with SA-CASSCF orbitals the **quartets are the
   lowest core-excited states** (Hund's rule) — the opposite of what ROHF
   orbitals gave. They are rigorously dark without SOC, so the previous
   version of this workflow discarded them; they cannot be discarded now.
3. **Spin purity is filtered, not assumed.** `direct_spin1` conserves Sz but
   not S². Measured on a 30-root solve, roots 0–23 are pure to 1e-9 or
   better in 2S+1 while roots 24–29 — the unconverged top of the Davidson
   window — deviate by 6e-4 to 0.46. The tail is discarded and the count
   reported. No S² penalty is used: `fix_spin_` cost 9× at Ms=3/2 (26.3 s vs
   2.9 s) and failed to converge at all at Ms=1/2, while solving nothing
   this calculation suffers from.
4. **Energies vs. wavefunctions.** MC-PDFT refines the energy of a CASCI
   density but does not define a wavefunction, so anything needing a matrix
   element *between* two states — transition dipoles, SOC couplings — uses
   the CASCI CI vectors. MC-PDFT enters only on the diagonal.
5. **MC-PDFT variant.** Default is **state-specific (SS-PDFT)** with the
   **tPBE** on-top functional — each root's density through the functional
   independently, no state interaction. `--ms=xms|cms|lpdft` switches to
   multi-state MC-PDFT, which matters here because the core-excited
   manifold is dense (19 states in ~6 eV) and SS-PDFT lets near-degenerate
   states cross without interacting. See "Which multi-state variant" below.
6. **Spin-orbit coupling.** Breit-Pauli, with the 2e spin-same/other-orbit
   terms folded into an effective one-electron operator by the standard SOMF
   construction (`h = h1e + Σ D[(ij|kl) − 3/2(il|kj) − 3/2(kj|il)]`). There
   is no SOMF in pyscf, so this is built from `int1e_prinvxp` and
   `int2e_p1vxp1`. The other Ms components SOC needs are generated from the
   solved states with the exact ladder operators S±, which keeps phases
   within a multiplet consistent by construction — and that is what makes
   the SOC matrix come out Hermitian with no Wigner-Eckart bookkeeping.
7. **RIXS** keeps the full 3×3 Cartesian scattering tensor uncontracted
   until after squaring (polarization-summed powder average). **Do not
   contract it to the tensor trace** — the trace keeps only the isotropic
   invariant and vanishes identically for loss features between states of
   different ligand-field symmetry, silently deleting the d-d channels. See
   `common.kramers_heisenberg_map`.

## Which multi-state variant

**XMS.** The reason is measurable rather than aesthetic. TiCl3 is near-C3v,
so the core-excited quartets come in degenerate E pairs. Splitting *within*
those pairs, on the 8-root quartet block:

| | splitting within degenerate pairs |
|---|---|
| CASCI (reference) | ~1–3 meV (the geometry is not exactly C3v) |
| **XMS-PDFT** | **0–1 meV** — degeneracy preserved |
| CMS-PDFT | 12–61 meV — degeneracy broken |

This is not a CMS bug, it is CMS working as designed. Its objective
maximizes the sum of state-specific classical Coulomb energies, i.e. it
deliberately selects *charge-localized* intermediate states; for a
symmetry-degenerate pair that means picking a localized combination over a
symmetry-adapted one. XMS instead diagonalizes the state-averaged Fock
operator, which commutes with the spatial symmetry operations and is
invariant to rotations within a degenerate set. CMS remains the
better-tested variant for the problems it was built for — conical
intersections and charge transfer — but a symmetric multiplet manifold is
not one of them.

Two structural constraints, both enforced in code:

- **`--ms` implies `--cvs`.** `multi_state` calls the underlying
  state-averaged CASCI, which returns the *lowest* nroots. For the
  core-excited manifold those are the valence states, 460 eV away. The CVS
  restriction makes the core-excited states the lowest roots, which is what
  makes a core-excited model space possible at all.
- **Both multiplicities share one model space, and that is safe.** The XMS
  rotation (state-averaged Fock) and the MS-PDFT off-diagonals (CASCI
  Hamiltonian) are both spin-free, so they vanish between different S and
  the rotation is block-diagonal in spin. Every rotated state's ⟨S²⟩ is
  measured and the run aborts if purity is lost, rather than assuming it.
  Selecting one multiplicity with an S² penalty instead does *not* work:
  pyscf's `mspdft` kernel rebuilds the model-space Hamiltonian without the
  penalty and then asserts its eigenvalues match the penalized solver's,
  tripping its own "Sanity fault" check.

A controlled SS-vs-XMS comparison for this system — identical orbitals,
active space, Ms block, CVS model spaces and SOC treatment, with only the
PDFT step differing — is in **`pdft_comparison.md`** in the TiCl₃ run directory, regenerable with
`mcxray compare-pdft --system=ticl3` (`--from-npz` re-renders the report and
the figure `pdft_comparison.png` from cached energies without
recomputing). Summary: the two are indistinguishable in the
valence manifold, differ by up to ~0.35 eV on individual core-excited states
with a mean shift of only +0.010 eV, and give nearly identical spectra. XMS
reproduces CASCI's count of near-degenerate pairs (6) where SS-PDFT loses
one (5), and neither inflates a splitting beyond a few meV.

**The model space must cover the whole manifold.** Unlike SS-PDFT, where
each state is independent and truncating the root list simply gives you
fewer states, MS-PDFT averages and diagonalizes over exactly the roots you
request. Truncating distorts what survives: `--ms=xms --nroots=16` captures
only 10 of the L2 states and reports a 81:19 branching ratio, against 65:35
from the 19-core-state SS-PDFT run. Size `--nroots` to span the full
core-excited manifold before reading branching ratios off an MS-PDFT run.

L-PDFT is wired up too, but this pyscf-forge build has a bug in it: the
object `linear_multi_state` returns reads `self.chk_ci`, which is never set,
and lacks `get_ci_adiabats`. Both are worked around in `_ci_adiabats`.

## Validation gates

Every one of these is checked in code and fatal on failure.

| gate | TiCl₃ | Cr(acac)₃ |
|---|---|---|
| Orbital orthonormality after import | 1.97e-07 → 4.2e-15 (Löwdin) | 2.49e-07 |
| PySCF CASCI reproduces ORCA's roots | ≤ 6.8e-09 Ha per root (5 roots) | ≤ 2.8e-08 Ha per root (50 roots) |
| …state-averaged energy | 2.9e-09 Ha | 1.8e-08 Ha |
| Active-space electron count vs NO occupations | 1e / 5o | 3e / 5o |
| Metal 2p character of the spliced core MOs | 99.4% | 99.4% |
| Spin purity of retained roots | \|2S+1 − n\| < 1e-6 | 1.2e-14 over all 450 core states |
| S± ladder normalization √(S(S+1)−Ms(Ms±1)) | asserted per component | ditto, now through Ms = 5/2 |
| Spin-flip TDM trace = ⟨S₊⟩ | 1.732051 vs √3 exactly | |
| SOC Hamiltonian Hermiticity | 5.2e-19 | |
| Zero-SOC limit vs spin-free energies | 0.00e+00 Ha | |
| ζ(metal 2p): 2e screening must reduce it | 4.051 → 3.536 eV (atomic ref ~3.7) | 6.009 eV (1e); L₃–L₂ = 1.5ζ = 9.01 eV vs ~9.3 expt |
| Batched vs direct 2e SOC contraction | 2.8e-16 relative | |
| Integral-direct (`get_jk`) vs batched 2e SOMF | 4.5e-13 (operator) | ζ 5.3001 vs 5.300 eV |
| Transition dipoles span one Ms block | fatal if not — see below | |

### Two traps in `--per-spin`

Both were latent while there were only two multiplicities and both become
wrong answers rather than crashes with three, so both are now gated:

1. **Double counting.** An Ms = S block holds every multiplet with S′ ≥ S,
   not just S. Cr(acac)₃'s Ms = 3/2 core block returns its 150 quartets
   *and* the 15 sextets, which the Ms = 5/2 block then returns again —
   silently doubling those states' contribution to the spectrum. Roots
   whose 2S+1 does not match the block are now dropped and counted.
2. **Silently zeroed transition dipoles.** States solved in different
   (nα, nβ) blocks have no computable transition density between them, and
   the old code skipped those pairs, leaving zero. That is correct only
   when the two states also differ in total spin. Under `--per-spin` they
   need not: a d³ ground quartet seeded at Ms = 1/2 and the core quartets
   seeded at Ms = 3/2 are the *bright* L-edge transitions, and zeroing them
   gives a spectrum with no intensity and no error anywhere. Step 05 now
   refuses to run on states spanning more than one block.

Neither restricts anything in practice: the default single-Ms-block path
carries every multiplicity at once and is what both examples use.

## Result for TiCl₃

With SOC on, quartets acquire real intensity (f ~ 1e-6 to 6e-5 at 93–99%
quartet character, against 1e-13 to 1e-17 before SOC) and the edge splits:

- L₃ branch: 44 states, 463.3–466.7 eV, Σf = 1.29e-02
- L₂ branch: 16 states, 468.0–469.0 eV, Σf = 6.95e-03
- centroid separation **3.72 eV**; branching ratio **65:35** against the
  statistical 67:33.

## Second example: Cr(acac)₃

Cr(III) is d³, so the ground state is **⁴A₂** — a quartet — and both
manifolds carry more multiplicities than the d¹ demo:

| manifold | microstates | states |
|---|---|---|
| valence 2p⁶3d³ | C(10,3) = 120 | 10 quartets (⁴F, ⁴P) + 40 doublets = **50** |
| core-excited 2p⁵3d⁴ | 6 × C(10,4) = 1260 | 285 doublets + 150 quartets + **15 sextets** = **450** |

The sextets are 2p⁵(²P) × 3d⁴(⁵D) → ⁶P + ⁶D + ⁶F, and they are new
relative to TiCl₃. Because the ground state is a quartet it is now the
core **quartets** that are dipole-bright; the doublets and sextets are
rigorously dark until SOC mixes them in — the reverse of the d¹ case, so
the intensity-borrowing diagnostic is defined against the ground-state
multiplicity rather than hard-coded to "quartet".

**Neither root count is a truncation.** The Ms = 1/2 CVS spaces are
*exactly* 50 and 450 determinants, so asking for 50 and 450 roots solves
the complete manifolds. One Ms = 1/2 solve returns all 450 core states —
285/150/15 by multiplicity, spin-pure to 1.2e-14, in 11 s. No dense
solver, no spin adaptation, and no `--per-spin` is needed: every
multiplicity, sextets included, has an Ms = 1/2 component.

**Do not truncate the core manifold here.** Hund's rule puts the highest
spin lowest, and for 2p⁵3d⁴ that is the sextets: a run stopped at the
8 lowest core roots returns *nothing but* sextets (2p population 5.000,
⁶P/⁶D/⁶F at 588.3–589.9 eV). Every one of them is rigorously dark against
a quartet ground state without SOC, so a truncated run produces an empty
spectrum — the same trap the d¹ demo hit with its quartets, one
multiplicity further along.

The orbitals arrive as an externally-run SA-CASSCF(3,5) `.gbw` with no
`.out`, so step 01 does not re-optimize anything — `cracac/cracac_casci_ref.inp`
reads those orbitals back (`MOREAD` + `NoIter`) and re-diagonalizes the
full d³ manifold in them. That is a *stronger* gate 2 than comparing one
state-averaged number, and it does not require knowing what weights the
original average used. (`maxiter 0` is not a usable CASCI route in ORCA 6:
it skips the CI entirely, prints `ROOT 0: E= 0.0000000000` and dies on an
I/O error. `maxiter 1` runs the initial CI state check at DE = 0.)

Check the active space before trusting any of this. A CAS(3,5) started
from the d shell can quietly rotate the e_g(σ*) pair out and a ligand
orbital in, and the natural-orbital occupations do **not** reveal it —
they sit at exactly 0.600 either way, because a state average over the d³
manifold is spherical. The rotation-invariant number that does reveal it
is the Cr 3d character summed over the five active MOs:

| | Σ %Cr 3d over the active space | e_g in the CAS? |
|---|---|---|
| rotated-out CASSCF | **2.42** (one active MO at 1.8% 3d, two 72% 3d MOs left *outside*) | no |
| correct CASSCF | **4.44** (all five MOs 88–89% 3d, nothing d-like outside) | yes |

Step 03 prints exactly this table. With the correct orbitals the
ligand-field spectrum is textbook: ⁴A₂ at 0, ⁴T₂ at 1.997/2.028/2.034 eV
(10Dq ≈ 2.02 eV), ²E at 2.388/2.388, ²T₁ at 2.487/2.522/2.523, ⁴T₁ at
3.070/3.081/3.169 eV.

**Cost.** Two steps dominate, both measured on 14 cores:

| step | measured @14 cores | why |
|---|---|---|
| 04 SS-PDFT | 28.3 s × 500 states ≈ **4 h** | on-top functional on a 523k-point grid, 43 atoms |
| 05 2e SOC | batched: 53.3 min on 60 cores; **integral-direct (default): 5.1 min** | `int2e_p1vxp1` is 3 × 430⁴ × 8 B = **764 GB** if stored |

### Where the external-code dependencies stop

Only the first two steps need anything outside PySCF, which is what makes
splitting the run across machines easy:

| step | needs |
|---|---|
| 01 | ORCA |
| 02 | ORCA + MOKIT (+ the `mokit-bridge` conda env) |
| 03 | PySCF only |
| 04–06 | PySCF only |

So a cluster without MOKIT is not a problem: run 01–03 on the workstation
that has it, then install `mcxray` on the cluster, stage the single 1.5 MB
`cracac/active_space.pkl` with its `system.toml`, and start at step 04.
`cracac/submit_456.sh` is a SLURM script that does exactly that —
it reads no `.gbw`, no `.xyz` and no ORCA output, because the geometry,
basis, charge, spin and reordered MO coefficients all live in that pickle.
It is resumable (each step is skipped if its `.npz` exists, so a crash in
step 05 does not repeat step 04's hours) and it preflights the
environment, including a dgemm check for the `OPENBLAS_NUM_THREADS` trap
below.

`cracac/run_cracac.sh` runs the whole 01→06 chain instead, for a
machine that does have ORCA and MOKIT.

### Step 04 does not scale with cores, and half of it is one Coulomb build

Measured on `energy_tot` for one Cr(acac)₃ state:

| threads | 4 | 8 | 14 |
|---|---|---|---|
| s/state | 36.4 | 31.6 | 28.7 |

Amdahl on those points gives **serial ≈ 25.6 s, parallel ≈ 43 s** — already
~89% serial at 14 threads, with a floor near 25.6 s/state however many
cores are thrown at it. A 40-core cluster node measured **22.4 s/state**,
*below* that floor, i.e. the gain came from faster individual cores, not
from parallelism. So 500 states ≈ **3.1 h largely regardless of core count.**

`cProfile` says where it goes, and it is half one thing:

| | time | |
|---|---|---|
| `energy_mcwfn` → `get_j` | **15.4 s** | direct Coulomb build, 430 AOs |
| `energy_dft` → `energy_ot` | 14.6 s | on-top functional (10.0 s in `eval_rho`) |

`dm = dm_core + dm_active`, and **`dm_core` is identical for all 500
states** because the orbitals are frozen — only the 8-orbital active block
moves. Step 04 therefore builds `J[dm_core]` once and evaluates each
state's wave-function energy from it, the active-space integrals and the
state's active 1-RDM (`build_mcwfn_cache` in `mcxray/cli/step04_casci_pdft.py`). This is
an identity, not an approximation: on TiCl₃ the cached and pyscf
`energy_mcwfn` values agree to 4.5e-13 Ha and the SS-PDFT totals to 1e-09
Ha, below the run-to-run scatter. Every run recomputes one state through
pyscf's own path and aborts on a mismatch above 1e-07 Ha; `--no-jcache`
restores the per-state J build. The cache is skipped for hybrid on-top
functionals. The timings above predate it — what remains per state is the
on-top functional.

### `--nproc`: one worker per state, not one thread per integral

That measurement is exactly why step 04 takes `--nproc=N`, which evaluates
the SS-PDFT energies over a pool of **single-threaded worker processes**.
The root loop is a pure map — each state's energy is an independent
function of its own CI vector — so N workers scale where N threads in one
process do not. `--nproc=1` is the default and remains the reference path.

Implementation notes, all in `mcxray/cli/step04_casci_pdft.py`; no PySCF internals:

- **`spawn`, not `fork`.** libgomp is not fork-safe and the parent has
  already run threaded OpenMP regions in the CASCI solve by this point,
  where forking can deadlock. Spawn costs a few seconds of import per
  worker, once.
- **Threads are pinned through `os.environ` before the pool is created**,
  because spawned children inherit the environment and OpenBLAS reads its
  thread count at load time — `lib.num_threads()` alone would not catch
  it. The parent's values are restored in a `finally`.
- **`max_memory` is divided by `nproc`**, or 40 workers each believe they
  own the whole node and independently pick incore algorithms.
- **Only `(index, ci, nelec)` is pickled per task** (~31 kB); `cas_mo`
  (1.5 MB) is sent once via `initargs`.

Measured at 430 AOs: **~1.25 GB per worker**, with the parent falling to
0.18 GB once the pool starts. 40 workers is ~50 GB of a 480 GB node, so
core count, not memory, is the limit — set `nproc` = cpus-per-task.

**On reproducibility.** The parallel path differs from serial by ~1.3e-07
Ha, but that is pre-existing scatter, not something parallelism
introduced:

| comparison | max ΔE_MCPDFT |
|---|---|
| serial vs serial, repeated | 9.53e-08 Ha |
| `--nproc=4` vs `--nproc=4`, repeated | 9.06e-08 Ha |
| serial vs `--nproc=4` | 1.32e-07 Ha |

The parallel path's own run-to-run scatter is marginally *smaller* than
the serial path's. The source is the Davidson: CI vectors converge to
~1e-5, and MC-PDFT is a functional of the density rather than variational,
so it inherits that at first order — while `e_casci`, being variational,
agrees to 1.6e-11. At 3.4 µeV against a 0.4 eV core-hole linewidth it
cannot move a spectrum. On Cr(acac)₃ itself, serial and 3-worker runs give
identical energies for all eight shared core states to printed precision.

If step 04 ever needs to scale past one node, **array-job across states**
for the same reason: N tasks buy ~N×, N cores in one task buy ~nothing.

### Step 05 scales better, but still not linearly

**`run_cracac.sh`'s 80-core column is the 14-core measurement divided by
80/14 and nothing more** — an optimistic bound, not a prediction. Step
04's measured 3.1 h against that column's ~40 min shows how wrong it can
be. Two further reasons to distrust it: the 1.8e7 integrals/s figure comes
from a single slab over Cr's first two *s* shells, the cheapest quartets
in the molecule (so >1.6 h is a floor, not an estimate); and
`soc.soc_matrix` is a pure-Python double loop over 1380² basis pairs,
single-threaded on any node. Measured libcint scaling on TiCl₃'s full
`int2e_p1vxp1` is 4.15× at 14 threads and flat from 8 to 14 — it is
bandwidth-bound writing its output, not compute-bound.

The 764 GB tensor is never materialized. Only its contraction with one
density is needed, which is a J/K build, so `soc.hso_2e_somf` hands it to
pyscf's integral-direct driver (`jk.get_jk`, `aosym='a4ij'`): 4× fewer
integrals from the i↔j antisymmetry, screening of negligible quartets, and
the contraction done in C as integrals are made.

| 2e SOMF build | TiCl₃ | Cr(acac)₃ |
|---|---|---|
| batched `s1` tensor + einsum (`--somf=batched`) | 3.4 s | **53.3 min** (60-core node, job 24838352) |
| **integral-direct J/K (default)** | 2.4 s, max\|Δ\| 4.5e-13 | **5.1 min** on the 14-core laptop, 0.3 GB peak |

Both give ζ(Cr 2p, SOMF) = 5.300 eV. The batched path stays available as
the reference; its memory cap is the system file's `soc_max_bytes` (16 GB here,
overridable with `--soc-max-bytes=2g`), and one AO of the batched index
already carries 3 × nao³ × 8 B = 1.78 GB, so its batch size cannot go below
one shell.

### The first cluster run: step 04 pinned to one core

Job 24838352 took **506 min** for step 04's 500 states with 60 workers,
about 61 min per state per worker against ~65 s single-threaded. States
finished in bursts of ~60 about once an hour: the whole pool was sharing one
core. The job scripts export `OMP_PROC_BIND=close` / `OMP_PLACES=cores`
(right for step 05's threaded integrals), and libgomp then pins the
parent's main thread to the first core when pyscf loads. Spawned workers
inherit that mask, and clearing the variables for the children does not
undo it. Measured locally: 14 job CPUs, parent pinned to 2, every child on
the same 2.

Step 04 now records the job's CPU set before pyscf is imported, resets
every worker to it, drops the binding variables for the pool only, and
aborts if any worker is still confined. Its log prints
`note: this process is pinned to N of the job's M CPUs` when this happens.

Lowering the MC-PDFT grid is **not** a useful economy here: level 1 is
only 26% faster than level 3 and shifts relative energies by <0.1 meV,
while level 0 saves nothing at all and costs 2.4 meV.

### ζ is now read rotation-invariantly

Setting this system up exposed a bug in the ζ diagnostic. Within a p shell
the SOC operator is ζ·l·s, so `max|h_z|` over the core block equals ζ
*only* when the two orbitals it connects are the in-plane pair for the lab
z axis. TiCl₃ is built with z along its C₃ axis, so the shortcut was right
there (4.050 vs 4.051 eV) and the error was invisible. Cr(acac)₃ sits at an
arbitrary orientation, where the same shortcut reads **4.424 eV against a
true ζ of 6.009 eV** — a 26% underestimate of the quantity that sets the
L₃–L₂ splitting.

`soc.core_shell_zeta` now contracts the whole core block instead,
ζ = √(Σₓ Σ_pq h[x]²_pq ⁄ 6), which is invariant because the Levi-Civita
symbol is. Only the *diagnostic* was ever wrong — `h_act` itself is
frame-independent and no spectrum was affected. The corrected value is a
much better sanity check: ζ(Cr)/ζ(Ti) = 1.48 against the ~(24/22)⁴ = 1.42
expected from Z-scaling, and 1.5ζ = 9.01 eV against ~9.3 eV measured.

## Scaling, and when you need `--cvs`

With frozen MOs and 2p in the active space, plain CASCI is exact and the
core-excited states are just high-lying roots. **That is the default and it
stays the reference.** But the number of valence roots below the core
manifold grows combinatorially with the valence active space:

| valence CAS | total CAS | roots to reach core manifold |
|---|---|---|
| 3d only | (7e,8o) | 6 |
| 3d + 4d | (7e,13o) | 11 |
| 3d + 3 Cl lone pairs | (13e,11o) | **3,921** (~72 GB of Davidson vectors) |
| 3d + 6 ligands | (19e,15o) | 731,809 |

So once ligand orbitals enter the valence space, `--cvs` restricts the
determinant space to a fixed number of core holes, making the core-excited
states the *lowest* roots (3920 → 105 determinants here). Cost of the
approximation, measured where both are computable: **4–9 meV** on a ~469 eV
excitation, with 2p population exactly 5.000. In CVS mode two solves are run
(`nhole=0` and `nhole=1`), because a `nhole=1` space contains no ground
state to reference excitation energies to.

## Known simplifications (and how to extend)

- **Frozen orbitals for the core hole.** No orbital relaxation in response
  to the 2p hole. This is the standard frozen-orbital approximation and is
  the main reason absolute edge positions land ~8–10 eV above experiment
  (Ti L₃ ≈ 456–458 eV). Relative structure — multiplet splittings, branching
  ratio, RIXS energy losses — is far better than the absolute position.
- **Small demo basis** (`def2-SVP`, all-electron — no ECP, since the 2p core
  must be present). It lacks core-valence flexibility; use a larger
  core-flexible basis for production. This applies to Cr(acac)₃ too: its
  430 AOs are 430 *small* AOs, and the 764 GB two-electron SOC tensor
  grows as nao⁴, so a triple-ζ basis is a ~5× cost in step 05 alone.
- **Valence CAS is 3d-only** in both shipped examples. Ligand orbitals (and
  hence covalency and LMCT satellites) need a larger valence CAS, which is
  what `--cvs` exists for. For Cr(acac)₃ the acac π system is the obvious
  next addition and is also where the CAS(3,5) is most fragile — see the
  e_g rotation trap above.
- **RIXS elastic-line subtraction** is not implemented; the elastic channel
  is included, as in real spectra. The ingredients to fit/subtract it are
  all computed.
- **No polarization resolution.** The powder average weights all nine tensor
  components equally; a parallel/perpendicular geometry needs the three
  rotational invariants separately.
- **Active-space selection is a checked heuristic** — `find_core_orbitals`
  asserts >90% metal-2p character, and the orbital table is printed for
  inspection rather than trusted.

## Requirements

`pyscf` (2.14), `pyscf-forge` (for `pyscf.mcpdft`), `numpy`, `matplotlib`,
`jupyter`, a working `orca` + `orca_2mkl` on `PATH`, and the `mokit-bridge`
conda env described above. No GPU needed; the TiCl₃ example runs in minutes.

Memory note: the 2e SOC integral `int2e_p1vxp1` is 3·nao⁴ — 1.25 GB at
85 AOs, **764 GB at 430** — and is never stored: the default integral-direct
build needs only a few nao² matrices. The `--somf=batched` reference path
builds it in shell batches instead, capped per system (`soc_max_bytes` in
`system.toml`: 200 MB for TiCl₃, 16 GB for Cr(acac)₃, override with
`--soc-max-bytes=2g`); that cap cannot usefully go below one shell's worth,
3·nao³·8 B per AO — 1.78 GB at 430 AOs.

Cr(acac)₃ is a cluster job, not a laptop job: `cracac/run_cracac.sh`
is a SLURM script for one 80-core / 512 GB node.
