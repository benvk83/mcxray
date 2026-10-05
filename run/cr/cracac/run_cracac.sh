#!/bin/bash
#SBATCH --job-name=cracac-ledge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=cracac-ledge.%j.log
#
# Cr(acac)3 L-edge XAS/RIXS -- full production run.
#
# Sized for one 80-core / 512 GB node. Everything below is shared-memory
# (OpenMP + BLAS); there is no MPI in this workflow, so --ntasks=1.
#
# ---------------------------------------------------------------------
# What this costs. The MEASURED column is real, on a 14-core box. The
# 80-core column is those numbers divided by 80/14 and nothing else --
# no scaling curve was ever measured. Treat it as an optimistic bound.
# ---------------------------------------------------------------------
#                                   measured@14    linear@80    scales?
#   01 ORCA CASCI reference           ~2 min        ~2 min      no (serial)
#   02 MOKIT import + gate-2 CASCI    ~4 min        ~1 min      partly
#   03 splice Cr 2p into the CAS      ~1 min        ~1 min      n/a
#   04 CASCI + SS-PDFT                ~4 h         ~40 min      yes-ish
#   05 SOC                            >1.6 h       >20 min      mixed
#   06 XAS + RIXS                     ~15 min      ~15 min      no (Python)
#                                     -------      --------
#                                      ~6 h         ~1.5 h
#
# The 24 h walltime above is deliberately generous because of the caveats
# below; shrink it once you have one real run to calibrate against.
#
# Step 04 is the wall-clock cost: 500 states x 28.3 s/state, measured on
# real CI vectors at 430 AOs / 43 atoms. Lowering the grid does NOT help
# (level 1 is only 26% faster than level 3 and shifts relative energies by
# <0.1 meV; level 0 saves nothing and costs 2.4 meV), so the default
# stays. The per-state cost was never measured at any other core count.
#
# Step 05's two-electron SOC integral is 3 x 430^4 x 8 B = 764 GB, never
# materialized -- soc.hso_2e_somf batches over the leading AO index at
# systems.py's `soc_max_bytes` (16 GB here).
#
#   CAVEAT 1 on the 1.8e7 integrals/s that >1.6 h comes from: it is a
#   single measurement on shls_slice=(0,2,...), i.e. Cr's first two S
#   shells -- the CHEAPEST quartets in the molecule. Two attempts to time
#   other slabs died on allocation before reporting. The d/f shells on Cr
#   and the p/d shells on C/O are slower per integral, so 1.6 h is a floor.
#
#   CAVEAT 2, and the reason the linear@80 column is optimistic: libcint
#   does NOT scale linearly. Measured on TiCl3's full int2e_p1vxp1
#   (1.25 GB of output), threads -> seconds:
#       1: 11.82   2: 9.10   4: 5.65   8: 3.24   14: 2.85
#   i.e. 4.15x at 14 threads and essentially FLAT from 8 to 14. It is
#   bandwidth-bound writing the output array, not compute-bound. A
#   2-socket node has more memory channels and should do better in
#   absolute terms, but do not expect 80/14 = 5.7x.
#
#   The contractions are fine: np.einsum is single-threaded, but at
#   ~1.6e9 elements/s the three passes over 1e11 elements are ~3 min
#   against >1.6 h of integral generation. Not worth parallelizing.
#
# Step 05 also caches one spin-flip basis per (state, Ms) pair:
# 1380 x 64 x 3920 x 8 B = 2.8 GB. And soc.soc_matrix is a pure-Python
# double loop over 1380^2 pairs calling trans_rdm1s -- single-threaded on
# any node, estimated ~10 min, never measured at this size.
# ---------------------------------------------------------------------

set -euo pipefail

# PySCF has no MPI (mpi4pyscf is a separate package and is not used), so
# this is shared-memory only -- hence --ntasks=1 above. But OMP_NUM_THREADS
# alone is NOT enough, because two independent threading layers are in play:
#
#   OMP_NUM_THREADS       libcint + PySCF's own C kernels (integrals, FCI
#                         contractions, DFT grid loops)
#   OPENBLAS/MKL_NUM_...  numpy/scipy linear algebra. This is NOT OpenMP.
#                         Measured here: OPENBLAS_NUM_THREADS takes
#                         PRECEDENCE over OMP_NUM_THREADS -- setting
#                         OMP=14 with OPENBLAS=1 gives 62 GFLOP/s on a
#                         3000^3 dgemm against 180 when both are 14.
#
# Set all three. Export them AFTER any `module load`, since cluster module
# files routinely pin OMP_NUM_THREADS=1.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-80}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS

# Bandwidth-bound work on a multi-socket node; keep threads near their
# memory. Drop these if the site's MPI/OpenMP wrappers set affinity.
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# In MB. Without it PySCF assumes ~4 GB and picks outcore algorithms it
# does not need to on a 512 GB node.
export PYSCF_MAX_MEMORY=400000

# sbatch runs a COPY of this script, so $0 is not reliable under SLURM.
# Set REPO explicitly, or export it before submitting.
REPO=${REPO:-$HOME/calc/pyscf}
cd "$REPO" || { echo "set REPO to the repository root" >&2; exit 1; }
SYS=--system=cracac

# Step 01 only regenerates the gate-2 reference (a CASCI in the supplied
# SA-CASSCF orbitals). It is skipped automatically if the .out is present;
# pass --force to redo it.
python scripts/01_orca_casscf.py $SYS
python scripts/02_import_orbitals.py $SYS
python scripts/03_build_active_space.py $SYS

# --cvs is REQUIRED here, not optional. Without it the core-excited states
# are roots 51 and up of a 3920-determinant problem and the Davidson has to
# converge the whole valence manifold first. With it, nhole=0 and nhole=1
# are solved separately in 50- and 450-determinant spaces.
#
# nroots/nvalence default to the complete manifolds from systems.py
# (450 core, 50 valence) -- these are counted from the determinant space,
# not truncations, so do not lower them without reading the MS-PDFT
# warning in the README about truncating a model space.
# --nproc spreads the 500 SS-PDFT energies over single-threaded worker
# processes. MC-PDFT barely threads (measured 36.4 / 31.6 / 28.7 s per
# state at 4 / 8 / 14 threads -> ~89% serial), so one worker per state
# scales where more threads per state does not. ~1.25 GB per worker.
python scripts/04_casci_pdft.py $SYS --cvs --nproc="$OMP_NUM_THREADS"

python scripts/05_soc.py $SYS
python scripts/06_xas_rixs.py $SYS

echo "done: see run/cr/cracac/{xas_spectrum.png,rixs_map.png,spectra.npz}"
