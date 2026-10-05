#!/bin/bash
#SBATCH --partition=exfel
#SBATCH --job-name=cracac-456
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=cracac-456.%j.log
#SBATCH --error=cracac-456.%j.err
#SBATCH --constraint='[9374F|75F3]'
#
# Cr(acac)3 L-edge XAS/RIXS -- steps 04-06 only.
#
# NO ORCA AND NO MOKIT ARE NEEDED FOR THIS JOB. That is the whole reason
# it starts at step 04. The external-code dependencies are confined to the
# two steps that come before it:
#
#     01   needs ORCA           (runs the CASCI reference)
#     02   needs ORCA + MOKIT   (.gbw -> .mkl -> .fch -> mo_coeff)
#     03   pure PySCF
#     04-06  pure PySCF         <-- this job
#
# Verified, not assumed: none of scripts/04,05,06 import orbital_io (where
# the MOKIT bridge lives) and none of them shell out to conda, orca_2mkl,
# mkl2fch or fch2py.
#
# So run 01-03 on the workstation that has MOKIT, then stage two things:
#
#     scripts/                             the whole directory
#     run/cr/cracac/active_space.pkl       1.5 MB, written by step 03
#
# Nothing else from run/cr/cracac is read -- not the .gbw, not the .xyz,
# not the ORCA output. Geometry, basis, charge, spin and the reordered MO
# coefficients are all inside active_space.pkl (it is 1.5 MB because
# cas_mo is 430 x 430 float64 = 1.48 MB, and that is essentially all of it).
#
# The requirements on the cluster are therefore just: pyscf, pyscf-forge,
# numpy, and matplotlib if you want step 06's .png files.
#
# ---------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------
# --cpus-per-task=40 rather than 80 on purpose. Measured on a 14-core box,
# libcint's int2e_p1vxp1 scales 1 -> 14 threads as
#     1: 11.82 s   2: 9.10   4: 5.65   8: 3.24   14: 2.85
# i.e. 4.15x at 14 threads and FLAT from 8 to 14 -- it is bandwidth-bound
# writing its output array, not compute-bound. Steps 05 and 06 also contain
# single-threaded Python sections that no core count helps. 80 cores may
# well not beat 40, and 40 usually queues sooner. Raise it once a real run
# says it is worth it.
#
# --time=24:00:00 is deliberately generous. Honest expectation, from
# 14-core measurements, is ~6 h there and somewhere between 1.5 h and 6 h
# on a big node; the spread is why the walltime is not tight. Shrink it
# after the first calibrated run.
# ---------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------
# 1. Environment. EDIT THIS BLOCK for your site.
# ---------------------------------------------------------------------
export LD_PRELOAD=""
source /etc/profile.d/modules.sh

module load maxwell
module load python/3.13
source /gpfs/exfel/data/user/kuikenb/calc/pyscf_env_py313/bin/activate

# Requirements: pyscf (2.14 tested), pyscf-forge (for pyscf.mcpdft), numpy,
# and matplotlib if you want step 06's .png files (it skips them silently
# without it). Again: no ORCA, no MOKIT, no conda bridge env.

REPO=${REPO:-/gpfs/exfel/data/user/kuikenb/calc/pyscf_casci/run3/pyscf}

# ---------------------------------------------------------------------
# 2. Threading. Export AFTER any `module load` -- cluster module files
#    routinely pin OMP_NUM_THREADS=1 and would silently undo this.
# ---------------------------------------------------------------------
# PySCF core has no MPI (mpi4pyscf is a separate package, unused here), so
# this is a single shared-memory task. But OMP_NUM_THREADS alone is NOT
# enough: two independent threading layers are in play.
#
#   OMP_NUM_THREADS        libcint + PySCF's own C kernels -- integrals,
#                          FCI contractions, DFT grid loops. This is where
#                          almost all of steps 04 and 05 live.
#   OPENBLAS/MKL_NUM_...   numpy/scipy linear algebra, which is NOT OpenMP.
#                          Measured: OPENBLAS_NUM_THREADS takes PRECEDENCE
#                          over OMP_NUM_THREADS. Setting OMP=14 with
#                          OPENBLAS=1 gave 62 GFLOP/s on a 3000^3 dgemm
#                          against 180 with both at 14.
#
# Set all three and let whichever backend is present pick its own up.
NTHREAD=${SLURM_CPUS_PER_TASK:-40}
export OMP_NUM_THREADS=$NTHREAD
export MKL_NUM_THREADS=$NTHREAD
export OPENBLAS_NUM_THREADS=$NTHREAD

# Bandwidth-bound work on a multi-socket node; keep threads near their
# memory. Drop these if your site's wrappers already set affinity.
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# In MB. Without it PySCF assumes ~4 GB and will choose outcore algorithms
# it has no need for on a 512 GB node.
export PYSCF_MAX_MEMORY=${PYSCF_MAX_MEMORY:-400000}

# ---------------------------------------------------------------------
# 3. Preflight. Fail in seconds, not four hours in.
# ---------------------------------------------------------------------
RUN="$REPO/run/cr/cracac"
cd "$REPO" || { echo "REPO=$REPO is not a directory" >&2; exit 1; }

[ -f "$RUN/active_space.pkl" ] || {
    echo "ERROR: $RUN/active_space.pkl is missing." >&2
    echo "Run steps 01-03 first:" >&2
    echo "    python scripts/01_orca_casscf.py     --system=cracac" >&2
    echo "    python scripts/02_import_orbitals.py --system=cracac" >&2
    echo "    python scripts/03_build_active_space.py --system=cracac" >&2
    exit 1
}

echo "=== preflight ==="
echo "host        : $(hostname)"
echo "repo        : $REPO"
echo "threads     : OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS"
# This exits non-zero, and so aborts the job, if anything is wrong. Better
# to lose 20 seconds here than to discover it after four hours of SS-PDFT.
python - "$NTHREAD" <<'PY'
import sys, time, numpy
from pyscf import lib, __version__ as v
want = int(sys.argv[1])
print(f'python      : {sys.version.split()[0]}')
print(f'numpy       : {numpy.__version__}')
print(f'pyscf       : {v}')
print(f'PYSCF_MAX_MEMORY: {lib.param.MAX_MEMORY} MB')
try:
    import pyscf.mcpdft                      # noqa: F401
    print('pyscf-forge : present (pyscf.mcpdft importable)')
except ImportError:
    sys.exit('ERROR: pyscf-forge missing -- step 04 needs pyscf.mcpdft')

# PySCF's OpenMP layer. If this reports 1 while more were requested, the
# variable was set after PySCF loaded, or a module file overrode it.
got = lib.num_threads()
print(f'OpenMP      : lib.num_threads() = {got}, requested {want}')
if want > 1 and got == 1:
    sys.exit('ERROR: PySCF sees 1 thread. Export OMP_NUM_THREADS after '
             '`module load`, not before.')

# The BLAS layer is separate and is controlled by OPENBLAS/MKL_NUM_THREADS,
# NOT by OMP_NUM_THREADS. A low number here means linear algebra is running
# single-threaded even though libcint is not.
a = numpy.random.rand(2000, 2000); b = a.copy(); a @ b
t = time.time(); a @ b; gflops = 2 * 2000 ** 3 / (time.time() - t) / 1e9
verdict = 'threaded' if gflops > 100 else 'LOOKS SINGLE-THREADED'
print(f'BLAS        : {gflops:.0f} GFLOP/s on a 2000^3 dgemm -- {verdict}')
if want > 1 and gflops < 100:
    print('WARNING: check OPENBLAS_NUM_THREADS / MKL_NUM_THREADS. '
          'Continuing, since only parts of steps 05-06 depend on BLAS.',
          file=sys.stderr)
PY
echo

# ---------------------------------------------------------------------
# 4. The steps. Each is skipped if its output is already there, so a job
#    that dies in step 05 does not repeat step 04's several hours.
#    Re-run everything with:  FORCE=1 sbatch submit_456.sh
# ---------------------------------------------------------------------
FORCE=${FORCE:-0}
SYS=--system=cracac

step () {                      # step <label> <output-file> <script> [args...]
    local label=$1 out=$2; shift 2
    if [ "$FORCE" != "1" ] && [ -f "$out" ]; then
        echo "=== $label: SKIP, $(basename "$out") exists (FORCE=1 to redo) ==="
        return
    fi
    echo "=== $label: start $(date '+%F %T') ==="
    local t0=$SECONDS
    python -u "$@"
    echo "=== $label: done in $(( (SECONDS-t0)/60 )) min ==="
    echo
}

# --cvs is REQUIRED, not an optimisation. Without it the core-excited
# states are roots 51+ of a 3920-determinant problem and the Davidson must
# converge the whole valence manifold first. With it, nhole=0 and nhole=1
# are solved separately in 50- and 450-determinant spaces.
#
# Root counts come from scripts/systems.py: 450 core + 50 valence. Both
# EQUAL the exact determinant-space size, so neither is a truncation.
# Do not lower them to save time -- Hund's rule puts the sextets lowest in
# 2p^5 3d^4, so a truncated run returns nothing but sextets, every one of
# them dark against the quartet ground state, and you get an empty
# spectrum with no error anywhere.
# --nproc runs the 500 SS-PDFT energies over a process pool, one
# single-threaded worker per state, instead of one many-threaded process
# per state in sequence. This is the single biggest lever in the workflow,
# because MC-PDFT barely threads:
#
#     threads    4       8      14        (measured, energy_tot, one state)
#     s/state   36.4    31.6    28.7
#
# An Amdahl fit gives serial 25.6 s + parallel 43.1 s, i.e. a floor near
# 25.6 s/state however many cores one process gets. A 40-core node measured
# 22.4 s/state serially -- 500 states = 3.1 h. Spread over N single-threaded
# workers the same work scales with N instead.
#
# Memory is not the limit: measured ~1.25 GB per worker at 430 AOs (the
# parent falls to 0.18 GB once the pool starts), so 40 workers is ~50 GB of
# the 480 GB requested. Core count is the limit, so nproc = cpus-per-task.
step "04 CASCI + SS-PDFT" "$RUN/casci_pdft.npz" \
     "$REPO/scripts/04_casci_pdft.py" $SYS --cvs --nproc="$NTHREAD"

step "05 spin-orbit coupling" "$RUN/soc_results.npz" \
     "$REPO/scripts/05_soc.py" $SYS

step "06 XAS + RIXS" "$RUN/spectra.npz" \
     "$REPO/scripts/06_xas_rixs.py" $SYS

echo "=== all done $(date '+%F %T') ==="
ls -la "$RUN"/{casci_pdft,soc_results,spectra}.npz \
       "$RUN"/{xas_spectrum,rixs_map}.png 2>/dev/null || true
