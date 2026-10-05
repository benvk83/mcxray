#!/bin/bash
#SBATCH --partition=exfel
#SBATCH --job-name=crddpd77d-456
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=96:00:00
#SBATCH --output=crddpd77d-456.%j.log
#SBATCH --error=crddpd77d-456.%j.err
#SBATCH --constraint='[9374F|75F3]'
#
# [Cr(ddpd)2]3+ from CASSCF(7,7) orbitals, L-edge XAS/RIXS -- steps 04-06
# with the DENSE CASCI and a windowed core manifold. Same active space
# and orbitals as run/cr/crddpd/cas77 (active_space.pkl here is a symlink
# to that one); results go to run/cr/crddpd/cas77_dense.
#
# Why: the 450-root Davidson of job 25137026 left core roots 302-449
# unconverged, so the core manifold stopped 7.6 eV above its onset.
# --dense diagonalizes the full 1225 / 5880 determinant CVS spaces and
# --window=$WINDOW keeps the core roots within that many eV (CASCI) of
# the lowest one. The step 04 log prints roots and SOC dimension per
# window before the cut is applied; casci_dense.npz holds the complete
# solution, so another window needs no new CASCI:
#
#   WINDOW=16 FORCE=1 sbatch submit_456.sh
#
# ---------------------------------------------------------------------
# Resources -- measured on job 25137026 (64 cores, 356 states), the
# rest is extrapolated and NOT measured.
# ---------------------------------------------------------------------
#   04  SS-PDFT: ~8 s wall per state on 64 workers, after 9 min for the
#       core Coulomb cache. The dense CASCI replaces two Davidson solves
#       that took ~155 min there, most of it presumably the active-space
#       integrals, which are now built once.
#   05  32 min for the 2e SOMF build, independent of the state count.
#       The SOC matrix is a Python loop over pairs of spin-orbit states:
#       9 min at dimension 1226, so ~dim^2 -- about 1.5 h at 4000 and
#       8 h at 9000.
#   06  CI-space only.
#
# --mem=450G fits both node types in the constraint (the 75F3 nodes have
# 462 GB). --time is left loose.
# ---------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------
export LD_PRELOAD=""
source /etc/profile.d/modules.sh

module load maxwell
module load python/3.13
source /gpfs/exfel/data/user/kuikenb/calc/pyscf_env_py313/bin/activate

REPO=${REPO:-/gpfs/exfel/data/user/kuikenb/calc/pyscf_casci/run3/pyscf}

# ---------------------------------------------------------------------
# 2. Threading. Export AFTER any `module load`.
# ---------------------------------------------------------------------
NTHREAD=${SLURM_CPUS_PER_TASK:-64}
export OMP_NUM_THREADS=$NTHREAD
export MKL_NUM_THREADS=$NTHREAD
export OPENBLAS_NUM_THREADS=$NTHREAD

# No OMP_PLACES / OMP_PROC_BIND. This venv loads several copies of libgomp
# (pyscf's, pytblis's, the system one); with binding on, the main thread is
# pinned to core 0 and 64 of the process's threads end up bound to that one
# core. Measured on job 25136929: affinity 0,64 of 128 CPUs, 199% CPU, the
# valence CASCI still running after 68 min (15 min unbound in step 02).
unset OMP_PLACES OMP_PROC_BIND

# In MB. Step 04 divides this by --nproc for its workers.
export PYSCF_MAX_MEMORY=${PYSCF_MAX_MEMORY:-400000}

# ---------------------------------------------------------------------
# 3. Preflight. Fail in seconds, not hours in.
# ---------------------------------------------------------------------
RUN="$REPO/run/cr/crddpd/cas77_dense"
cd "$REPO" || { echo "REPO=$REPO is not a directory" >&2; exit 1; }

[ -f "$RUN/active_space.pkl" ] || {
    echo "ERROR: $RUN/active_space.pkl is missing." >&2
    echo "Run steps 02-03 first:  sbatch $REPO/run/cr/crddpd/cas77/submit_23.sh" >&2
    exit 1
}

echo "=== preflight ==="
echo "host        : $(hostname)"
echo "repo        : $REPO"
echo "threads     : OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS"
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

got = lib.num_threads()
print(f'OpenMP      : lib.num_threads() = {got}, requested {want}')
if want > 1 and got == 1:
    sys.exit('ERROR: PySCF sees 1 thread. Export OMP_NUM_THREADS after '
             '`module load`, not before.')

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
#    that dies in step 05 does not repeat step 04.
#    Re-run everything with:  FORCE=1 sbatch submit_456.sh
# ---------------------------------------------------------------------
FORCE=${FORCE:-0}
SYS=--system=crddpd-cas77-dense
# Core-excited roots within this many eV (CASCI) of the lowest one.
WINDOW=${WINDOW:-12}

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

# --dense implies --cvs. 50 valence roots as in run/cr/crddpd/cas77
# (--nvalence= to change). --nproc: one single-threaded worker per state.
step "04 dense CASCI + SS-PDFT" "$RUN/casci_pdft.npz" \
     "$REPO/scripts/04_casci_pdft.py" $SYS --dense --window="$WINDOW" \
     --nproc="$NTHREAD"

step "05 spin-orbit coupling" "$RUN/soc_results.npz" \
     "$REPO/scripts/05_soc.py" $SYS

step "06 XAS + RIXS" "$RUN/spectra.npz" \
     "$REPO/scripts/06_xas_rixs.py" $SYS

echo "=== all done $(date '+%F %T') ==="
ls -la "$RUN"/{casci_pdft,soc_results,spectra}.npz \
       "$RUN"/{xas_spectrum,rixs_map}.png 2>/dev/null || true
