#!/bin/bash
#SBATCH --partition=exfel
#SBATCH --job-name=crddpd-456
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=450G
#SBATCH --time=96:00:00
#SBATCH --output=crddpd-456.%j.log
#SBATCH --error=crddpd-456.%j.err
#SBATCH --constraint='[9374F|75F3]'
#
# [Cr(ddpd)2]3+ L-edge XAS/RIXS -- steps 04-06 only. Pure PySCF: no ORCA,
# no MOKIT. Reads nothing but run/cr/crddpd/active_space.pkl, which
# submit_23.sh writes. Same structure as run/cr/cracac/submit_456.sh;
# the reasoning behind the threading and --nproc choices is spelled out
# there and in the README and is not repeated here.
#
# ---------------------------------------------------------------------
# Resources -- NOTHING below is measured for this system yet.
# ---------------------------------------------------------------------
# The state count is identical to Cr(acac)3 (50 valence + 450 core), but
# the AO basis is 1613 functions (def2-TZVP, 79 atoms) against 430. What
# that changes:
#
#   04  the CI solves are unchanged (same 50 / 450 determinants). The
#       SS-PDFT energies are not: each state is a direct Coulomb build
#       plus the on-top functional on a 79-atom grid, both in 1613 AOs.
#       Cr(acac)3 measured ~65 s/state single-threaded; expect tens of
#       minutes per state per worker here. The log prints s/state as the
#       pool drains -- read the real number off the first batch.
#   05  the integral-direct 2e SOMF build took 5.1 min at 430 AOs on 14
#       cores and scales between nao^2 and nao^4 depending on screening:
#       hours, not minutes. The SOC matrix assembly afterwards depends
#       only on the CI space and costs what it did for Cr(acac)3.
#       --somf=batched is NOT usable: one AO slab is 101 GB.
#   06  CI-space only; same as Cr(acac)3.
#
# --mem=450G fits both node types in the constraint (the 75F3 nodes have
# 462 GB). --time is deliberately loose for the reasons above; tighten
# both after the first run.
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

# Right for step 05's threaded integrals. Step 04 undoes the pinning for
# its worker pool itself (see "The first cluster run" in the README).
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# In MB. Step 04 divides this by --nproc for its workers.
export PYSCF_MAX_MEMORY=${PYSCF_MAX_MEMORY:-400000}

# ---------------------------------------------------------------------
# 3. Preflight. Fail in seconds, not hours in.
# ---------------------------------------------------------------------
RUN="$REPO/run/cr/crddpd"
cd "$REPO" || { echo "REPO=$REPO is not a directory" >&2; exit 1; }

[ -f "$RUN/active_space.pkl" ] || {
    echo "ERROR: $RUN/active_space.pkl is missing." >&2
    echo "Run steps 02-03 first:  sbatch $RUN/submit_23.sh" >&2
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
SYS=--system=crddpd

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

# --cvs is REQUIRED and the root counts (450 core + 50 valence, from
# scripts/systems.py) are the complete d^3 manifolds, exactly as for
# Cr(acac)3 -- do not lower them. --nproc: one single-threaded worker per
# state.
step "04 CASCI + SS-PDFT" "$RUN/casci_pdft.npz" \
     "$REPO/scripts/04_casci_pdft.py" $SYS --cvs --nproc="$NTHREAD"

step "05 spin-orbit coupling" "$RUN/soc_results.npz" \
     "$REPO/scripts/05_soc.py" $SYS

step "06 XAS + RIXS" "$RUN/spectra.npz" \
     "$REPO/scripts/06_xas_rixs.py" $SYS

echo "=== all done $(date '+%F %T') ==="
ls -la "$RUN"/{casci_pdft,soc_results,spectra}.npz \
       "$RUN"/{xas_spectrum,rixs_map}.png 2>/dev/null || true
