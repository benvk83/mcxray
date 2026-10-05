#!/bin/bash
#SBATCH --partition=exfel
#SBATCH --job-name=crddpd77-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=crddpd77-23.%j.log
#SBATCH --error=crddpd77-23.%j.err
#SBATCH --constraint='[9374F|75F3]'
#
# [Cr(ddpd)2]3+ from CASSCF(7,7) orbitals -- steps 02-03: import and active space.
#
# There is no step 01 for this system. crddpd-cas77-1.gbw already IS the
# SA-CASSCF(7,7) wavefunction (five Cr 3d + two ligand orbitals); all
# that is left is to bring it into PySCF and rotate the Cr 2p into the
# active space.
#
#     02   .gbw -> .mkl -> .fch -> mo_coeff via MOKIT (needs orca_2mkl and
#          the mokit-bridge conda env), orthonormality gate, and a valence
#          CASCI in the imported orbitals whose d-d spectrum is printed
#     03   splice Cr 2p into the active space -> active_space.pkl
#
# Kept apart from submit_456.sh because its log is a checkpoint: look at
# the ligand-field energies and the active-space table before spending
# hours on steps 04-06.

set -euo pipefail

export LD_PRELOAD=""
source /etc/profile.d/modules.sh

module load maxwell
module load python/3.13

# `conda` is only needed for `conda run -n mokit-bridge` (the MOKIT env,
# see the README). Appended, so the venv's python stays first on PATH.
export PATH=$PATH:/software/mamba/2026.05/bin
source /gpfs/exfel/data/user/kuikenb/calc/pyscf_env_py313/bin/activate

# Only orca_2mkl is used, from the ORCA build that wrote the .gbw.
ORCAPATH=/software/orca/6.1.1_avx2
export LD_LIBRARY_PATH=$ORCAPATH:${LD_LIBRARY_PATH:-}
export ORCA_2MKL=$ORCAPATH/orca_2mkl
[ -x "$ORCA_2MKL" ] || { echo "no orca_2mkl at $ORCAPATH" >&2; exit 1; }

REPO=${REPO:-/gpfs/exfel/data/user/kuikenb/calc/pyscf_casci/run3/pyscf}
RUN="$REPO/run/cr/crddpd/cas77"
cd "$REPO" || { echo "REPO=$REPO is not a directory" >&2; exit 1; }

# Export AFTER the module loads; see run/cr/cracac/submit_456.sh for why
# all three are set.
NTHREAD=${SLURM_CPUS_PER_TASK:-32}
export OMP_NUM_THREADS=$NTHREAD
export MKL_NUM_THREADS=$NTHREAD
export OPENBLAS_NUM_THREADS=$NTHREAD
export PYSCF_MAX_MEMORY=${PYSCF_MAX_MEMORY:-150000}

echo "=== preflight ==="
echo "host      : $(hostname)"
echo "orca_2mkl : $ORCA_2MKL"
echo "python    : $(which python)"
conda run -n mokit-bridge python -c \
    "from mokit.lib.fch2py import fch2py; print('mokit     : bridge env ok')"
echo

# Each step is skipped if its output exists (FORCE=1 to redo).
FORCE=${FORCE:-0}
SYS=--system=crddpd-cas77

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

step "02 MOKIT import + gates" "$RUN/orbitals.pkl" \
     "$REPO/scripts/02_import_orbitals.py" $SYS

step "03 build active space" "$RUN/active_space.pkl" \
     "$REPO/scripts/03_build_active_space.py" $SYS

echo "=== all done $(date '+%F %T') ==="
echo "Check the ligand-field energies and the active-space table above,"
echo "then:  sbatch $RUN/submit_456.sh"
