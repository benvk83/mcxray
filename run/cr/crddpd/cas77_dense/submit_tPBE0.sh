#!/bin/bash
#SBATCH --partition=exfel
#SBATCH --job-name=crddpd77d-tPBE0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=crddpd77d-tPBE0.%j.log
#SBATCH --error=crddpd77d-tPBE0.%j.err
#
# [Cr(ddpd)2]3+ CAS(7,7), dense CASCI -- tPBE0 energies on the states of
# submit_456.sh. Run after that job:
#
#   sbatch --dependency=afterok:<jobid of submit_456.sh> submit_tPBE0.sh
#
# tPBE0 = 0.25 E_CASCI + 0.75 E_tPBE per state, so nothing is recomputed:
# casci_pdft_tPBE0.npz is casci_pdft.npz with e_pdft replaced, step 05b
# swaps the diagonal of the stored SOC matrix, and steps 06-07 take
# --tag=tPBE0. Same outputs as run/cr/crddpd/cas77/*_tPBE0*.

set -euo pipefail

export LD_PRELOAD=""
source /etc/profile.d/modules.sh

module load maxwell
module load python/3.13
source /gpfs/exfel/data/user/kuikenb/calc/pyscf_env_py313/bin/activate

REPO=${REPO:-/gpfs/exfel/data/user/kuikenb/calc/pyscf_casci/run3/pyscf}

NTHREAD=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=$NTHREAD
export MKL_NUM_THREADS=$NTHREAD
export OPENBLAS_NUM_THREADS=$NTHREAD
# No OMP_PLACES / OMP_PROC_BIND (see submit_456.sh).
unset OMP_PLACES OMP_PROC_BIND

RUN="$REPO/run/cr/crddpd/cas77_dense"
SYS=--system=crddpd-cas77-dense
EXP="$REPO/run/cr/crddpd/Crddpd2_0T_xas.txt"
cd "$REPO" || { echo "REPO=$REPO is not a directory" >&2; exit 1; }

for f in casci_pdft soc_results spectra; do
    [ -f "$RUN/$f.npz" ] || {
        echo "ERROR: $RUN/$f.npz is missing -- submit_456.sh has not finished." >&2
        exit 1
    }
done

echo "=== tPBE0 energies: start $(date '+%F %T') ==="
python -u - "$RUN" <<'PY'
import os, sys
import numpy as np
run = sys.argv[1]
d = dict(np.load(os.path.join(run, 'casci_pdft.npz'), allow_pickle=True))
d['e_pdft'] = 0.25 * d['e_casci'] + 0.75 * d['e_pdft']
out = os.path.join(run, 'casci_pdft_tPBE0.npz')
np.savez(out, **d)
print(f'Saved {len(d["e_pdft"])} states to {out}')
PY

python -u "$REPO/scripts/05b_soc_swap_energies.py" $SYS --tag=tPBE0
python -u "$REPO/scripts/06_xas_rixs.py" $SYS --tag=tPBE0
python -u "$REPO/scripts/06_xas_rixs.py" $SYS --tag=tPBE0 --pol=pi --theta=90
python -u "$REPO/scripts/07_rixs_by_initial_doublet.py" $SYS --tag=tPBE0 \
       --pol=pi --theta=90
python -u "$REPO/scripts/plot_xas_vs_experiment.py" $SYS --exp="$EXP"
python -u "$REPO/scripts/plot_xas_vs_experiment.py" $SYS --exp="$EXP" --gamma=0.2

echo "=== all done $(date '+%F %T') ==="
ls -la "$RUN"/*tPBE0* "$RUN"/xas_vs_experiment*.png 2>/dev/null || true
