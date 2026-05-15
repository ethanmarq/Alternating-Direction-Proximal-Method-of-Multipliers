#!/bin/bash
# ============================================================================
# submit_spca.sh — SLURM batch script for sPCA solver comparison
#
# Each array task does:
#   1. find_stepsize on this one (dataset, p, mu) config -> writes/updates
#      best_rho.mat with the entry for this config
#   2. run_spca_experiment using that best_rho.mat
#
# Usage:
#     sbatch submit_spca.sh                          # uses DATASETS[0]
#     sbatch --export=DATASET=gisette submit_spca.sh # override which one
#     sbatch --array=0-4 submit_spca.sh              # all datasets in parallel
# ============================================================================
#SBATCH --job-name=spca
#SBATCH --output=logs/spca_%A_%a.out
#SBATCH --error=logs/spca_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
mkdir -p logs results

# ----------------------------- USER CONFIG ----------------------------------
PROJECT_DIR="${PROJECT_DIR:-$HOME/repos/Alternating-Direction-Proximal-Method-of-Multipliers}"
DATA_DIR="${DATA_DIR:-/scratch/marque6/libsvm_data}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/results}"

# One row per array index: name  mat_file  p (#components)
DATASETS=(
  "gisette        gisette.mat         50"
  "leu            leu.mat             20"
  "rcv1_train     rcv1_train.mat      50"
  "usps           usps.mat            10"
  "news20         news20.mat          50"
)

# Defaults shared across datasets
MU=0.01
N_ITER=20
AVG=5
SWEEP_N=20
SWEEP_AVG=1

# Per-task best_rho.mat. Using one per task means parallel array jobs can't
# clobber each other's writes; main run_spca_experiment will read this exact
# file via opts.best_rho_file.
BEST_RHO_DIR="${BEST_RHO_DIR:-$PROJECT_DIR/best_rho}"
mkdir -p "$BEST_RHO_DIR"

# --------------------------- pick dataset -----------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -ge "${#DATASETS[@]}" ]; then
    echo "Array index $TASK_ID out of range; have ${#DATASETS[@]} datasets."
    exit 1
fi
read -r NAME MAT_FILE P_COMP <<< "${DATASETS[$TASK_ID]}"

# Allow override via --export=DATASET=gisette
if [ -n "${DATASET:-}" ]; then
    NAME=$DATASET
    MAT_FILE=${DATASET}.mat
fi

DATA_PATH="$DATA_DIR/$MAT_FILE"
RUN_NAME="${NAME}_p${P_COMP}_mu${MU}"
BEST_RHO_FILE="$BEST_RHO_DIR/${RUN_NAME}_best_rho.mat"

echo "============================================================"
echo "SLURM job ${SLURM_JOB_ID:-local} array ${TASK_ID}"
echo "host         : $(hostname)"
echo "started      : $(date)"
echo "dataset      : $NAME"
echo "data file    : $DATA_PATH"
echo "components   : $P_COMP"
echo "mu           : $MU"
echo "iterations   : sweep N=$SWEEP_N (avg=$SWEEP_AVG); main N=$N_ITER (avg=$AVG)"
echo "best_rho     : $BEST_RHO_FILE"
echo "results dir  : $RESULTS_DIR"
echo "============================================================"

# --------------------------- environment ------------------------------------
if command -v module >/dev/null 2>&1; then
    module load matlab 2>/dev/null || echo "(no 'matlab' module — assuming matlab is on PATH)"
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

cd "$PROJECT_DIR"

# --------------------------- step 1: find_stepsize --------------------------
echo ""
echo "--- Step 1: find_stepsize ---"

# Build the configs struct for this single (dataset, p, mu) combo, and the
# opts struct controlling the sweep. One MATLAB invocation does the sweep.
SWEEP_CFG="struct('data_path','$DATA_PATH','name','$NAME','p',$P_COMP,'mu',$MU)"
SWEEP_OPTS="struct('N',$SWEEP_N,'avg',$SWEEP_AVG,'output_file','$BEST_RHO_FILE')"

matlab -nodisplay -nosplash -nodesktop \
    -batch "maxNumCompThreads($OMP_NUM_THREADS); \
            addpath('$PROJECT_DIR'); \
            addpath('$PROJECT_DIR/misc'); \
            addpath('$PROJECT_DIR/SSN_subproblem'); \
            configs = $SWEEP_CFG; \
            find_stepsize(configs, $SWEEP_OPTS);"

# Sanity: the file should now exist
if [ ! -f "$BEST_RHO_FILE" ]; then
    echo "ERROR: find_stepsize did not produce $BEST_RHO_FILE" >&2
    exit 2
fi

# --------------------------- step 2: run_spca_experiment --------------------
echo ""
echo "--- Step 2: run_spca_experiment ---"

OPTS="struct('data_path','$DATA_PATH'"
OPTS+=",'p',$P_COMP,'mu',$MU,'N',$N_ITER,'avg',$AVG"
OPTS+=",'output_dir','$RESULTS_DIR','name','$NAME'"
OPTS+=",'best_rho_file','$BEST_RHO_FILE'"
OPTS+=",'algorithms',{{'all'}}"
OPTS+=",'show_plots',false,'save_plots',true)"

matlab -nodisplay -nosplash -nodesktop \
    -batch "maxNumCompThreads($OMP_NUM_THREADS); \
            addpath('$PROJECT_DIR'); \
            addpath('$PROJECT_DIR/misc'); \
            addpath('$PROJECT_DIR/SSN_subproblem'); \
            run_spca_experiment($OPTS);"

echo "============================================================"
echo "finished     : $(date)"
echo "============================================================"
