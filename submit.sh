#!/usr/bin/env bash
#SBATCH --job-name=spca_mu
#SBATCH --output=logs/spca_mu_%A_%a.out
#SBATCH --error=logs/spca_mu_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-7

set -euo pipefail
mkdir -p logs

# Map array index -> mu value
MU_LIST=(0.01 0.1 1 5 10 20 50 100)
MU=${MU_LIST[$SLURM_ARRAY_TASK_ID]}

echo "[task ${SLURM_ARRAY_TASK_ID}] mu=${MU} on $(hostname) at $(date)"

module load matlab

matlab -nodisplay -nosplash -batch "\
dataset = 'news20'; \
load_h; \
N = 500; p = 200; time_limit = 360; \
mu = ${MU}; \
F = @(X) -0.5*trace(X'*(H*X)) + mu*sum(abs(X(:))); \
solvers;"

# Interative Version
salloc --nodes=1 --mem=256gb --cpus-per-task=64 --time=12:00:00
module load matlab
matlab -nodisplay
# SSC MNIST
clear
dataset = 'mnist'; p = 10;
load_ssc;
x_mode = 'time'; N = 10000; time_limit = 360; ssc_solvers;

# SSC USPS
clear
dataset = 'usps'; p = 10;
load_ssc;
x_mode = 'time'; N = 10000; time_limit = 100; ssc_solvers;

# SSC Synthetic
clear
load_ssc_synthetic;
x_mode = 'time'; N = 1000; time_limit = 30; ssc_solvers;

# sPCA Synthetic
clear
load_spca_synthetic;
x_mode = 'iter'; N = 150; time_limit = 1000; spca_solvers;
x_mode = 'time'; N = 10000; time_limit = 90; spca_solvers;

exit
exit
