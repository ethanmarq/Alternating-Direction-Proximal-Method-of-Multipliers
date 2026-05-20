#!/usr/bin/env sh
#SBATCH --job-name=spca
#SBATCH --output=logs/spca_%A_%a.out
#SBATCH --error=logs/spca_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1

module load matlab
matlab -nodisplay -nosplash
dataset = 'rcv1_train';
load_h
N=5000; solvers
clear
dataset = 'news20'
load_h
N=5000; solvers
exit
exit
