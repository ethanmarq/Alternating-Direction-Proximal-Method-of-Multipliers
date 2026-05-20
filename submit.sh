#!/usr/bin/env sh
#SBATCH --job-name=spca
#SBATCH --output=logs/spca_%A_%a.out
#SBATCH --error=logs/spca_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks=1

salloc --nodes=1 --mem=64gb --cpus-per-task=64 --time=08:00:00
module load matlab
matlab -nodisplay -nosplash
dataset = 'rcv1_train';
load_h
N=5000; p=50; time_limit=360; solvers
exit
exit



dataset = 'news20'
load_h
N=500; p=100; solvers
exit
exit


salloc --nodes=1 --mem=64gb --cpus-per-task=64 --time=08:00:00
module load matlab
matlab -nodisplay -nosplash
dataset = 'rcv1_train';
load_h
N=500; p=10; solvers

module load matlab
matlab -nodisplay -nosplash
dataset = 'rcv1_train';
load_h
N=500; p=100; solvers

dataset = 'rcv1_train';
load_h
N=500; p=200; solvers
