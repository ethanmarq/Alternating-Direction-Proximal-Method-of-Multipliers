#!/bin/bash
#SBATCH --array=0-5
#SBATCH --mem=256gb
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=log/test_%a.out

module load matlab

tests=(
  # SSC MNIST
  "dataset='mnist'; p=10;
   load_ssc;
   x_mode='time'; N=10000; time_limit=300; ssc_solvers;
   x_mode='iter'; N=300;   time_limit=300; ssc_solvers;"

  # SSC USPS
  "dataset='usps'; p=10;
   load_ssc;
   x_mode='time'; N=10000; time_limit=20;  ssc_solvers;
   x_mode='iter'; N=300;   time_limit=100; ssc_solvers;"

  # SSC Synthetic
  "load_ssc_synthetic;
   x_mode='iter'; N=100;   time_limit=100; ssc_solvers;
   x_mode='time'; N=10000; time_limit=5;   ssc_solvers;"

  # sPCA Synthetic
  "load_spca_synthetic;
   x_mode='iter'; N=30;    time_limit=1000; spca_solvers;
   x_mode='time'; N=10000; time_limit=5;    spca_solvers;"

  # sPCA news20
  "dataset='news20';
   load_spca;
   x_mode='iter'; N=20;    time_limit=1000; spca_solvers;
   x_mode='time'; N=10000; time_limit=90;   spca_solvers;"

  # sPCA rcv1
  "dataset='rcv1_train';
   load_spca;
   x_mode='iter'; N=40;    time_limit=1000; spca_solvers;
   x_mode='time'; N=10000; time_limit=90;   spca_solvers;"
)

matlab -batch "log/run_test_${SLURM_ARRAY_TASK_ID}"
rm -f "$testfile"


testfile="log/run_test_${SLURM_ARRAY_TASK_ID}.m"
cat > "$testfile" <<EOF
try
${tests[$SLURM_ARRAY_TASK_ID]}
catch e
  disp(getReport(e, 'extended', 'hyperlinks', 'off'));
  exit(1);
end
EOF
matlab -batch "addpath('log'); run_test_${SLURM_ARRAY_TASK_ID}"
rm -f "$testfile"
