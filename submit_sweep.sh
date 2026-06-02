#!/bin/bash
#SBATCH --array=0-7
#SBATCH --mem=256gb
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=log/test_%a.out

module load matlab
tests=(
    # sPCA news20
    # mu = 0.01
  "dataset='news20';
   load_spca;
   mu=0.01; x_mode='time'; N=1000000; time_limit=150; spca_solvers;"

    # mu = 0.1
  "dataset='news20';
   load_spca;
   mu=0.1; x_mode='time'; N=1000000; time_limit=150; spca_solvers;"

    # mu = 1
  "dataset='news20';
   load_spca;
   mu=1; x_mode='time'; N=1000000; time_limit=150; spca_solvers;"

    # mu = 5
  "dataset='news20';
   load_spca;
   mu=5; x_mode='time'; N=1000000; time_limit=150; spca_solvers;"

    # mu = 10
  "dataset='news20';
   load_spca;
   mu=10; x_mode='time'; N=1000000; time_limit=150; spca_solvers;"

    # mu = 20
  "dataset='news20';
   load_spca;
   mu=20; x_mode='time'; N=1000000; time_limit=150; spca_solvers;"

    # mu = 50
  "dataset='news20';
   load_spca;
   mu=50; x_mode='time'; N=1000000; time_limit=150; spca_solvers;"

    # mu = 100
  "dataset='usps';
   load_spca;
   mu=100; x_mode='time'; N=1000000; time_limit=150; spca_solvers;"
)

# tests=(
#     # SSC usps
#     # mu = 0.01
#   "dataset='usps'; p=10;
#    load_ssc;
#    mu=0.01; x_mode='time'; N=1000000; time_limit=15; ssc_solvers;"

#     # mu = 0.1
#   "dataset='usps'; p=10;
#    load_ssc;
#    mu=0.1; x_mode='time'; N=1000000; time_limit=15; ssc_solvers;"

#     # mu = 1
#   "dataset='usps'; p=10;
#    load_ssc;
#    mu=1; x_mode='time'; N=1000000; time_limit=15; ssc_solvers;"

#     # mu = 5
#   "dataset='usps'; p=10;
#    load_ssc;
#    mu=5; x_mode='time'; N=1000000; time_limit=15; ssc_solvers;"

#     # mu = 10
#   "dataset='usps'; p=10;
#    load_ssc;
#    mu=10; x_mode='time'; N=1000000; time_limit=15; ssc_solvers;"

#     # mu = 20
#   "dataset='usps'; p=10;
#    load_ssc;
#    mu=20; x_mode='time'; N=1000000; time_limit=20; ssc_solvers;"

#     # mu = 50
#   "dataset='usps'; p=10;
#    load_ssc;
#    mu=50; x_mode='time'; N=1000000; time_limit=15; ssc_solvers;"

#     # mu = 100
#   "dataset='usps'; p=10;
#    load_ssc;
#    mu=100; x_mode='time'; N=1000000; time_limit=15; ssc_solvers;"
# )


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
