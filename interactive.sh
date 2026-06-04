#!/usr/bin/env sh

# =========== #
# Interactive #

salloc --nodes=1 --mem=256gb --cpus-per-task=64 --time=12:00:00
module load matlab
matlab -nodisplay

# SSC MNIST
clear
dataset = 'mnist'; p = 10;
load_ssc;
x_mode = 'time'; N = 10000; time_limit = 300; ssc_solvers;
x_mode = 'iter'; N = 300; time_limit = 300; ssc_solvers;

# SSC USPS
clear
dataset = 'usps'; p = 10;
load_ssc;
x_mode = 'time'; N = 10000; time_limit = 20; ssc_solvers;
x_mode = 'iter'; N = 300; time_limit = 100; ssc_solvers;

# SSC Synthetic
clear
load_ssc_synthetic;
x_mode = 'iter'; N = 100; time_limit = 100; ssc_solvers;
x_mode = 'time'; N = 10000; time_limit = 5; ssc_solvers;

# sPCA Synthetic
clear
load_spca_synthetic;
x_mode = 'iter'; N = 30; time_limit = 1000; spca_solvers;
x_mode = 'time'; N = 10000; time_limit = 5; spca_solvers;

# sPCA news20
clear
dataset = 'news20';
load_spca;
x_mode = 'time'; N = 10000; time_limit = 90; spca_solvers;
x_mode = 'iter'; N = 20; time_limit = 1000; spca_solvers;


# sPCA rcv1
clear
dataset = 'rcv1_train';
load_spca;
x_mode = 'time'; N = 10000; time_limit = 200; spca_solvers;
x_mode = 'iter'; N = 40; time_limit = 1000; spca_solvers;

exit
exit

# Testing Scaling Plots
salloc --nodes=1 --mem=256gb --cpus-per-task=64 --time=12:00:00
module load matlab
matlab -nodisplay
clear
dataset = 'news20';
load_spca;
x_mode = 'time'; N = 5; time_limit = 90; mu = 50; spca_solvers;
exit
exit
