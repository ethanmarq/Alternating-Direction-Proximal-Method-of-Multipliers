# Interactive Matlab on Slurm
## Results
Figures in detail under ./paper_figures/

## sPCA 
```bash
salloc --mem=64gb --cpus-per-task=64 --time=02:00:00
matlab -nodisplay -nosplash
dataset='news20' # ['news20' 'rcv1']
load_spca # load_spca (requries dataset) OR load_spca_synthetic (does not require dataset)
spca_solvers
```

## SSC
```bash
salloc --mem=64gb --cpus-per-task=64 --time=02:00:00
matlab -nodisplay -nosplash
dataset='mnist' # ['mnist' 'usps']
load_ssc # load_ssc (requries dataset) OR load_ssc_synthetic (does not require dataset)
ssc_solvers
```
