# Interactive Matlab on Slurm
## sPCA 
```bash
salloc --mem=64gb --cpus-per-task=64 --time=02:00:00
matlab -nodisplay -nosplash
load_spca
spca_solvers
```

## SSC
```bash
salloc --mem=64gb --cpus-per-task=64 --time=02:00:00
matlab -nodisplay -nosplash
load_ssc
ssc_solvers
```
