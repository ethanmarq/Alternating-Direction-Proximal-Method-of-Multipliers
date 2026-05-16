# Interactive Matlab on Slurm
```bash
salloc --mem=64gb --cpus-per-task=64 --time=02:00:00
matlab -nodisplay -nosplash
load_h
rho=10;
solvers
```

