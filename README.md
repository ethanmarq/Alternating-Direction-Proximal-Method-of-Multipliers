# Interactive Matlab on Slurm
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
### ADPMM-NS algorithm
``` matlab
% F = @(X) 0.5*trace(X'*(Lap*X)) + mu*sum(abs(X(:)));
% Lap: Normalized Laplacian matrix
% L: Largest eigenvalue of Lap
% rho = L
% X0: Orthogonal n x p matrix random initializaiton
X = X0; Z = X0; Y = zeros(n, p);
F_adpmm = zeros(1, N); F_adpmm(1) = F(X);
T_adpmm = zeros(1, N); T_adpmm(1) = 0;

for k = 2:N
    % G = L*I - Lap
    % rho*Z - Y + G*X
    % rho*Z - Y + L*X - Lap*X
    B = L*X - Lap*X + rho*Z - Y;
    % Newton Schulz with 10 iterations
    nrmB = norm(B, 'fro');
    if nrmB < eps
        W = B;
    else
        W = B / (nrmB * 1.05);
        Ip = eye(p);
        for j = 1:10
            W = 0.5 * W * (3*Ip - W'*W);
        end
    end
    X = W;
    % Z-update: soft-thresholding
    V = X + Y/rho;
    Z = sign(V) .* max(abs(V) - mu/rho, 0);
    % Dual update
    Y = Y + rho*(X - Z);
    F_adpmm(k) = F(Z);
    T_adpmm(k) = toc;
    if T_adpmm(k) >= time_limit, break; end
end

```


