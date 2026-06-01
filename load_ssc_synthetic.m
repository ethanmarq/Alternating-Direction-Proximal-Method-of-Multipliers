% load_ssc_synthetic.m
% Small synthetic SSC problem with clear cluster structure.
% Uses the same Gaussian-kernel + unnormalized Laplacian as load_ssc.m.
% Outputs: dataset, n, p, mu, N, rho, eta, t, L, F, Lap, time_limit.

% ============================== CONFIG ======================================
dataset    = 'synthetic_blobs';
n_per      = 1000;     % points per cluster
p          = 10;       % number of clusters / orthogonal columns
d          = 20;      % ambient dimension of the data points
sep        = 5;       % cluster center scale (well-separated blobs)
N          = 100;
seed       = 0;
time_limit = 30;
x_mode = 'iter';   % 'iter' or 'time'


rng(seed);
% ============================== BUILD DATA ==================================
n       = n_per * p;
centers = sep * randn(p, d);
labels  = repelem((1:p)', n_per, 1);
X_data  = centers(labels, :) + randn(n, d);
fprintf('X_data: %d samples x %d features (%d blobs of %d)\n', ...
    n, d, p, n_per);

% ============================== BUILD LAPLACIAN =============================
X      = X_data;
nx2    = sum(X.^2, 2);
D2     = nx2 + nx2' - 2*(X*X');
D2     = max(full(D2), 0);
D2(1:n+1:end) = 0;
sigma2 = median(D2(D2 > 0));
W      = exp(-D2 / (2*sigma2));
W(1:n+1:end) = 0;
deg    = sum(W, 2);
Lap    = diag(deg) - W;
fprintf('Lap: %dx%d, nnz=%d, density=%.2e\n', ...
    n, n, nnz(Lap), full(nnz(Lap)/n^2));

% ============================== PARAMS ======================================
mu  = .5;
F   = @(X) 0.5*trace(X'*(Lap*X)) + mu*sum(abs(X(:)));
L   = full(eigs(Lap, 1));
rho = 0.5 * L;
eta = 1 / (L + rho);
t   = 1 / L;
fprintf('L=%.3e, rho=%.3e, eta=%.3e, t=%.3e, mu=%.3e\n', ...
    L, rho, eta, t, mu);
