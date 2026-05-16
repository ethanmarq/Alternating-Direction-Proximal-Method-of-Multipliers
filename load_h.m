% ============================== CONFIG ======================================
data_path = '/scratch/marque6/libsvm_data/rcv1_train.mat';
p   = 50;       % number of sparse components
mu  = 0.10;     % l1 weight
N   = 100;      % outer iterations
rho = 100;      % ADMM penalty (single value, no sweep)
seed = 0;

% ============================== LOAD ========================================
S = load(data_path);                % expects S.X (n_samples x p_features)
X_data = S.X;
H = X_data' * X_data;               % p_feat x p_feat, no centering
[n, ~] = size(H);
fprintf('H: %dx%d, nnz=%d, density=%.2e\n', n, n, nnz(H), full(nnz(H)/n^2));

L = full(eigs(H, 1));               % spectral radius
eta = 1 / (L + rho);                % gradient step
t   = 1 / L;                        % ManPG step
fprintf('L=%.3e, rho=%.3e, eta=%.3e, t=%.3e\n', L, rho, eta, t);

% Common objective
F = @(X) -0.5*trace(X'*(H*X)) + mu*sum(abs(X(:)));

% Common starting point
rng(seed);
X0 = orth(randn(n, p));
