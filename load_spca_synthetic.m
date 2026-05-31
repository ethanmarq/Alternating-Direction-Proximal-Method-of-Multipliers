% load_spca_synthetic.m
% Small synthetic sparse-PCA problem with clear low-rank + sparse structure.
% Plants k orthogonal, sparse loading vectors in a spiked covariance, then
% builds H exactly as load_spca.m (unnormalized centered scatter matrix).
% Outputs: dataset, n, p, mu, N, rho, eta, t, L, F, H, time_limit.

% ============================== CONFIG ======================================
dataset    = 'synthetic_spike';
n          = 8000;     % ambient dimension (# variables)  -> H is n x n
p          = 50;       % number of sparse components to extract
k_true     = 50;       % number of planted spikes (true components)
m          = 10000;     % number of samples
s          = 10;      % support size (nonzeros) per planted loading
spike      = 20;      % leading signal strength (variance along components)
noise      = 1;       % isotropic noise std
mu         = 0.10;    % l1 weight
N          = 100;     % outer iterations
seed       = 0;
time_limit = 100;
x_mode = 'iter';   % 'iter' or 'time'

rng(seed);

% ============================== BUILD DATA ==================================
% Disjoint random supports -> orthogonal, sparse, unit-norm loadings.
V    = zeros(n, k_true);
perm = randperm(n);
for j = 1:k_true
    idx       = perm((j-1)*s + (1:s));
    vj        = zeros(n, 1);
    vj(idx)   = randn(s, 1);
    V(:, j)   = vj / norm(vj);
end
spikes  = spike * linspace(1, 0.5, k_true);     % decreasing spike strengths
Z       = randn(m, k_true) .* sqrt(spikes);     % signal scores
X_data  = Z * V' + noise * randn(m, n);         % m samples x n features
fprintf('X_data: %d samples x %d features (%d planted spikes, support %d)\n', ...
    m, n, k_true, s);

% ============================== BUILD H =====================================
a = size(X_data, 1);
b = mean(X_data, 1);
H = X_data' * X_data;
H = H - a*(b'*b);
[n, ~] = size(H);
fprintf('H: %dx%d, nnz=%d, density=%.2e\n', n, n, nnz(H), full(nnz(H)/n^2));

% ============================== PARAMS ======================================
L   = full(eigs(H, 1));               % spectral radius
rho = L;
eta = 1 / (L + rho);                  % gradient step
t   = 1 / L;                          % ManPG step
F   = @(X) -0.5*trace(X'*(H*X)) + mu*sum(abs(X(:)));
fprintf('L=%.3e, rho=%.3e, eta=%.3e, t=%.3e, mu=%.3e\n', ...
    L, rho, eta, t, mu);
