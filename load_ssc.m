% ============================== CONFIG ======================================
%dataset = 'news20' %news20.binary: 2 classes
data_path = sprintf('/scratch/marque6/libsvm_data/%s.mat', dataset);
p   = 2;       % number of classes
k   = 10;       % nearest neighbors for graph construction
mu  = 0.10;     % l1 weight
N   = 100;      % outer iterations
seed = 0;
time_limit = 60;
% ============================== LOAD ========================================
S = load(data_path); % expects S.X (n_samples x p_features)
X_data = S.X;

rng(seed);
% X_data: INPUT
% k: number of nearest neighbors for sparsity
% noramlized: symmetric normalization (L_sym), otherwise unnormalized D-W
n = size(X_data, 1); % length of first column

[idx, dist] = knnsearch(X_data, X_data, 'K', k+1); % +1 because col 1 is self
idx  = idx(:, 2:end); % drop self-match
dist = dist(:, 2:end);

sigma = median(dist(:)); %RBF kernel width, using median heuristic
if sigma < eps, sigma = 1; end % fix bad points

rows = repmat((1:n)', 1, k);
aff  = exp(-(dist.^2) / (2*sigma^2));
W = sparse(rows(:), idx(:), aff(:), n, n);

W = max(W, W');

deg = full(sum(W, 2));
D   = spdiags(deg, 0, n, n);

dinv = 1 ./ sqrt(max(deg, eps));
Dinv = spdiags(dinv, 0, n, n);
Lap  = speye(n) - Dinv * W * Dinv; % L_sym = I - D^{-1/2} W D^{-1/2}
Lap  = (Lap + Lap') / 2; % clean up numerical asymmetry
% Lap = D - W; % unnormalized

F = @(X) 0.5*trace(X'*(Lap*X)) + mu*sum(abs(X(:)));

[n, ~] = size(Lap);
fprintf('Lap: %dx%d, nnz=%d, density=%.2e\n', n, n, nnz(Lap), full(nnz(Lap)/n^2));

L = full(eigs(Lap, 1));               % spectral radius
rho = L;
eta = 1 / (L + rho);                % gradient step
t   = 1 / L;                        % ManPG step
fprintf('L=%.3e, rho=%.3e, eta=%.3e, t=%.3e\n', L, rho, eta, t);
