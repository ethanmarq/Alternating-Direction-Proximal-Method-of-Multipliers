% ============================== CONFIG ======================================
dataset = 'news20' %news20.binary: 2 classes
data_path = sprintf('/scratch/marque6/libsvm_data/%s.mat', dataset);
p   = 2;       % number of classes
k   = 10;       % nearest neighbors for graph construction
mu  = 0.10;     % l1 weight
N   = 10000;      % outer iterations
seed = 0;
time_limit = 10;
% ============================== LOAD ========================================
S = load(data_path); % expects S.X (n_samples x p_features)
X_data = S.X;
n = size(X_data, 1);
fprintf('X_data: %d samples x %d features, nnz=%d\n', n, size(X_data,2), nnz(X_data));

rng(seed);
% X_data: INPUT
% k: number of nearest neighbors for sparsity
% noramlized: symmetric normalization (L_sym), otherwise unnormalized D-W
n = size(X_data, 1); % length of first column

% L2-normalize rows so inner product = cosine similarity
rn = sqrt(sum(X_data.^2, 2));  rn(rn < eps) = 1;
Xn = spdiags(1./rn, 0, n, n) * X_data; % sparse, unit-norm rows

% Block-wise top-k cosine neighbours
bs = 2000; % rows per block
I  = zeros(n, k);  Dk = zeros(n, k);
for s = 1:bs:n
    e = min(s+bs-1, n);
    Sblk = full(Xn(s:e, :) * Xn'); % (e-s+1) x n
    Sblk(sub2ind(size(Sblk), 1:(e-s+1), s:e)) = -inf; % mask self
    [vals, ix] = maxk(Sblk, k, 2); % top-k per row
    I(s:e, :)  = ix;
    Dk(s:e, :) = vals;
    fprintf('  knn rows %d-%d / %d\n', s, e, n);
end

rows = repmat((1:n)', 1, k);
aff  = max(Dk, 0); % cosine similarity as weight
W = sparse(rows(:), I(:), aff(:), n, n);
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
