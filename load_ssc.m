% ============================== CONFIG ======================================
dataset = 'rcv1_train' %news20.binary: 2 classes
data_path = sprintf('/scratch/marque6/libsvm_data/%s.mat', dataset);
p   = 2;       % number of classes
mu  = 1;     % l1 weight
N   = 10000;      % outer iterations
seed = 0;
time_limit = 10;
% ============================== LOAD ========================================
S = load(data_path); % expects S.X (n_samples x p_features)
X_data = S.X;
n = size(X_data, 1);
% mu = 1 / sqrt(n);
fprintf('X_data: %d samples x %d features, nnz=%d\n', n, size(X_data,2), nnz(X_data));

rng(seed);
% X_data: INPUT
% noramlized: symmetric normalization (L_sym), otherwise unnormalized D-W
n = size(X_data, 1); % length of first column

X     = full(X_data);
X      = X_data;
nx2    = sum(X.^2, 2);
D2     = nx2 + nx2' - 2*(X*X');
D2     = max(full(D2), 0);
D2(1:n+1:end) = 0;

sigma2 = median(D2(D2 > 0));
W     = exp(-D2 / (2*sigma2));
W(1:n+1:end) = 0;
deg   = sum(W, 2);
Lap   = diag(deg) - W;

F = @(X) 0.5*trace(X'*(Lap*X)) + mu*sum(abs(X(:)));

[n, ~] = size(Lap);
fprintf('Lap: %dx%d, nnz=%d, density=%.2e\n', n, n, nnz(Lap), full(nnz(Lap)/n^2));

L = full(eigs(Lap, 1));               % spectral radius
rho = .5*L;
eta = 1 / (L + rho);                % gradient step
t   = 1 / L;                        % ManPG step
fprintf('L=%.3e, rho=%.3e, eta=%.3e, t=%.3e\n', L, rho, eta, t);
