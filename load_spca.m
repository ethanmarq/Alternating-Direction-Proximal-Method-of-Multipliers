% ============================== CONFIG ======================================
% dataset = 'rcv1_train'
data_path = sprintf('/scratch/marque6/libsvm_data/%s.mat', dataset);
p   = 50;       % number of sparse components
mu  = 0.10;     % l1 weight
N   = 100;      % outer iterations
seed = 0;
time_limit = 60;
x_mode = 'iter';   % 'iter' or 'time'

% ============================== LOAD ========================================
S = load(data_path);                % expects S.X (n_samples x p_features)
X_data = S.X;
a  = size(X_data, 1);
b = mean(X_data, 1);
H  = X_data' * X_data;
H = H - a*(b'*b);
[n, ~] = size(H);
fprintf('H: %dx%d, nnz=%d, density=%.2e\n', n, n, nnz(H), full(nnz(H)/n^2));

L = full(eigs(H, 1));               % spectral radius
rho = L;
eta = 1 / (L + rho);                % gradient step
t   = 1 / L;                        % ManPG step
fprintf('L=%.3e, rho=%.3e, eta=%.3e, t=%.3e\n', L, rho, eta, t);


% Common objective
F = @(X) -0.5*trace(X'*(H*X)) + mu*sum(abs(X(:)));

% Common starting point
rng(seed);

