function [u, varargout] = qdwh(x, varargin)
%QDWH  QR-based Dynamically Weighted Halley (QDWH) polar decomposition.
%
%   [U, NUM_ITERS, IS_CONVERGED] = QDWH(X)
%       Computes the polar factor U of X, so that X = U*H where H is
%       symmetric positive semi-definite.
%
%   [U, H, NUM_ITERS, IS_CONVERGED] = QDWH(X, 'compute_hermitian', true)
%       Also returns the symmetric (Hermitian) factor H = (U'*X + X'*U)/2.
%
%   Name-value options:
%     'is_hermitian'      - logical, currently unused (default false)
%     'compute_hermitian' - logical (default false)
%     'max_iterations'    - positive integer (default 10)
%     'eps'               - machine-epsilon override; [] means auto (default [])
%     'dynamic_shape'     - [m, n] if X is zero-padded to a larger size (default [])
%
%   Reference:
%     Nakatsukasa, Bai, Gygi (2010). "Optimizing Halley's iteration for
%     computing the matrix polar decomposition." SIAM J. Matrix Anal. Appl.,
%     31(5), 2700-2720.

p = inputParser;
addParameter(p, 'is_hermitian',      false);
addParameter(p, 'compute_hermitian', false);
addParameter(p, 'max_iterations',    10);
addParameter(p, 'eps',               []);
addParameter(p, 'dynamic_shape',     []);
parse(p, varargin{:});

compute_hermitian = p.Results.compute_hermitian;
max_iterations    = p.Results.max_iterations;
eps_val           = p.Results.eps;
dynamic_shape     = p.Results.dynamic_shape;

% Trim to true shape if a padded input is provided.
if ~isempty(dynamic_shape)
    m = dynamic_shape(1);
    n = dynamic_shape(2);
    x = x(1:m, 1:n);
end

[u, num_iters, is_converged] = qdwh_core(x, max_iterations, eps_val);

if compute_hermitian
    h = u' * x;
    h = (h + h') / 2;
    varargout = {h, num_iters, is_converged};
else
    varargout = {num_iters, is_converged};
end
end


%% =========================================================================
%%  Local function: core QDWH iteration loop
%% =========================================================================

function [u, num_iters, is_converged] = qdwh_core(x, max_iterations, eps_val)

if isempty(eps_val)
    eps_val = eps(class(x));   % e.g. 2.22e-16 for double, 1.19e-7 for single
end

[~, n] = size(x);

% Scale x so its 2-norm is approximately 1.
one_norm = norm(x, 1);
inf_norm = norm(x, inf);
if one_norm == 0
    alpha_inv = 1;
else
    alpha_inv = 1 / sqrt(one_norm * inf_norm);
end
u = x * alpha_inv;

% l is a lower bound approximation of the smallest singular value of u.
l        = eps_val;
tol_l    = 10 * eps_val / 2;
tol_norm = tol_l^(1/3);

CHOLESKY_CUTOFF = 100;

% Pre-compute iteration coefficients.
qr_coefs   = {};
chol_coefs = {};
k = 0;
while l + tol_l < 1 && k < max_iterations
    k   = k + 1;
    l2  = l^2;
    dd  = (4 * (1/l2 - 1) / l2)^(1/3);
    sqd = sqrt(1 + dd);
    a   = sqd + sqrt(2 - dd + 2*(2 - l2)/(l2 * sqd));
    b   = (a - 1)^2 / 4;
    c   = a + b - 1;
    l   = l * (a + b*l2) / (1 + c*l2);

    e = b / c;
    if c > CHOLESKY_CUTOFF
        qr_coefs{end+1} = struct('amebsc', (a-e)/sqrt(c), ...
                                 'sqrtc',  sqrt(c), ...
                                 'e',      e);          %#ok<AGROW>
    else
        chol_coefs{end+1} = struct('ame', a-e, 'c', c, 'e', e); %#ok<AGROW>
    end
end

% ── QR-based iterations (no convergence check) ────────────────────────────
for k_idx = 1:numel(qr_coefs)
    u = qdwh_use_qr(u, qr_coefs{k_idx});
end

% ── Cholesky-based iterations (check convergence on the last step) ────────
is_not_converged = true;
for k_idx = 1:numel(chol_coefs)
    u_prev = u;
    u      = qdwh_use_chol(u, n, chol_coefs{k_idx});
    if k_idx == numel(chol_coefs)
        is_not_converged = norm(u - u_prev, 'fro') > tol_norm;
    end
end

% ── Halley iterations (a=3, b=1, c=3) until convergence ─────────────────
%    As l -> 1 the QDWH coefficients converge to Halley's method.
halley = struct('ame', 3 - 1/3, 'c', 3, 'e', 1/3);
k_counter = numel(qr_coefs) + numel(chol_coefs);
while is_not_converged && k_counter < max_iterations
    u_prev = u;
    u      = qdwh_use_chol(u, n, halley);
    is_not_converged = norm(u - u_prev, 'fro') > tol_norm;
    k_counter = k_counter + 1;
end

num_iters = k_counter;

% Final Newton-Schulz refinement for improved orthogonality.
u = 1.5*u - 0.5*(u * (u' * u));

is_converged = ~is_not_converged;
end


%% =========================================================================
%%  Local function: one QDWH step via QR decomposition
%% =========================================================================

function u_new = qdwh_use_qr(u, p)
%QDWH_USE_QR  One QDWH update using a QR factorisation.
%
%   Stacks [sqrt(c)*U; I] and applies economy QR.  No matrix inversion needed.
%   Struct p fields: amebsc = (a-e)/sqrt(c),  sqrtc = sqrt(c),  e.

[m, n]  = size(u);
amebsc  = p.amebsc;
sqrtc   = p.sqrtc;
e       = p.e;

% Build the (m+n) x n matrix for the QR step.
y      = [sqrtc * u; eye(n, class(u))];
[q, ~] = qr(y, 0);          % economy QR; q is (m+n) x n

q1     = q(1:m,     :);     % m x n
q2     = q(m+1:end, :)';    % n x n  (conjugate-transpose for complex support)

u_new  = e * u + amebsc * (q1 * q2);
end


%% =========================================================================
%%  Local function: one QDWH step via Cholesky decomposition
%% =========================================================================

function u_new = qdwh_use_chol(u, n, p)
%QDWH_USE_CHOL  One QDWH update using a Cholesky factorisation.
%
%   Avoids explicit matrix inversion by solving two triangular systems.
%   Struct p fields: ame = a-e,  c,  e.

ame = p.ame;
c   = p.c;
e   = p.e;

% Form the n x n SPD matrix:  X = c*(U'*U) + I.
X = c * (u' * u) + eye(n, class(u));

% Cholesky:  R'*R = X   (R is upper triangular in MATLAB).
R = chol(X);

% Compute  Z = U * inv(X)  via two back-substitutions:
%   inv(X) * U'  =  R \ (R' \ U')
%   =>  Z  =  (R \ (R' \ U'))'
z = (R \ (R' \ u'))';

u_new = e * u + ame * z;
end