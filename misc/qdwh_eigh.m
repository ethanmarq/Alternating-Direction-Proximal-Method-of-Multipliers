function [Uout, eigvals] = qdwh_eigh(H, varargin)
%QDWH_EIGH  Eigendecomposition of a real symmetric matrix via QDWH.
%
%   [UOUT, EIGVALS] = QDWH_EIGH(H)
%       Returns an orthogonal matrix UOUT whose columns are eigenvectors of
%       the symmetric matrix H, and a vector EIGVALS of eigenvalues in
%       ascending order.
%
%       NOTE: Although EIGVALS is sorted in ascending order, UOUT's columns
%       are ordered so that the first column corresponds to the *largest*
%       eigenvalue (matching the behaviour of the original source).
%       Call QDWH_SVD for a consistent SVD interface.
%
%   Name-value options:
%     'normH'   - Frobenius norm of H; computed internally if empty (default [])
%     'minlen'  - minimum subproblem size before stopping recursion (default 1)
%     'NS'      - apply Newton-Schulz post-processing to UOUT (default true)

p = inputParser;
addParameter(p, 'normH',  []);
addParameter(p, 'minlen', 1);
addParameter(p, 'NS',     true);
parse(p, varargin{:});

normH  = p.Results.normH;
minlen = p.Results.minlen;
NS     = p.Results.NS;

eps_val = eps(class(H));
backtol = 10 * eps_val / 2;

if isempty(normH)
    normH = norm(H, 'fro');
end

[Uout, eigvals] = qdwh_eigh_rep(H, normH, minlen, backtol);

if NS
    Uout = 1.5*Uout - 0.5*(Uout * (Uout' * Uout));
end

% Sort eigenvalues ascending; reverse the eigenvector column order so that
% Uout(:,1) corresponds to the largest eigenvalue.
[eigvals, ~] = sort(eigvals(:), 'ascend');
Uout         = fliplr(Uout);
end


%% =========================================================================
%%  Local function: recursive QDWH eigen-solver
%% =========================================================================

function [Uout, eigvals] = qdwh_eigh_rep(H, normH, minlen, backtol)
%QDWH_EIGH_REP  Internal recursive routine called by QDWH_EIGH.

n   = size(H, 1);
I_n = eye(n, class(H));

% ── Base case: H is already nearly diagonal ───────────────────────────────
Hd = diag(diag(H));
if norm(H - Hd, 'fro') / normH < backtol
    [sorted_d, IX] = sort(diag(H), 'descend');
    Uout   = I_n(:, IX);
    eigvals = sorted_d;
    return;
end

% Symmetrise to suppress accumulated round-off.
H = 0.5 * (H + H');

% Median-diagonal shift to improve conditioning.
shift = median(diag(H));
Hs    = H - shift * I_n;

% Polar decomposition of the shifted matrix.
[U, ~, ~] = qdwh(Hs);
U = 0.5 * (U + I_n);           % convert polar factor to orthogonal projector

% ── Subspace iteration to split the spectrum ─────────────────────────────
[U1, U2] = subspaceit(U);
minoff   = blk_off_norm(U2, H, U1, normH);

if minoff > backtol
    [U1, U2] = subspaceit(U, 'U1_init', U1);
    minoff   = blk_off_norm(U2, H, U1, normH);
end

if minoff > backtol
    for iter = 1:2                                          %#ok<FORPF>
        [U1b, U2b] = subspaceit(U, 'use_rand', true);
        minoff2    = blk_off_norm(U2b, H, U1b, normH);
        if minoff2 < minoff
            U1     = U1b;
            U2     = U2b;
            minoff = minoff2;
        end
    end
end

% ── Collect eigenvalues from any trivial 1-column blocks ─────────────────
eigvals_extra = zeros(0, 1, class(H));
if size(U1, 2) == 1
    eigvals_extra(end+1, 1) = U1' * H * U1;
end
if size(U2, 2) == 1
    eigvals_extra(end+1, 1) = U2' * H * U2;
end

eigvals1 = zeros(0, 1, class(H));
eigvals2 = zeros(0, 1, class(H));

% ── Process the U1 block ──────────────────────────────────────────────────
if size(U1, 2) > minlen
    % Recurse.
    H_sub = U1' * H * U1;
    [Ua, eigvals1] = qdwh_eigh_rep(H_sub, normH, minlen, backtol);
    U1 = U1 * Ua;
elseif size(U1, 2) > 1
    % Direct eigensolver for small block (only reached when minlen > 1).
    H_sub     = 0.5 * (U1'*H*U1 + (U1'*H*U1)');
    [Ua, Lam] = eig(H_sub);
    ev        = real(diag(Lam));
    [ev, idx] = sort(ev, 'descend');
    U1        = U1 * Ua(:, idx);
    eigvals1  = flipud(ev);
end

% ── Process the U2 block ──────────────────────────────────────────────────
if size(U2, 2) > minlen
    % Recurse.
    H_sub = U2' * H * U2;
    [Ua, eigvals2] = qdwh_eigh_rep(H_sub, normH, minlen, backtol);
    U2 = U2 * Ua;
elseif size(U2, 2) > 1
    % Direct eigensolver for small block (only reached when minlen > 1).
    H_sub     = 0.5 * (U2'*H*U2 + (U2'*H*U2)');
    [Ua, Lam] = eig(H_sub);
    ev        = real(diag(Lam));
    [ev, idx] = sort(ev, 'descend');
    U2        = U2 * Ua(:, idx);
    eigvals2  = flipud(ev);
end

Uout    = [U1, U2];
eigvals = [eigvals_extra; eigvals1; eigvals2];
end


%% =========================================================================
%%  Local helper: relative norm of an off-diagonal block
%% =========================================================================

function val = blk_off_norm(U2, H, U1, normH)
%BLK_OFF_NORM  Returns ||U2'*H*U1||_F / normH; 0 if U2 is empty.
if isempty(U2) || size(U2, 2) == 0
    val = 0;
else
    val = norm(U2' * H * U1, 'fro') / normH;
end
end