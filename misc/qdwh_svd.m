function [Uout, singvals, Vout] = qdwh_svd(A, varargin)
%QDWH_SVD  Singular value decomposition of a matrix via QDWH.
%
%   [U, S, V] = QDWH_SVD(A)
%       Computes a (thin) SVD of the m x n matrix A such that
%           A  ≈  U * diag(S) * V'
%       where U is m x n with orthonormal columns, V is n x n orthogonal,
%       and S is an n-vector of singular values in descending order.
%
%   The algorithm first computes the polar decomposition A = Uini * H using
%   QDWH, then eigendecomposes the symmetric factor H via QDWH_EIGH.
%
%   Name-value options:
%     'minlen' - minimum subproblem size for the recursive eigen-solver (default 1)
%     'NS'     - Newton-Schulz post-processing for improved orthogonality (default true)
%
%   Requires on the MATLAB path: qdwh.m, qdwh_eigh.m, subspaceit.m

p = inputParser;
addParameter(p, 'minlen', 1);
addParameter(p, 'NS',     true);
parse(p, varargin{:});

minlen = p.Results.minlen;
NS     = p.Results.NS;

[m, n] = size(A);

% Ensure A is tall (m >= n); transpose if it is wide.
do_flip = (m < n);
if do_flip
    A = A';
    [m, n] = size(A);
end

normA = norm(A, 'fro');

% ── Initial QR reduction for highly rectangular matrices ─────────────────
%    When m > 1.15*n the QR step reduces cost before the polar decomposition.
Qini = [];
if m > 1.15 * n
    [Qini, A] = qr(A, 0);   % economy QR; A is now n x n
    m = n;                   %#ok<NASGU>
end

% ── Polar decomposition: A = Uini * HH ───────────────────────────────────
[Uini, HH, ~, ~] = qdwh(A, 'compute_hermitian', true);

% Detect rank deficiency (polar factor fails to be unitary).
rankdef = (norm(Uini, 'fro')^2 < n - 0.5);

% ── Eigendecomposition of the symmetric factor HH ────────────────────────
%    Eigenvalues of HH are the singular values of A.
[Vout, singvals] = qdwh_eigh(HH, 'normH', normA, 'minlen', minlen, 'NS', NS);

% Sort singular values descending and permute Vout accordingly.
[singvals, idx] = sort(singvals, 'descend');
Vout = Vout(:, idx);

% ── Form left singular vectors ────────────────────────────────────────────
Uout = Uini * Vout;
if ~isempty(Qini)
    Uout = Qini * Uout;
end

% ── Rank-deficient correction: re-orthogonalise via QR ───────────────────
if rankdef
    [Uout, R] = qr(Uout, 0);
    % Multiply each column by the sign of the corresponding diagonal of R
    % so that the leading non-zero entry in each column is positive.
    signs = sign(diag(R))';
    Uout  = Uout .* signs;
end

% ── Newton-Schulz post-processing ────────────────────────────────────────
if NS
    Uout = 1.5*Uout - 0.5*(Uout * (Uout' * Uout));
end

% ── Restore orientation for originally wide matrices ─────────────────────
if do_flip
    [Uout, Vout] = deal(Vout, Uout);
end
end