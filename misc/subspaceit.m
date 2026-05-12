function [U0, U1] = subspaceit(U, varargin)
%SUBSPACEIT  Two steps of subspace iteration to split an invariant subspace.
%
%   [U0, U1] = SUBSPACEIT(U)
%       Performs two QR-based subspace iteration steps on the n x n matrix U
%       and returns orthonormal bases U0 (first ~rank(U) columns) and U1
%       (complementary columns).
%
%   [U0, U1] = SUBSPACEIT(U, 'use_rand', true)
%       Uses a random starting matrix instead of the leading columns of U.
%
%   [U0, U1] = SUBSPACEIT(U, 'U1_init', V)
%       Uses V as the initial subspace guess (overrides 'use_rand').

p = inputParser;
addParameter(p, 'use_rand', false);
addParameter(p, 'U1_init', []);
parse(p, varargin{:});

use_rand = p.Results.use_rand;
U1_init  = p.Results.U1_init;

n     = size(U, 1);
% Estimate the rank as round(||U||_F^2); clamp to [1, n].
xsize = max(1, min(round(norm(U, 'fro')^2), n));
k     = min(xsize + 3, n);

% Choose starting matrix for the iteration.
if ~isempty(U1_init)
    UU = U * U1_init;
elseif use_rand
    UU = U * randn(n, k, class(U));
else
    UU = U(:, 1:k);
end

% Two steps of orthogonalisation.
[Q,  ~] = qr(UU, 0);
UU      = U * Q;
[Q2, ~] = qr(UU, 0);

% Split into two complementary orthonormal blocks.
U0 = Q2(:, 1:xsize);
if xsize < size(Q2, 2)
    U1 = Q2(:, xsize+1:end);
else
    U1 = zeros(n, 0, class(U));   % empty second block
end
end
