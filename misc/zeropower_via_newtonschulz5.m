% Newton-Schulz orthogonalisation utilities
% Adopted from Muon by Keller Jordan: https://github.com/KellerJordan/Muon
%
% Requirements:
%   MATLAB R2020b or later  — pagemtimes / pagetranspose
%   MATLAB R2018b or later  — vector dimension argument to sum()
%
% Differences from the Python/PyTorch original:
%   * bfloat16 is unavailable in MATLAB; single precision is used instead.
%   * @torch.compile has no MATLAB equivalent and is omitted.
%   * Batched input convention: matrix dims are always dim-1 and dim-2;
%     any additional trailing dims are treated as batch dims
%     (e.g. m×n×batch, matching MATLAB's pagemtimes convention).
%   * For complex G, replace pagetranspose with pagectranspose so that
%     the conjugate transpose is used throughout.

% -----------------------------------------------------------------------

function [X, varargout] = zeropower_via_newtonschulz5( ...
        G, compute_hermitian, max_iterations, a, b, c)
%ZEROPOWER_VIA_NEWTONSCHULZ5  Newton-Schulz orthogonalisation of G.
%
%   X = ZEROPOWER_VIA_NEWTONSCHULZ5(G)
%   X = ZEROPOWER_VIA_NEWTONSCHULZ5(G, compute_hermitian, max_iterations, a, b, c)
%   [X, H] = ZEROPOWER_VIA_NEWTONSCHULZ5(G, true, ...)
%
%   Uses a quintic iteration whose coefficients are chosen to maximise the
%   slope at zero, producing something like U*S'*V' (S' diagonal with
%   entries ~Uniform(0.5, 1.5)) rather than U*V' from the exact SVD.
%
%   Inputs
%     G                 – real matrix, 2-D (m×n) or batched (m×n×...)
%     compute_hermitian – (optional) also return Hermitian factor H  [false]
%     max_iterations    – (optional) number of NS iterations          [5]
%     a, b, c           – (optional) quintic coefficients   [3.4445, -4.7750, 2.0315]
%
%   Outputs
%     X  – orthogonalised matrix, same leading shape as G
%     H  – (optional) Hermitian factor H = (G'*X' + X*G) / 2

    if nargin < 2 || isempty(compute_hermitian), compute_hermitian = false; end
    if nargin < 3 || isempty(max_iterations),    max_iterations   = 5;     end
    if nargin < 4 || isempty(a),                 a = 3.4445;               end
    if nargin < 5 || isempty(b),                 b = -4.7750;              end
    if nargin < 6 || isempty(c),                 c = 2.0315;               end

    assert(ndims(G) >= 2, 'G must have at least 2 dimensions.');

    % bfloat16 is unavailable in MATLAB; single is the nearest alternative.
    X = single(G);

    rows = size(X, 1);
    cols = size(X, 2);
    transposed = rows > cols;
    if transposed
        X = pagetranspose(X);   % cols×rows×batch…  (wide orientation)
    end

    % Divide by per-page Frobenius norm so that the spectral norm is <= 1.
    % sum(..., [1 2]) reduces over the two matrix dims, leaving batch dims intact.
    nrm = sqrt(sum(X .^ 2, [1, 2]));   % 1×1×batch…
    X   = X ./ (nrm + single(1e-7));

    % Quintic Newton-Schulz iterations.
    for k = 1:max_iterations
        A = pagemtimes(X, pagetranspose(X));        % square: r×r×batch…
        B = b * A + c * pagemtimes(A, A);           % quintic strategy
        X = a * X + pagemtimes(B, X);
    end

    if compute_hermitian
        % H = G' * X'  then symmetrise.
        H = pagemtimes(pagetranspose(cast(G, 'like', X)), pagetranspose(X));
        H = (H + pagetranspose(H)) / 2;
    end

    if transposed
        X = pagetranspose(X);
        if compute_hermitian
            H = pagetranspose(H);
        end
    end

    if compute_hermitian
        varargout{1} = H;
    end
end

% -----------------------------------------------------------------------

