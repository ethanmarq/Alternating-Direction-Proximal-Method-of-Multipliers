function [X, varargout] = precond_newtonschulz(G, compute_hermitian)
%PRECOND_NEWTONSCHULZ  Preconditioned Newton-Schulz polar factor of G.
%
%   X = PRECOND_NEWTONSCHULZ(G)
%   [X, H] = PRECOND_NEWTONSCHULZ(G, true)
%
%   Computes the polar factor of G via a preconditioned NS iteration
%   followed by a standard NS postprocessing loop.
%
%   Inputs
%     G                 – real matrix, 2-D (m×n) or batched (m×n×...)
%     compute_hermitian – (optional) also return Hermitian factor H  [false]
%
%   Outputs
%     X  – polar factor of G, same leading shape as G
%     H  – (optional) Hermitian factor H = (G'*X' + X*G) / 2

    if nargin < 2 || isempty(compute_hermitian), compute_hermitian = false; end

    assert(ndims(G) >= 2, 'G must have at least 2 dimensions.');

    X = single(G);

    rows = size(X, 1);
    cols = size(X, 2);
    transposed = rows > cols;
    if transposed
        X = pagetranspose(X);
    end

    % Divide by per-page Frobenius norm.
    nrm = sqrt(sum(X .^ 2, [1, 2]));
    X   = X ./ (nrm + single(1e-7));

    % Preconditioned iterations.
    % a is derived so that s (a scalar tracking convergence) reaches s_ = 0.1.
    s   = eps('single');
    s_  = 0.1;
    a   = 1.5 * sqrt(3) - s_;

    while s < s_
        s = a * s * (1 - (4/27) * a^2 * s^2);
        X = a * X - (4/27) * a^3 * pagemtimes(pagemtimes(X, pagetranspose(X)), X);
    end

    % Newton-Schulz postprocessing: iterate until change falls below tolerance.
    tol   = max([rows, cols]) * eps('single');
    delta = Inf;
    while delta > tol
        X_new = (3/2) * X - (1/2) * pagemtimes(pagemtimes(X, pagetranspose(X)), X);
        diff  = X_new - X;
        % Take the maximum Frobenius norm across all pages (scalar convergence test).
        delta = max(sqrt(sum(diff .^ 2, [1, 2])), [], 'all');
        X     = X_new;
    end

    if compute_hermitian
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
