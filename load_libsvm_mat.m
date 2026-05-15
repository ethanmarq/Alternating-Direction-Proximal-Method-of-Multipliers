function [H, X, y, info] = load_libsvm_mat(mat_path, mode, opts)
% LOAD_LIBSVM_MAT  Load a LIBSVM-derived .mat file and build H for sPCA / SSC.
%
%   [H, X, y, info] = load_libsvm_mat(mat_path)
%   [H, X, y, info] = load_libsvm_mat(mat_path, mode)
%   [H, X, y, info] = load_libsvm_mat(mat_path, mode, opts)
%
% Inputs:
%   mat_path : path to a .mat saved by download_libsvm_to_mat.py
%   mode     : 'spca'      -> H = (1/n) * X' * X   (feature-space covariance,
%                                                   p_feat x p_feat).
%              'spca_n'    -> H = (1/p) * X * X'   (sample-space gram,
%                                                   n x n; useful when n < p).
%              'ssc_knn'   -> Build a symmetric k-NN affinity W, then return
%                             H = normalized Laplacian.
%              'ssc_gauss' -> Gaussian affinity, then normalized Laplacian.
%              (default: 'spca')
%   opts     : struct with optional fields
%                .k          (k-NN; default 10)
%                .sigma      (Gaussian bandwidth; default median pairwise dist)
%                .max_n      (subsample to at most this many rows; default Inf)
%                .center     (center features before covariance; default true)
%                .seed       (rng seed for subsampling; default 0)
%
% Outputs:
%   H    : symmetric matrix ready to plug into your sPCA / SSC drivers.
%          For sPCA your code uses f(X) = -0.5 * trace(X' * H * X), so pass
%          H = X' * X (the data Gram in feature space). For SSC you want
%          f(X) =  0.5 * trace(X' * L * X), so the sign convention may need
%          to be flipped in the driver.
%   X    : sparse n x p data matrix as stored.
%   y    : n x 1 labels (for clustering eval via NMI / ARI / ACC).
%   info : struct with .n, .p, .name, .mode, and any derived parameters.
%
% Example:
%   [H, X, y, info] = load_libsvm_mat('libsvm_data/mat/gisette.mat', 'spca');
%   % Then in compare_sPCA.m, replace the synthetic-H block with this H.
%
%   [L, X, y, info] = load_libsvm_mat('libsvm_data/mat/usps.mat', ...
%                                     'ssc_knn', struct('k', 10, 'max_n', 5000));

    if nargin < 2 || isempty(mode), mode = 'spca'; end
    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'k'),      opts.k = 10;       end
    if ~isfield(opts, 'max_n'),  opts.max_n = Inf;  end
    if ~isfield(opts, 'center'), opts.center = true; end
    if ~isfield(opts, 'seed'),   opts.seed = 0;     end

    S = load(mat_path);
    X = S.X;        % sparse n x p
    y = S.y;        % n x 1
    [n, p] = size(X);

    % Optional subsampling (useful for the huge ones)
    if isfinite(opts.max_n) && n > opts.max_n
        rng(opts.seed);
        idx = randperm(n, opts.max_n);
        X = X(idx, :);
        y = y(idx);
        n = size(X, 1);
    end

    info = struct('n', n, 'p', p, 'name', char(S.name), 'mode', mode);

    switch lower(mode)
        case 'spca'
                H = (X' * X);
            end

        case 'spca_n'
            % Sample-space gram: n x n  (good when n < p)
            if opts.center
                mu = full(mean(X, 1));
                Xc_row_norms = (X * mu') * 2;         % cross term
                H = X * X' - Xc_row_norms - Xc_row_norms' + n * (mu * mu');
                H = (H + H') / 2;
                H = H / max(p - 1, 1);
            else
                H = X * X';
                H = (H + H') / 2;
            end

        case 'ssc_knn'
            W = knn_affinity(X, opts.k);
            H = normalized_laplacian(W);
            info.affinity = 'symmetric k-NN, binary';
            info.k = opts.k;

        case 'ssc_gauss'
            if ~isfield(opts, 'sigma') || isempty(opts.sigma)
                % median heuristic on a sample
                m = min(n, 2000);
                rng(opts.seed);
                ii = randperm(n, m);
                D = pdist2(full(X(ii,:)), full(X(ii,:)));
                opts.sigma = median(D(D > 0));
            end
            W = gauss_affinity(X, opts.sigma);
            H = normalized_laplacian(W);
            info.affinity = 'Gaussian';
            info.sigma = opts.sigma;

        otherwise
            error('Unknown mode: %s', mode);
    end
end


% ------------------------- affinity helpers ----------------------------------

function W = knn_affinity(X, k)
% Symmetric k-NN affinity (binary). For large n you may want to swap to ANN.
    [n, ~] = size(X);
    % pdist2 on sparse needs full rows; do it in blocks to stay memory-safe.
    block = 1000;
    rows = cell(ceil(n/block), 1);
    cols = cell(ceil(n/block), 1);
    for b = 1:ceil(n/block)
        i0 = (b-1)*block + 1;
        i1 = min(b*block, n);
        Xi = full(X(i0:i1, :));
        D  = pdist2(Xi, full(X));               % (i1-i0+1) x n
        D(:, i0:i1) = D(:, i0:i1) + diag(inf(i1-i0+1,1));  % exclude self
        [~, idx] = mink(D, k, 2);
        r = repmat((i0:i1)', 1, k);
        rows{b} = r(:);
        cols{b} = idx(:);
    end
    rows = vertcat(rows{:});
    cols = vertcat(cols{:});
    W = sparse(rows, cols, 1, n, n);
    W = max(W, W');                              % symmetrize
end


function W = gauss_affinity(X, sigma)
    n = size(X, 1);
    block = 1000;
    Wcells = cell(ceil(n/block), 1);
    for b = 1:ceil(n/block)
        i0 = (b-1)*block + 1;
        i1 = min(b*block, n);
        D  = pdist2(full(X(i0:i1,:)), full(X));
        Wcells{b} = sparse(exp(-(D.^2) / (2 * sigma^2)));
    end
    W = vertcat(Wcells{:});
    W = W - spdiags(diag(W), 0, n, n);          % zero diagonal
    W = (W + W') / 2;
end


function L = normalized_laplacian(W)
    d = sum(W, 2);
    d(d == 0) = 1;                              % avoid div-by-zero
    dinv = 1 ./ sqrt(d);
    n = size(W, 1);
    Dinv = spdiags(dinv, 0, n, n);
    L = speye(n) - Dinv * W * Dinv;
    L = (L + L') / 2;
end
