function params = compute_step_sizes(H, mu, opts)
% COMPUTE_STEP_SIZES  Principled step sizes and penalty parameters scaled to H.
%
%   params = compute_step_sizes(H, mu)
%   params = compute_step_sizes(H, mu, opts)
%
% Returns a struct of step sizes / penalties for every solver, scaled by
% lambda_max(H) instead of hardcoded constants. This is what prevents
% NaN/Inf when going from synthetic toy problems to real datasets.
%
% Inputs:
%   H    : Stiefel-problem matrix (p x p, dense or sparse). For sPCA this is
%          the data Gram (=X'X or covariance); for SSC this is the Laplacian.
%   mu   : l1 regularization weight
%   opts : (optional) struct with fields
%            .rho_scale  : multiplier on lambda_max for ADMM penalty
%                          (default 1.0; sweep this if needed)
%            .verbose    : print chosen values (default true)
%
% Outputs (fields of params):
%   .lambda_max : spectral radius of H (the master scale)
%   .L          : Lipschitz constant of nabla f = lambda_max (sPCA: f=-0.5*tr(X'HX))
%
%   For each solver, the step size / penalty that has theoretical backing:
%
%   ManPG      : .manpg_t      = 1 / L                  [Chen-Ma-So-Zhang 2020]
%   SOC        : .soc_eta      = 1 / (L + rho)          inner PG step
%                .soc_rho      = max(1, lambda_max * rho_scale)
%   MADMM      : .madmm_eta    = 1 / (L + rho)
%                .madmm_rho    = max(1, lambda_max * rho_scale)
%   RADMM      : .radmm_eta    = 1 / (L + rho)          [Li-Ma-Srivastava 2022]
%                .radmm_rho    = max(1, lambda_max * rho_scale)
%                .radmm_gamma  = 1e-8                   (Moreau envelope)
%   ARADMM     : .aradmm_eta0  = 1 / (L + rho0)         decayed by iter^(1/3)
%                .aradmm_rho0  = max(1, lambda_max * rho_scale / 10)
%   OADMM      : .oadmm_rho0   = max(1, lambda_max * rho_scale)
%                                (NOT 10*mu — that's tiny for real data!)
%                .oadmm_sigma  = 1.1
%                .oadmm_delta  = 1e-3
%   AD-PMM     : .adpmm_rho    = max(1, lambda_max * rho_scale)
%                .adpmm_ns_steps = 5                    (Newton-Schulz iters)
%
% Sweep guidance: if NaN/Inf persists, sweep .rho_scale over [0.01, 0.1, 1, 10, 100].
%
% Example:
%   params = compute_step_sizes(H, 0.01);
%   % Plug into the solvers in place of hardcoded etas/rhos.

    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'rho_scale'), opts.rho_scale = 1.0; end
    if ~isfield(opts, 'verbose'),   opts.verbose   = true; end

    % -------- master scale: largest eigenvalue of H --------
    % This is the Lipschitz constant of the smooth part for sPCA / SSC.
    % Using eigs is fine for sparse H; falls back to full() for tiny dense H.
    n = size(H, 1);
    if n <= 500
        L = max(abs(eig(full(H))));   % exact for small matrices
    else
        % iterative; works for sparse. 'lm' = largest magnitude.
        try
            L = abs(eigs(H, 1, 'lm', struct('disp', 0, 'maxit', 300)));
        catch
            % fallback for rank-deficient or numerically tricky H
            L = svds(H, 1);
        end
    end

    % Guard against pathological cases
    if ~isfinite(L) || L <= 0
        warning('compute_step_sizes:bad_L', ...
                'Spectral radius came back as %g; falling back to 1', L);
        L = 1.0;
    end

    rho_base = max(eps, L * opts.rho_scale);

    % -------- assemble params --------
    params = struct();
    params.lambda_max = L;
    params.L          = L;
    params.rho_scale  = opts.rho_scale;

    % ManPG: theoretical t = 1/L from [Chen-Ma-So-Zhang 2020]
    params.manpg_t = 1 / L;

    % SOC: inner proximal-gradient step. Eta = 1/(L + rho).
    params.soc_rho = rho_base;
    params.soc_eta = 1 / (L + params.soc_rho);

    % MADMM: Riemannian gradient step
    params.madmm_rho = rho_base;
    params.madmm_eta = 1 / (L + params.madmm_rho);

    % RADMM: Li-Ma-Srivastava 2022
    params.radmm_rho   = rho_base;
    params.radmm_eta   = 1 / (L + params.radmm_rho);
    params.radmm_gamma = 1e-8;

    % ARADMM: starts smaller, decays by iter^(1/3)
    params.aradmm_rho0  = max(eps, rho_base / 10);
    params.aradmm_eta0  = 1 / (L + params.aradmm_rho0);
    params.aradmm_beta0 = 50;
    params.aradmm_crho  = 1;
    params.aradmm_cbeta = 50;

    % OADMM: original code uses 10*mu which is WRONG for real data
    % (e.g. on news20, mu=0.01 -> rho=0.1, but L could be 1e4)
    params.oadmm_rho0  = rho_base;
    params.oadmm_sigma = 1.1;
    params.oadmm_delta = 1e-3;

    % AD-PMM (NS-ADMM): paper's G = L*I + H, so X-update step is implicitly 1/(L+rho)
    params.adpmm_rho      = rho_base;
    params.adpmm_ns_steps = 5;

    if opts.verbose
        fprintf('compute_step_sizes:\n');
        fprintf('  lambda_max(H)  = %.4e\n', L);
        fprintf('  rho_scale      = %.4g\n', opts.rho_scale);
        fprintf('  ADMM rho       = %.4e\n', rho_base);
        fprintf('  ManPG t        = %.4e\n', params.manpg_t);
        fprintf('  SOC/MADMM eta  = %.4e\n', params.soc_eta);
        fprintf('  AD-PMM rho     = %.4e\n', params.adpmm_rho);
    end
end
