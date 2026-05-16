function results = run_spca_experiment(opts)
% RUN_SPCA_EXPERIMENT  Headless driver for sPCA solver comparison.
%
%   results = run_spca_experiment(opts)
%
% Reads best rho_scale per algorithm from opts.best_rho_file (produced by
% find_stepsize.m), builds principled step sizes via compute_step_sizes,
% and runs every requested solver.
%
% opts struct fields (defaults shown):
%   .data_path     ''               path to a .mat file (load_libsvm_mat format);
%                                   '' for synthetic data
%   .data_mode     'spca'           'spca' or 'spca_n' for load_libsvm_mat
%   .n_synth       400              synthetic data dim (only if data_path empty)
%   .p             50               number of sparse components (Stiefel cols)
%   .mu            0.01             l1 sparsity weight
%   .N             500              max outer iterations per solver
%   .avg           1                repetitions with different seeds
%   .seed          0                base RNG seed
%   .output_dir    './results'      output directory
%   .name          'run'            output filename prefix
%   .algorithms    {'all'}          subset of {'soc','madmm','radmm','aradmm',
%                                              'oadmm','manpg','adpmm'} or {'all'}
%   .save_plots    true             save .png/.fig of convergence plots
%   .show_plots    false            display figures (false on cluster!)
%   .verbose       true             print progress
%   .best_rho_file 'best_rho.mat'   path to find_stepsize output
%   .sweep_mode    false            INTERNAL: when true, skip best_rho lookup
%                                   and use opts.params_override directly.
%                                   Used by find_stepsize.m for the sweep.
%   .params_override (struct)       INTERNAL: when sweep_mode is true, this
%                                   is used as params_per_alg.<algorithm> for
%                                   every algorithm.

    % ---------------------- option defaults ----------------------------------
    if nargin < 1, opts = struct(); end
    opts = set_default(opts, 'data_path',       '');
    opts = set_default(opts, 'data_mode',       'spca');
    opts = set_default(opts, 'n_synth',         400);
    opts = set_default(opts, 'p',               50);
    opts = set_default(opts, 'mu',              0.01);
    opts = set_default(opts, 'N',               500);
    opts = set_default(opts, 'avg',             1);
    opts = set_default(opts, 'seed',            0);
    opts = set_default(opts, 'output_dir',      './results');
    opts = set_default(opts, 'name',            'run');
    opts = set_default(opts, 'algorithms',      {'all'});
    opts = set_default(opts, 'save_plots',      true);
    opts = set_default(opts, 'show_plots',      false);
    opts = set_default(opts, 'verbose',         true);
    opts = set_default(opts, 'best_rho_file',   'best_rho.mat');
    opts = set_default(opts, 'sweep_mode',      false);
    opts = set_default(opts, 'params_override', []);

    addpath('misc');

    if ~exist(opts.output_dir, 'dir'); mkdir(opts.output_dir); end

    % Diary
    diary_file = fullfile(opts.output_dir, [opts.name '_log.txt']);
    if exist(diary_file, 'file'); delete(diary_file); end
    diary(diary_file);
    cleanup_diary = onCleanup(@() diary('off'));  %#ok<NASGU>

    if ~opts.show_plots
        set(0, 'DefaultFigureVisible', 'off');
    end

    fprintf('================================================================\n');
    fprintf('run_spca_experiment: %s\n', opts.name);
    fprintf('================================================================\n');
    try
        fprintf('host         : %s\n', char(java.net.InetAddress.getLocalHost.getHostName));
    catch
        fprintf('host         : (unknown)\n');
    end
    fprintf('start time   : %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf('data_path    : %s\n', tern(isempty(opts.data_path), '<synthetic>', opts.data_path));
    fprintf('p (comps)    : %d\n', opts.p);
    fprintf('mu           : %g\n', opts.mu);
    fprintf('N (max iter) : %d\n', opts.N);
    fprintf('avg (reps)   : %d\n', opts.avg);
    fprintf('seed         : %d\n', opts.seed);
    fprintf('output_dir   : %s\n', opts.output_dir);
    if iscell(opts.algorithms)
        fprintf('algorithms   : %s\n', strjoin(opts.algorithms, ', '));
    end
    if opts.sweep_mode
        fprintf('sweep_mode   : TRUE  (called from find_stepsize)\n');
    end
    fprintf('----------------------------------------------------------------\n');

    % ---------------------- data ---------------------------------------------
    rng(opts.seed);
    if isempty(opts.data_path)
        n = opts.n_synth;
        fprintf('Generating synthetic data, n=%d\n', n);
        A = randn(n, n); A = orth(A);
        S = diag(abs(randn(n, 1)));
        H = A * S * A.';
    else
        fprintf('Loading data from %s\n', opts.data_path);
        t0 = tic;
        [H, ~, ~, info] = load_libsvm_mat(opts.data_path, opts.data_mode);
        fprintf('  load + H construction took %.1fs\n', toc(t0));
        fprintf('  dataset name : %s\n', info.name);
        fprintf('  H size       : %d x %d\n', size(H,1), size(H,2));
        if issparse(H)
            fprintf('  H sparsity   : nnz=%d (%.2e density)\n', ...
                    nnz(H), nnz(H)/numel(H));
        end
        n = size(H, 1);
    end

    p   = opts.p;
    mu  = opts.mu;
    N   = opts.N;
    avg = opts.avg;
    algs = {'soc','madmm','radmm','aradmm','oadmm','manpg','adpmm'};
    run_all = any(strcmpi(opts.algorithms, 'all'));
    want = @(name) run_all || any(strcmpi(opts.algorithms, name));

    assert(n >= p, 'Need n >= p for St(n,p); got n=%d, p=%d', n, p);

    % ---------------------- per-algorithm step sizes -------------------------
    % Two modes:
    %   (1) sweep_mode = true  -> use opts.params_override for every algorithm
    %                             (find_stepsize sets this when sweeping rho)
    %   (2) sweep_mode = false -> read best_rho.mat, look up cfg_key,
    %                             call compute_step_sizes per algorithm.
    params_per_alg = struct();
    if opts.sweep_mode
        assert(~isempty(opts.params_override), ...
               'sweep_mode=true requires opts.params_override');
        for a = 1:numel(algs)
            params_per_alg.(algs{a}) = opts.params_override;
        end
        fprintf('Using sweep-mode params (rho_scale = %g)\n', ...
                opts.params_override.rho_scale);
    else
        if ~exist(opts.best_rho_file, 'file')
            error('run_spca_experiment:no_best_rho_file', ...
                  ['best_rho file not found: %s\n' ...
                   'Run find_stepsize.m first.'], opts.best_rho_file);
        end
        S = load(opts.best_rho_file);
        if ~isfield(S, 'best_rho')
            error('run_spca_experiment:bad_best_rho_file', ...
                  '%s does not contain a "best_rho" struct', opts.best_rho_file);
        end
        cfg_key = make_cfg_key(opts.name, opts.p, opts.mu);
        if ~isfield(S.best_rho, cfg_key)
            error('run_spca_experiment:missing_cfg', ...
                  ['best_rho.mat has no entry for "%s".\nAvailable keys:\n  %s\n' ...
                   'Add this (dataset, p, mu) combo to find_stepsize and rerun.'], ...
                  cfg_key, strjoin(fieldnames(S.best_rho), ', '));
        end
        rho_entry = S.best_rho.(cfg_key);

        fprintf('Step sizes per algorithm (from %s, key=%s):\n', ...
                opts.best_rho_file, cfg_key);
        for a = 1:numel(algs)
            alg_name = algs{a};
            if ~want(alg_name); continue; end
            if ~isfield(rho_entry, alg_name)
                error('run_spca_experiment:missing_alg', ...
                      ['best_rho.mat entry "%s" has no rho_scale for "%s". ' ...
                       'Rerun find_stepsize with this algorithm included.'], ...
                      cfg_key, alg_name);
            end
            rs = rho_entry.(alg_name);
            params_per_alg.(alg_name) = compute_step_sizes(H, mu, ...
                struct('rho_scale', rs, 'verbose', false));
            fprintf('  %-7s rho_scale=%-6g  rho=%.3e  eta=%.3e\n', ...
                    alg_name, rs, get_rho_for(params_per_alg.(alg_name), alg_name), ...
                    get_eta_for(params_per_alg.(alg_name), alg_name));
        end
    end

    % ---------------------- problem functions --------------------------------
    f   = @(X) -0.5 * trace(X.' * (H * X));
    g   = @(Y) mu * sum(sum(abs(Y)));
    g_gamma = @(Z, gamma) mu * (g(wthresh(Z,'s',gamma)) ...
                + 1/(2*gamma) * norm(wthresh(Z,'s',gamma) - Z, 'fro')^2);
    F   = @(X) f(X) + g(X);
    L_M = @(X, Z, Lambda, gamma, rho) f(X) + mu*g_gamma(Z, gamma) ...
                + trace(Lambda.' * (X - Z)) + rho/2 * norm(X - Z, 'fro')^2;

    % ---------------------- ManPG bookkeeping --------------------------------
    if want('manpg')
        L_lip      = params_per_alg.manpg.lambda_max;
        t_step     = params_per_alg.manpg.manpg_t;
        inner_iter = 100;
        prox_fun   = @(b, l, r) proximal_l1(b, l*mu, r);
        t_min      = 1e-4;
        Dn         = sparse(DuplicationM(p));
        pDn        = (Dn' * Dn) \ Dn';
        nu         = mu;
        alpha      = 1;
        tol        = 1e-8 * n * p;
        fprintf('ManPG: Lipschitz L = %g, t = %g\n', L_lip, t_step);
    end

    % ---------------------- accumulators -------------------------------------
    F_avg    = struct();
    CPU_acc  = struct();
    iter_min = struct();
    for a = 1:numel(algs)
        F_avg.(algs{a})    = zeros(1, N);
        CPU_acc.(algs{a})  = zeros(avg, N);
        iter_min.(algs{a}) = N;
    end
    avg_min_among_all = 0;

    % ---------------------- repetition loop ----------------------------------
    for k = 1:avg
        rng(opts.seed + k - 1);
        if opts.verbose, fprintf('\n[rep %d/%d]\n', k, avg); end

        X0 = randn(n, p);
        X0 = orth(X0);

        Fv = struct();
        for a = 1:numel(algs); Fv.(algs{a}) = zeros(1, N); end
        it = struct();

        % -- SOC ----------------------------------------------------------
        if want('soc')
            fprintf('  >>> SOC starting (eta=%.3e rho=%.3e)\n', ...
                    params_per_alg.soc.soc_eta, params_per_alg.soc.soc_rho);
            flush_diary(); t_solver = tic;
            [Fv.soc, CPU_acc.soc(k,:), it.soc] = ...
                solver_soc(X0, H, F, mu, N, params_per_alg.soc);
            iter_min.soc = min(iter_min.soc, it.soc);
            fprintf('  <<< SOC finished in %.1fs (%d iters)\n', toc(t_solver), it.soc);
            log_done(opts.verbose, 'SOC', it.soc, Fv.soc);
            flush_diary();
        end

        % MADMM --------------------------------------------------------
        if want('madmm')
            fprintf('  >>> MADMM starting (eta=%.3e rho=%.3e)\n', ...
                    params_per_alg.madmm.madmm_eta, params_per_alg.madmm.madmm_rho);
            flush_diary(); t_solver = tic;
            [Fv.madmm, CPU_acc.madmm(k,:), it.madmm] = ...
                solver_madmm(X0, H, F, mu, N, params_per_alg.madmm);
            iter_min.madmm = min(iter_min.madmm, it.madmm);
            fprintf('  <<< MADMM finished in %.1fs (%d iters)\n', toc(t_solver), it.madmm);
            log_done(opts.verbose, 'MADMM', it.madmm, Fv.madmm);
            flush_diary();
        end

        % -- RADMM --------------------------------------------------------
        if want('radmm')
            fprintf('  >>> RADMM starting (eta=%.3e rho=%.3e)\n', ...
                    params_per_alg.radmm.radmm_eta, params_per_alg.radmm.radmm_rho);
            flush_diary(); t_solver = tic;
            [Fv.radmm, CPU_acc.radmm(k,:), it.radmm] = ...
                solver_radmm(X0, H, F, mu, N, params_per_alg.radmm);
            iter_min.radmm = min(iter_min.radmm, it.radmm);
            fprintf('  <<< RADMM finished in %.1fs (%d iters)\n', toc(t_solver), it.radmm);
            log_done(opts.verbose, 'RADMM', it.radmm, Fv.radmm);
            flush_diary();
        end
        % -- ARADMM -------------------------------------------------------
        if want('aradmm')
            fprintf('  >>> ARADMM starting (eta0=%.3e rho0=%.3e)\n', ...
                    params_per_alg.aradmm.aradmm_eta0, params_per_alg.aradmm.aradmm_rho0);
            flush_diary(); t_solver = tic;
            [Fv.aradmm, CPU_acc.aradmm(k,:), it.aradmm] = ...
                solver_aradmm(X0, H, F, mu, N, params_per_alg.aradmm);
            iter_min.aradmm = min(iter_min.aradmm, it.aradmm);
            fprintf('  <<< ARADMM finished in %.1fs (%d iters)\n', toc(t_solver), it.aradmm);
            log_done(opts.verbose, 'ARADMM', it.aradmm, Fv.aradmm);
            flush_diary();
        end

        % -- OADMM (depends on aradmm for stop condition) ------------------
        if want('oadmm')
            fprintf('  >>> OADMM starting (rho0=%.3e)\n', ...
                    params_per_alg.oadmm.oadmm_rho0);
            flush_diary(); t_solver = tic;
            [Fv.oadmm, CPU_acc.oadmm(k,:), it.oadmm] = ...
                solver_oadmm(X0, H, F, L_M, mu, N, Fv.aradmm, params_per_alg.oadmm);
            iter_min.oadmm = min(iter_min.oadmm, it.oadmm);
            fprintf('  <<< OADMM finished in %.1fs (%d iters)\n', toc(t_solver), it.oadmm);
            log_done(opts.verbose, 'OADMM', it.oadmm, Fv.oadmm);
            flush_diary();
        end

        % -- ManPG --------------------------------------------------------
        if want('manpg')
            fprintf('  >>> ManPG starting (t=%.3e L=%.3e)\n', t_step, L_lip);
            flush_diary(); t_solver = tic;
            [Fv.manpg, CPU_acc.manpg(k,:), it.manpg] = ...
                solver_manpg(X0, H, F, n, p, t_step, t_min, nu, alpha, ...
                             inner_iter, prox_fun, Dn, pDn, tol, N);
            iter_min.manpg = min(iter_min.manpg, it.manpg);
            fprintf('  <<< ManPG finished in %.1fs (%d iters)\n', toc(t_solver), it.manpg);
            log_done(opts.verbose, 'ManPG', it.manpg, Fv.manpg);
            flush_diary();
        end

        % -- ADPMM --------------------------------------------------------
        if want('adpmm')
            fprintf('  >>> ADPMM starting (rho=%.3e L=%.3e)\n', ...
                    params_per_alg.adpmm.adpmm_rho, params_per_alg.adpmm.lambda_max);
            flush_diary(); t_solver = tic;
            [Fv.adpmm, CPU_acc.adpmm(k,:), it.adpmm] = ...
                solver_adpmm(X0, H, F, mu, N, params_per_alg.adpmm);
            iter_min.adpmm = min(iter_min.adpmm, it.adpmm);
            fprintf('  <<< ADPMM finished in %.1fs (%d iters)\n', toc(t_solver), it.adpmm);
            log_done(opts.verbose, 'ADPMM', it.adpmm, Fv.adpmm);
            flush_diary();
        end

        %         shift by best objective seen this rep ----------------------
        active = {};
        for a = 1:numel(algs)
            if isfield(Fv, algs{a}) && any(Fv.(algs{a}) ~= 0)
                active{end+1} = algs{a}; %#ok<AGROW>
            end
        end
        min_this = inf;
        for a = 1:numel(active)
            min_this = min(min_this, min(Fv.(active{a})(Fv.(active{a}) ~= 0)));
        end
        for a = 1:numel(active)
            v = Fv.(active{a});
            v(v ~= 0) = v(v ~= 0) - min_this;
            F_avg.(active{a}) = F_avg.(active{a}) + v;
        end
        avg_min_among_all = avg_min_among_all + min_this;
    end

    % ---------------------- average ------------------------------------------
    for a = 1:numel(algs)
        F_avg.(algs{a}) = F_avg.(algs{a}) / avg;
    end
    cpu_mean = struct();
    for a = 1:numel(algs)
        cpu_mean.(algs{a}) = sum(CPU_acc.(algs{a}), 1) / avg;
    end
    avg_min_among_all = avg_min_among_all / avg;

    % ---------------------- build results struct -----------------------------
    results = struct();
    results.opts = opts;
    results.n = n;
    results.p = p;
    results.mu = mu;
    results.N = N;
    results.avg = avg;
    results.avg_min_among_all = avg_min_among_all;
    for a = 1:numel(algs)
        results.(['F_val_' algs{a} '_avg']) = F_avg.(algs{a});
        results.(['cpu_time_'   algs{a}])   = cpu_mean.(algs{a});
        results.(['iter_'       algs{a}])   = iter_min.(algs{a});
    end

    % ---------------------- save ---------------------------------------------
    if ~opts.sweep_mode
        out_mat = fullfile(opts.output_dir, [opts.name '_results.mat']);
        save(out_mat, '-struct', 'results', '-v7.3');
        fprintf('\nSaved results: %s\n', out_mat);

        if opts.save_plots
            plot_one(results, opts, algs, 'iter');
            plot_one(results, opts, algs, 'time');
        end
    end

    fprintf('end time     : %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
end


% ============================================================================
%                               SOLVERS
% ============================================================================

function [Fv, cpu, last_iter] = solver_soc(X0, H, F, mu, N, params)
    X = X0; Y = X0; Lambda = zeros(size(X));
    eta = params.soc_eta; rho = params.soc_rho;
    Fv = zeros(1, N); cpu = zeros(1, N); cpu(1) = eps;
    last_iter = N;
    for iter = 2:N
        t0 = tic;
        for i = 1:100
            grad_f = -H*X + rho*(X - Y + Lambda);
            grad_map = (X - wthresh(X - eta*grad_f, 's', mu*eta)) / eta;
            if norm(grad_map, 'fro') < 1e-8, break; end
            X = X - eta * grad_map;
        end
        % Project (X + Lambda) onto Stiefel
        [U, ~, V] = svd(X + Lambda, 'econ');
        Y = U * V.';
        Lambda = Lambda + (X - Y);
        et = toc(t0);
        Fv(iter) = F(Y);
        if abs(Fv(iter) - Fv(iter-1)) <= 1e-8, last_iter = iter; break; end
        cpu(iter) = cpu(iter) + et;
        if iter <= N - 1, cpu(iter+1) = cpu(iter); end
        last_iter = iter;
        if mod(iter, 10) == 0
            fprintf('    SOC iter %4d/%d in %6.2fs (F=%.4e, %.3fs/iter)\n', ...
                    iter, N, cpu(iter), Fv(iter), cpu(iter)/iter);
        end
    end
end

function [Fv, cpu, last_iter] = solver_madmm(X0, H, F, mu, N, params)
    X = X0; Y = X0; Lambda = zeros(size(X));
    eta = params.madmm_eta; rho = params.madmm_rho;
    Fv = zeros(1, N); cpu = zeros(1, N); cpu(1) = eps;
    last_iter = N;
    for iter = 2:N
        t0 = tic;
        for i = 1:100
            gx = -H*X + rho*(X - Y + Lambda);
            rgx = proj(X, gx);
            if norm(rgx, 'fro') < 1e-8, break; end
            X = retr(X, -eta*rgx);
        end
        Y = wthresh(X + Lambda, 's', mu/rho);
        Lambda = Lambda + (X - Y);
        et = toc(t0);
        Fv(iter) = F(X);
        if abs(Fv(iter) - Fv(iter-1)) <= 1e-8, last_iter = iter; break; end
        cpu(iter) = cpu(iter) + et;
        if iter <= N - 1, cpu(iter+1) = cpu(iter); end
        last_iter = iter;
        if mod(iter, 10) == 0
            fprintf('    SOC iter %4d/%d in %6.2fs (F=%.4e, %.3fs/iter)\n', ...
                    iter, N, cpu(iter), Fv(iter), cpu(iter)/iter);
        end
    end
end

function [Fv, cpu, last_iter] = solver_radmm(X0, H, F, mu, N, params)
    X = X0; Z = X0; Lambda = zeros(size(X));
    eta = params.radmm_eta; gamma = params.radmm_gamma; rho = params.radmm_rho;
    Fv = zeros(1, N); cpu = zeros(1, N); cpu(1) = eps;
    last_iter = N;
    for iter = 2:N
        t0 = tic;
        gx = -H*X + Lambda + rho*(X - Z);
        rgx = proj(X, gx);
        X = retr(X, -eta*rgx);
        Y = wthresh(X + Lambda/rho, 's', mu*(1 + rho*gamma)/rho);
        Z = (Y/gamma + Lambda + rho*X) / (1/gamma + rho);
        Lambda = Lambda + rho*(X - Z);
        et = toc(t0);
        Fv(iter) = F(X);
        if abs(Fv(iter) - Fv(iter-1)) <= 1e-8, last_iter = iter; break; end
        cpu(iter) = cpu(iter) + et;
        if iter <= N - 1, cpu(iter+1) = cpu(iter); end
        last_iter = iter;
        if mod(iter, 10) == 0
            fprintf('    Radmm iter %4d/%d in %6.2fs (F=%.4e, %.3fs/iter)\n', ...
                    iter, N, cpu(iter), Fv(iter), cpu(iter)/iter);
        end
    end
end

function [Fv, cpu, last_iter] = solver_aradmm(X0, H, F, mu, N, params)
    X = X0; Z = X0; Lambda = zeros(size(X));
    etak  = params.aradmm_eta0;  rhok  = params.aradmm_rho0;
    beta0 = params.aradmm_beta0; crho  = params.aradmm_crho;
    cbeta = params.aradmm_cbeta;
    newcv = norm(Z - X, 'fro'); inicv = newcv;
    betak = beta0;
    Fv = zeros(1, N); cpu = zeros(1, N); cpu(1) = eps;
    last_iter = N;
    for iter = 2:N
        t0 = tic;
        oldcv = newcv;
        Z = wthresh(X + Lambda/rhok, 's', mu/rhok);
        gx = -H*X + Lambda + rhok*(X - Z);
        rgx = proj(X, gx);
        X = retr(X, -etak*rgx/(iter^(1/3)));
        newcv = norm(Z - X, 'fro');
        if newcv > oldcv
            betak = min(beta0*(inicv*(log(2))^2)/(newcv*(iter+1)^2*log(iter+2)), ...
                        cbeta/(iter^(1/3)*(log(iter+1)^2)));
            rhok = crho * rhok * (iter^(1/3));
        end
        Lambda = Lambda + betak*(X - Z);
        et = toc(t0);
        Fv(iter) = F(X);
        if abs(Fv(iter) - Fv(iter-1)) <= 1e-8, last_iter = iter; break; end
        cpu(iter) = cpu(iter) + et;
        if iter <= N - 1, cpu(iter+1) = cpu(iter); end
        last_iter = iter;
        if mod(iter, 10) == 0
            fprintf('    Aradmm iter %4d/%d in %6.2fs (F=%.4e, %.3fs/iter)\n', ...
                    iter, N, cpu(iter), Fv(iter), cpu(iter)/iter);
        end
    end
end

function [Fv, cpu, last_iter] = solver_oadmm(X0, H, F, L_M, mu, N, Fv_aradmm, params)
    X = X0; Z = X0; Lambda = zeros(size(X));
    orho = params.oadmm_rho0; sigma = params.oadmm_sigma; delta = params.oadmm_delta;
    if isempty(Fv_aradmm) || all(Fv_aradmm == 0)
        aradmm_min = inf;
    else
        aradmm_min = min(Fv_aradmm(Fv_aradmm ~= 0));
        if isempty(aradmm_min), aradmm_min = inf; end
    end
    Fv = zeros(1, N); cpu = zeros(1, N); cpu(1) = eps;
    last_iter = N;
    for iter = 2:N
        t0 = tic;
        ogamma = 4 / ((2 - sigma) * orho);
        Xbar = X;
        oeta = 1 / orho;
        gx = -H*Xbar + Lambda + orho*(Xbar - Z);
        rgx = proj(Xbar, gx);
        X = retr(Xbar, -oeta*rgx);
        Gra_norm = norm(rgx, 'fro');
        ls_cut = 1;
        while L_M(X, Z, Lambda, ogamma, orho) > ...
              L_M(Xbar, Z, Lambda, ogamma, orho) - delta*oeta*Gra_norm^2 ...
              && ls_cut <= 10
            oeta = 0.5 * oeta;
            X = retr(Xbar, -oeta*rgx);
            ls_cut = ls_cut + 1;
        end
        Y = wthresh(X + Lambda/orho, 's', mu*(1 + orho*ogamma)/orho);
        Z = (Y/ogamma + Lambda + orho*X) / (1/ogamma + orho);
        Lambda = Lambda + sigma*orho*(X - Z);
        orho = orho * (1 + 0.1*iter^(1/3));
        et = toc(t0);
        Fv(iter) = F(X);
        if abs(Fv(iter) - Fv(iter-1)) <= 1e-8 && Fv(iter) <= aradmm_min
            last_iter = iter; break;
        end
        cpu(iter) = cpu(iter) + et;
        if iter <= N - 1, cpu(iter+1) = cpu(iter); end
        last_iter = iter;
        if mod(iter, 10) == 0
            fprintf('    Oadmm iter %4d/%d in %6.2fs (F=%.4e, %.3fs/iter)\n', ...
                    iter, N, cpu(iter), Fv(iter), cpu(iter)/iter);
        end
    end
end

function [Fv, cpu, last_iter] = solver_manpg(X0, H, F, n, p, t, t_min, ...
                                              nu, alpha, inner_iter, ...
                                              prox_fun, Dn, pDn, tol, N)
    U = X0;
    num_inexact = 0; inner_flag = 0;
    Lam_x = zeros(p);
    Fv = zeros(1, N); cpu = zeros(1, N); cpu(1) = eps;
    last_iter = N;
    for iter = 2:N
        t0 = tic;
        neg_pg = H * U;
        if alpha < t_min || num_inexact > 10
            inner_tol = max(5e-16, min(1e-14, 1e-5 * tol * t^2));
        else
            inner_tol = max(1e-13, min(1e-11, 1e-3 * tol * t^2));
        end
        [PU, ~, Lam_x, ~, in_flag] = Semi_newton_matrix( ...
            n, p, U, t, U + t*neg_pg, nu*t, inner_tol, prox_fun, ...
            inner_iter, Lam_x, Dn, pDn);
        if in_flag == 1, inner_flag = inner_flag + 1; end
        V = PU - U;
        PU = U + alpha * V;
        [T, SIG, S] = svd(PU' * PU);
        SIG = diag(SIG);
        U = PU * (T * diag(sqrt(1./SIG)) * S');
        et = toc(t0);
        Fv(iter) = F(U);
        if abs(Fv(iter) - Fv(iter-1)) <= 1e-8, last_iter = iter; break; end
        cpu(iter) = cpu(iter) + et;
        if iter <= N - 1, cpu(iter+1) = cpu(iter); end
        last_iter = iter;
        if mod(iter, 10) == 0
            fprintf('    ManPg iter %4d/%d in %6.2fs (F=%.4e, %.3fs/iter)\n', ...
                    iter, N, cpu(iter), Fv(iter), cpu(iter)/iter);
        end
    end
end

function [Fv, cpu, last_iter] = solver_adpmm(X0, H, F, mu, N, params)
    X = X0; Z = X0; Y_dual = zeros(size(X));
    rho_adpmm = params.adpmm_rho;
    L_lip     = params.lambda_max;   % for paper's G = lambda_max*I + H
    Fv = zeros(1, N); cpu = zeros(1, N); cpu(1) = eps;
    last_iter = N;
    for iter = 2:N
        t0 = tic;
        % Paper's full G*X + rho*Z - Y formulation
        X = newtonschulz5(L_lip*X + H*X + rho_adpmm*Z - Y_dual);
        Vk = X + Y_dual / rho_adpmm;
        Z = sign(Vk) .* max(abs(Vk) - mu/rho_adpmm, 0);
        Y_dual = Y_dual + rho_adpmm * (X - Z);
        et = toc(t0);
        Fv(iter) = F(X);
        if abs(Fv(iter) - Fv(iter-1)) <= 1e-8, last_iter = iter; break; end
        cpu(iter) = cpu(iter) + et;
        if iter <= N - 1, cpu(iter+1) = cpu(iter); end
        last_iter = iter;
        if mod(iter, 10) == 0
            fprintf('    Adpmm iter %4d/%d in %6.2fs (F=%.4e, %.3fs/iter)\n', ...
                    iter, N, cpu(iter), Fv(iter), cpu(iter)/iter);
        end
    end
end


% ============================================================================
%                               HELPERS
% ============================================================================

function W = newtonschulz5(B, steps)
    if nargin < 2, steps = 5; end
    if any(~isfinite(B(:))), W = B; return; end
    nrmB = norm(B);                       % spectral, not Frobenius
    if nrmB < eps, W = B; return; end
    W = B / (nrmB * 1.05);                % scale so sigma_max < 1 (safety margin)
    [~, k] = size(B);
    I = eye(k);
    for i = 1:steps
        WtW = W' * W;
        W = 0.5 * W * (3*I - WtW);
        if any(~isfinite(W(:))), return; end
    end
end

function opts = set_default(opts, key, val)
    if ~isfield(opts, key) || (isnumeric(opts.(key)) && isempty(opts.(key)) ...
                               && ~isstruct(val) && ~iscell(val))
        opts.(key) = val;
    elseif ~isfield(opts, key)
        opts.(key) = val;
    end
end

function y = tern(cond, a, b)
    if cond, y = a; else, y = b; end
end

function flush_diary()
    d = get(0, 'Diary');
    if strcmp(d, 'on')
        df = get(0, 'DiaryFile');
        diary off; diary(df);
    end
end

function log_done(verbose, name, k, Fv)
    if ~verbose, return; end
    last_F = Fv(find(Fv ~= 0, 1, 'last'));
    if isempty(last_F), last_F = NaN; end
    fprintf('  %-7s done at iter %4d, last F = %.6e\n', name, k, last_F);
end

function r = get_rho_for(params, alg_name)
% Return the rho value applicable to this algorithm (for logging).
    switch alg_name
        case 'soc',    r = params.soc_rho;
        case 'madmm',  r = params.madmm_rho;
        case 'radmm',  r = params.radmm_rho;
        case 'aradmm', r = params.aradmm_rho0;
        case 'oadmm',  r = params.oadmm_rho0;
        case 'adpmm',  r = params.adpmm_rho;
        case 'manpg',  r = NaN;   % no rho (no ADMM penalty)
        otherwise,     r = NaN;
    end
end

function e = get_eta_for(params, alg_name)
    switch alg_name
        case 'soc',    e = params.soc_eta;
        case 'madmm',  e = params.madmm_eta;
        case 'radmm',  e = params.radmm_eta;
        case 'aradmm', e = params.aradmm_eta0;
        case 'oadmm',  e = 1/params.oadmm_rho0;
        case 'adpmm',  e = NaN;   % NS has no step size
        case 'manpg',  e = params.manpg_t;
        otherwise,     e = NaN;
    end
end

function plot_one(results, opts, algs, kind)
    fig = figure('Visible', tern(opts.show_plots, 'on', 'off'));
    clf;
    styles = {'-.','-.','-.','-','-','-','-'};
    for a = 1:numel(algs)
        nm = algs{a};
        F = results.(['F_val_' nm '_avg']);
        k = results.(['iter_' nm]);
        if k < 2 || all(F == 0); continue; end
        rng_idx = 1:k;
        if strcmp(kind, 'iter')
            semilogy(rng_idx, max(F(rng_idx), eps), styles{a}, 'LineWidth', 2);
        else
            T = results.(['cpu_time_' nm]);
            loglog(max(T(rng_idx), eps), max(F(rng_idx), eps), styles{a}, 'LineWidth', 2);
        end
        hold on;
    end
    if strcmp(kind, 'iter')
        xlabel('Iteration','interpreter','latex','FontSize',18);
    else
        xlabel('CPU time (s)','interpreter','latex','FontSize',18);
    end
    ylabel('$f(x)-f^*$','interpreter','latex','FontSize',18);
    legend(upper(algs), 'Location', 'best', 'FontSize', 14);
    title(sprintf('%s  (n=%d, p=%d, \\mu=%g, avg=%d)', ...
        strrep(opts.name,'_','\_'), results.n, results.p, results.mu, results.avg));
    base = fullfile(opts.output_dir, [opts.name '_' kind]);
    saveas(fig, [base '.png']);
    savefig(fig, [base '.fig']);
    if ~opts.show_plots, close(fig); end
    fprintf('Saved plot: %s.{png,fig}\n', base);
end

function key = make_cfg_key(name, p, mu)
% Must match the keying in find_stepsize.m
    mu_str = sprintf('%g', mu);
    mu_str = strrep(mu_str, '.', 'p');
    mu_str = strrep(mu_str, '-', 'm');
    key = sprintf('%s_p%d_mu%s', name, p, mu_str);
    key = regexprep(key, '[^A-Za-z0-9_]', '_');
    if ~isempty(key) && isstrprop(key(1), 'digit')
        key = ['x' key];
    end
end
