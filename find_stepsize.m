function find_stepsize(configs, opts)
% FIND_STEPSIZE  Sweep ADMM penalty rho for multiple datasets and (p, mu) combos.
%
%   find_stepsize(configs)
%   find_stepsize(configs, opts)
%
% For each (dataset, p, mu) config and each rho_scale in opts.rho_grid, calls
% run_spca_experiment in sweep_mode and records the best rho_scale per
% algorithm. Writes the result to a .mat file consumed by run_spca_experiment.
%
% This file is the SINGLE driver for parameter sweeping. It calls the same
% solvers as run_spca_experiment (no duplicate code).
%
% Inputs:
%   configs : struct array, one entry per (dataset, p, mu) combo.
%             Required fields:
%               .data_path   path to a .mat file (from download_libsvm_to_mat.py)
%                            or '' for synthetic data
%               .name        short name, e.g. 'gisette'
%               .p           number of Stiefel columns
%               .mu          l1 weight
%             Optional:
%               .data_mode   'spca' or 'spca_n' (default 'spca')
%               .n_synth     synthetic data size (only if data_path is empty)
%
%   opts    : struct (optional). Fields:
%               .rho_grid     log-grid of rho scales (default [0.01 0.1 1 10 100])
%               .N            max iter per sweep run (default 200)
%               .avg          reps per (rho, alg) (default 1)
%               .algorithms   cell array (default {'adpmm','manpg','soc','madmm','radmm'})
%               .output_file  output .mat path (default 'best_rho.mat')
%               .seed         base RNG seed (default 0)
%               .verbose      (default true)
%
% Output file:
%   best_rho.mat contains:
%     best_rho.<key>.<alg> = rho_scale     (key like 'gisette_p50_mu0p01')
%     summary              : struct array, one row per (key, alg)
%     opts                 : the options used
%
% Example:
%   configs(1) = struct('data_path','libsvm_data/mat/gisette.mat', ...
%                       'name','gisette','p',50,'mu',0.01);
%   configs(2) = struct('data_path','libsvm_data/mat/gisette.mat', ...
%                       'name','gisette','p',50,'mu',1.00);
%   find_stepsize(configs);

    % -------- defaults --------
    if nargin < 2, opts = struct(); end
    opts = set_default(opts, 'rho_grid',    [0.01 0.1 1 10 100]);
    opts = set_default(opts, 'N',           200);
    opts = set_default(opts, 'avg',         1);
    opts = set_default(opts, 'algorithms',  {'adpmm','manpg','soc','madmm','radmm','aradmm','oadmm'});
    opts = set_default(opts, 'output_file', 'best_rho.mat');
    opts = set_default(opts, 'seed',        0);
    opts = set_default(opts, 'verbose',     true);

    addpath('misc');
    addpath('SSN_subproblem');

    % -------- defensive checks on configs --------
    n_cfg = numel(configs);
    if n_cfg == 0
        error('find_stepsize:empty_configs', 'configs struct array is empty');
    end
    for ci = 1:n_cfg
        for f = {'name','p','mu'}
            if ~isfield(configs(ci), f{1}) || isempty(configs(ci).(f{1}))
                error('find_stepsize:bad_config', ...
                      'configs(%d) missing required field "%s"', ci, f{1});
            end
        end
        if ~isfield(configs(ci), 'data_path'), configs(ci).data_path = ''; end
        if ~isfield(configs(ci), 'data_mode'), configs(ci).data_mode = 'spca'; end
        if ~isfield(configs(ci), 'n_synth'),   configs(ci).n_synth   = 400; end
    end

    % -------- output containers --------
    best_rho = struct();
    summary  = repmat(struct('key', '', 'dataset', '', 'p', 0, 'mu', 0, ...
                             'algorithm', '', 'rho_scale', NaN, ...
                             'final_F', NaN, 'stable', false), 0, 1);

    fprintf('================================================================\n');
    fprintf('find_stepsize: sweeping %d (dataset, p, mu) configs\n', n_cfg);
    fprintf('  rho_grid   = [%s]\n', num2str(opts.rho_grid));
    fprintf('  algorithms = {%s}\n', strjoin(opts.algorithms, ', '));
    fprintf('  N (iters)  = %d  (avg = %d)\n', opts.N, opts.avg);
    fprintf('  output     = %s\n', opts.output_file);
    fprintf('================================================================\n');

    for ci = 1:n_cfg
        cfg = configs(ci);
        key = make_key(cfg.name, cfg.p, cfg.mu);
        fprintf('\n[%d/%d] %s\n', ci, n_cfg, key);
        fprintf('  data_path = %s\n', ...
                tern(isempty(cfg.data_path), '<synthetic>', cfg.data_path));

        % -------- build H once (so we can call compute_step_sizes per rho_scale) --------
        H = build_H(cfg);
        fprintf('  H size: %d x %d\n', size(H,1), size(H,2));

        % -------- sweep rho_scale --------
        % For each rho_scale we run the FULL run_spca_experiment in sweep_mode.
        % The results we care about are the per-algorithm final F values.
        best_per_alg = init_best(opts.algorithms);

        for ri = 1:numel(opts.rho_grid)
            rho_scale = opts.rho_grid(ri);
            fprintf('\n  --- rho_scale = %g (%d/%d) ---\n', ...
                    rho_scale, ri, numel(opts.rho_grid));

            % Build params for this rho_scale (same for every algorithm)
            params = compute_step_sizes(H, cfg.mu, ...
                struct('rho_scale', rho_scale, 'verbose', opts.verbose));

            % Build the opts struct to pass into run_spca_experiment
            run_opts = struct();
            run_opts.data_path       = cfg.data_path;
            run_opts.data_mode       = cfg.data_mode;
            run_opts.n_synth         = cfg.n_synth;
            run_opts.p               = cfg.p;
            run_opts.mu              = cfg.mu;
            run_opts.N               = opts.N;
            run_opts.avg             = opts.avg;
            run_opts.seed            = opts.seed;
            run_opts.name            = sprintf('%s_sweep_rho%g', cfg.name, rho_scale);
            run_opts.output_dir      = './results/sweep';   % scratch dir
            run_opts.algorithms      = opts.algorithms;
            run_opts.save_plots      = false;
            run_opts.show_plots      = false;
            run_opts.verbose         = false;               % keep the inner output quiet
            run_opts.sweep_mode      = true;
            run_opts.params_override = params;

            try
                results = run_spca_experiment(run_opts);
            catch ME
                fprintf('    !! run_spca_experiment threw: %s\n', ME.message);
                continue;
            end

            % Extract final F per algorithm and decide stability
            for ai = 1:numel(opts.algorithms)
                alg = opts.algorithms{ai};
                if ~isfield(results, ['F_val_' alg '_avg'])
                    continue;
                end
                Fv = results.(['F_val_' alg '_avg']);
                last_iter = results.(['iter_' alg]);

                % "stable" = no NaN/Inf and bounded
                Fv_used = Fv(1:max(2, last_iter));
                Fv_used = Fv_used(Fv_used ~= 0);
                if isempty(Fv_used)
                    final_F = inf; stable = false;
                else
                    final_F = Fv_used(end);
                    stable  = isfinite(final_F) && abs(final_F) < 1e15;
                end

                fprintf('    %-7s : rho_scale=%-6g final_F=%.4e stable=%d\n', ...
                        alg, rho_scale, final_F, stable);

                if stable && final_F < best_per_alg.(alg).final_F
                    best_per_alg.(alg).final_F   = final_F;
                    best_per_alg.(alg).rho_scale = rho_scale;
                    best_per_alg.(alg).stable    = true;
                end
            end
        end

        % -------- record per-algorithm winners for this config --------
        entry = struct();
        for ai = 1:numel(opts.algorithms)
            alg = opts.algorithms{ai};
            b = best_per_alg.(alg);

            if ~b.stable
                error('find_stepsize:no_stable_run', ...
                    ['No stable rho found for algorithm "%s" on config "%s" ' ...
                     'across grid [%s]. Widen opts.rho_grid or check H.'], ...
                    alg, key, num2str(opts.rho_grid));
            end

            entry.(alg) = b.rho_scale;
            summary(end+1) = struct( ...
                'key', key, 'dataset', cfg.name, 'p', cfg.p, 'mu', cfg.mu, ...
                'algorithm', alg, 'rho_scale', b.rho_scale, ...
                'final_F', b.final_F, 'stable', true);  %#ok<AGROW>
        end
        best_rho.(key) = entry;

        % -------- incremental save --------
        save(opts.output_file, 'best_rho', 'summary', 'opts');
        fprintf('  saved progress to %s\n', opts.output_file);
    end

    fprintf('\n================================================================\n');
    fprintf('Done. Saved %d configs to %s\n', n_cfg, opts.output_file);
    fprintf('================================================================\n\n');
    print_summary(summary);
end


% ============================================================================
%                              HELPERS
% ============================================================================

function key = make_key(name, p, mu)
    mu_str = sprintf('%g', mu);
    mu_str = strrep(mu_str, '.', 'p');
    mu_str = strrep(mu_str, '-', 'm');
    key = sprintf('%s_p%d_mu%s', name, p, mu_str);
    key = regexprep(key, '[^A-Za-z0-9_]', '_');
    if ~isempty(key) && isstrprop(key(1), 'digit')
        key = ['x' key];
    end
end

function H = build_H(cfg)
    if isempty(cfg.data_path)
        rng(0);
        n = cfg.n_synth;
        A = randn(n, n); A = orth(A);
        S = diag(abs(randn(n, 1)));
        H = A * S * A.';
    else
        if ~exist(cfg.data_path, 'file')
            error('Data file not found: %s', cfg.data_path);
        end
        [H, ~, ~, ~] = load_libsvm_mat(cfg.data_path, cfg.data_mode);
    end
end

function best = init_best(algorithms)
    best = struct();
    for ai = 1:numel(algorithms)
        best.(algorithms{ai}) = struct( ...
            'final_F', inf, 'rho_scale', NaN, 'stable', false);
    end
end

function print_summary(summary)
    if isempty(summary), return; end
    fprintf('--- Summary table ---\n');
    fprintf('%-40s %-10s %10s %12s\n', 'config', 'algorithm', 'rho_scale', 'final_F');
    fprintf('%s\n', repmat('-', 1, 74));
    for i = 1:numel(summary)
        s = summary(i);
        fprintf('%-40s %-10s %10g %12.4e\n', ...
                s.key, s.algorithm, s.rho_scale, s.final_F);
    end
end

function opts = set_default(opts, key, val)
    if ~isfield(opts, key) || isempty(opts.(key))
        opts.(key) = val;
    end
end

function y = tern(cond, a, b)
    if cond, y = a; else, y = b; end
end
