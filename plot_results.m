function plot_results(result_files, out_dir)
% PLOT_RESULTS  Load one or more _results.mat files and replot locally.
%
%   plot_results(result_files)
%   plot_results(result_files, out_dir)
%
% Inputs:
%   result_files : char path, or cell array of paths, to <name>_results.mat
%                  files saved by run_spca_experiment.
%   out_dir      : (optional) directory to save .png/.pdf into. If omitted,
%                  figures are only displayed.
%
% Examples:
%   % Single run
%   plot_results('results/gisette_p50_mu0.01_results.mat');
%
%   % All runs in a directory
%   files = dir('results/*_results.mat');
%   paths = cellfun(@(d,n) fullfile(d,n), {files.folder}, {files.name}, ...
%                   'UniformOutput', false);
%   plot_results(paths, 'figures');

    if ischar(result_files) || isstring(result_files)
        result_files = {char(result_files)};
    end
    if nargin < 2, out_dir = ''; end
    if ~isempty(out_dir) && ~exist(out_dir, 'dir'); mkdir(out_dir); end

    algs = {'soc','madmm','radmm','aradmm','oadmm','manpg','adpmm'};
    styles = {'-.','-.','-.','-','-','-','-'};

    for f = 1:numel(result_files)
        R = load(result_files{f});
        nm = R.opts.name;
        fprintf('Plotting %s (n=%d p=%d mu=%g avg=%d)\n', ...
                nm, R.n, R.p, R.mu, R.avg);

        % ---- F vs iter ----
        fig1 = figure('Position', [100 100 800 500]);
        for a = 1:numel(algs)
            F = R.(['F_val_' algs{a} '_avg']);
            k = R.(['iter_' algs{a}]);
            if k < 2 || all(F == 0); continue; end
            semilogy(1:k, max(F(1:k), eps), styles{a}, 'LineWidth', 2);
            hold on;
        end
        xlabel('Iteration', 'Interpreter', 'latex', 'FontSize', 18);
        ylabel('$f(x) - f^*$', 'Interpreter', 'latex', 'FontSize', 18);
        legend(upper(algs), 'Location', 'best', 'FontSize', 14);
        title(sprintf('%s  (n=%d, p=%d, \\mu=%g, avg=%d)', ...
            strrep(nm,'_','\_'), R.n, R.p, R.mu, R.avg));
        grid on;

        % ---- F vs CPU time ----
        fig2 = figure('Position', [100 100 800 500]);
        for a = 1:numel(algs)
            F = R.(['F_val_' algs{a} '_avg']);
            T = R.(['cpu_time_' algs{a}]);
            k = R.(['iter_' algs{a}]);
            if k < 2 || all(F == 0); continue; end
            loglog(max(T(1:k), eps), max(F(1:k), eps), styles{a}, 'LineWidth', 2);
            hold on;
        end
        xlabel('CPU time (s)', 'Interpreter', 'latex', 'FontSize', 18);
        ylabel('$f(x) - f^*$', 'Interpreter', 'latex', 'FontSize', 18);
        legend(upper(algs), 'Location', 'best', 'FontSize', 14);
        title(sprintf('%s  (n=%d, p=%d, \\mu=%g, avg=%d)', ...
            strrep(nm,'_','\_'), R.n, R.p, R.mu, R.avg));
        grid on;

        if ~isempty(out_dir)
            saveas(fig1, fullfile(out_dir, [nm '_iter.png']));
            saveas(fig2, fullfile(out_dir, [nm '_time.png']));
            % Vector formats for the paper
            try
                exportgraphics(fig1, fullfile(out_dir, [nm '_iter.pdf']), 'ContentType','vector');
                exportgraphics(fig2, fullfile(out_dir, [nm '_time.pdf']), 'ContentType','vector');
            catch
                % exportgraphics is R2020a+; fall back to print
                print(fig1, '-dpdf', fullfile(out_dir, [nm '_iter.pdf']));
                print(fig2, '-dpdf', fullfile(out_dir, [nm '_time.pdf']));
            end
        end
    end
end
