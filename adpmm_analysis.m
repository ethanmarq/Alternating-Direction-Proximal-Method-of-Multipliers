% ADPMM_orth_compare.m
% Compare X-update orthogonalizations in ADPMM: SVD, QR, MGS, Polar,
% Newton-Schulz, Coupled Newton  (reproduces Fig. 1: Objective vs Time).
% Expects in workspace: X0, L, H, rho, mu, n, m, p, N, F  (optional: x_mode, dataset).
X0 = orth(randn(n, p));

methods = { ...
    'SVD',            @orth_svd; ...
    'QR',             @orth_qr; ...
    'MGS',            @orth_mgs; ...
    'Polar',          @orth_polar; ...
    'Newton-Schulz',  @orth_ns;
};

M = size(methods, 1);
algs = methods(:, 1)';
Fc = cell(1, M); Tc = cell(1, M); iters = zeros(1, M);

for i = 1:M
    fprintf('ADPMM [%s]...\n', algs{i});
    [Fc{i}, Tc{i}, iters(i), ores] = run_adpmm(methods{i,2}, X0, L, H, rho, mu, n, p, N, F);
    fprintf('  done in %.2fs at iter %d, F=%.4e, ||X''X-I||_F=%.1e\n', ...
        Tc{i}(end), iters(i), Fc{i}(end), ores);
    if ores > 1e-3, fprintf('  WARNING: %s DID NOT ORTHOGONALIZE\n', algs{i}); end
end

% ============================== PLOT ========================================
if ~exist('x_mode', 'var'), x_mode = 'time'; end
if ~exist('dataset', 'var'), dataset = 'data'; end

Fstar = min(cellfun(@min, Fc));
if strcmp(x_mode, 'time')
    Xc = Tc; xlbl = 'Time (s)'; xname = 'Time'; xtag = 'time';
else
    Xc = cellfun(@(f) 1:numel(f), Fc, 'UniformOutput', false);
    xlbl = 'Iteration'; xname = 'Iteration'; xtag = 'iter';
end

styles = {'-', '-o', ':', '-d', '-^', '-v'}; % SVD QR MGS Polar NS
figure('Visible', 'off'); hold on; grid on; set(gca, 'FontSize', 16);
for i = 1:M
    mi = unique(round(linspace(1, numel(Xc{i}), min(12, numel(Xc{i})))));
    plot(Xc{i}, Fc{i}, styles{mod(i-1, numel(styles)) + 1}, ...
        'LineWidth', 2, 'MarkerIndices', mi, 'MarkerSize', 6);
end
ylim([Fstar - 5, max(cellfun(@max, Fc)) + 5]);
xlabel(xlbl, 'FontSize', 20); ylabel('Objective', 'FontSize', 20);
title(sprintf('sPCA ADPMM Polar Orthogonalization Methods', ...
    xname, strrep(dataset, '_', '\_'), m, p, mu), 'FontSize', 18);
fname = sprintf('adpmm_orth_%s_n%d_p%d_mu%.2f_%s.png', dataset, m, p, mu, xtag);
legend(algs, 'Location', 'northeast');
exportgraphics(gcf, fname, 'Resolution', 300);
legend(algs, 'Location', 'northeast');
exportgraphics(gcf, fname, 'Resolution', 300);
fprintf('Saved: %s\n', fname);

% =========================== LOCAL FUNCTIONS ================================
function [Fh, Th, it, ores] = run_adpmm(orthfun, X0, L, H, rho, mu, n, p, N, F)
    X = X0; Z = X0; Y = zeros(n, p);
    Fh = zeros(1, N); Th = zeros(1, N); Fh(1) = F(X);
    t0 = tic;
    for k = 2:N
        B = H*X + rho*Z - Y; % X-update target
        X = orthfun(B); % project onto Stiefel
        V = X + Y/rho; % Z-update: soft-threshold
        Z = sign(V) .* max(abs(V) - mu/rho, 0);
        Y = Y + rho*(X - Z); % dual update
        Fh(k) = F(X); Th(k) = toc(t0);
        if abs(Fh(k) - Fh(k-1)) <= 1e-8, break; end
    end
    it = k; Fh = Fh(1:it); Th = Th(1:it);
    ores = norm(X'*X - eye(p), 'fro');
end

function W = orth_svd(B)
    [U, ~, V] = svd(B, 'econ'); W = U*V';
end

function W = orth_qr(B)
    [Q, R] = qr(B, 0); W = Q .* sign(diag(R)).';% sign fix -> canonical Q
end

function W = orth_mgs(B)
    [n, p] = size(B); W = zeros(n, p);
    for j = 1:p
        v = B(:, j);
        for i = 1:j-1, v = v - (W(:,i)'*v) * W(:,i); end
        W(:, j) = v / norm(v);
    end
end

function W = orth_polar(B)
    W = B / sqrtm(B'*B); % U = B (B'B)^{-1/2}
end

% function W = orth_ns(B)
%     c = 0.4; a = 1.5 + c; b = -0.5 - 2*c;
%     W = B / (norm(B, 'fro') + 1e-7);
%     tr = size(W, 1) > size(W, 2); if tr, W = W'; end
%     % for j = 1:15, A = W*W'; W = a*W + (b*A + c*A*A)*W; end
%     for j = 1:10
%         A = W * W';
%         B = b*A + c*A*A;
%         W = a*W + B*W;
%     end
%     if tr, W = W'; end
% end
function W = orth_ns(B)
    % c = 0.4; a = 1.5 + c; b = -0.5 - 2*c;
    % a = 3.4445; b = -4.775; c = 2.0315;
    % a = 3; b = -0.5*a; c = 1 - a + 0.5*a;
    % a = 1.876; b = -1.2510; c = 0.376;
    a = 2.438; b = -2.375; c = 0.938;
    tr = size(B,1) > size(B,2); if tr, B = B'; end
    W = B / (norm(B,2) + 1e-12);
    for j = 1:1
        A = W*W';
        % if norm(A - eye(size(A)),'fro') < 1e-12, break; end
        W = a*W + (b*A + c*A*A)*W;
    end
    if tr, W = W'; end
end
