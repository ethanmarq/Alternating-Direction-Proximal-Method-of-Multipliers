% solvers.m
% Minimal sPCA solver comparison on a single dataset.
% Solvers ported from the trusted-author repo (compare_sPCA.m).
% Expects H, X0, n, p, mu, N, rho, eta, t, L, F, dataset, time_limit to be in workspace.
% time_limit (seconds) caps wall-clock time per algorithm; N is just a safety cap.

addpath('misc');

X0 = orth(randn(n, p));

% ============================== ADPMM =======================================
% Paper's AD-PMM (NS-ADMM): G = L*I + H, X-update via Newton-Schulz.
fprintf('ADPMM...\n');
X = X0; Z = X0; Y = zeros(n, p);
F_adpmm = zeros(1, N); F_adpmm(1) = F(X);
T_adpmm = zeros(1, N); T_adpmm(1) = 0;
tic
for k = 2:N
    % X-update: orthogonalize L*X + H*X + rho*Z - Y via Newton-Schulz
    B = H*X + rho*Z - Y;
    nrmB = norm(B, 'fro');
    % if nrmB < eps
    %     W = B;
    % else
    %     W = B / (nrmB * 1.05);
    %     Ip = eye(p);
    %     for j = 1:10
    %         W = 0.5 * W * (3*Ip - W'*W);
    %     end
    % end
    c = 0.4; a = 1.5 + c; b = -0.5 - 2*c;
    W = B;
    W = B / (nrmB + 1e-7);
    transposed = size(W, 1) > size(W, 2);
    if transposed, W = W'; end
    for j = 1:5
        A = W * W';
        B = b*A + c*A*A;
        W = a*W + B*W;
    end
    if transposed, W = W'; end
    X = W;
    % Z-update: soft-thresholding
    V = X + Y/rho;
    Z = sign(V) .* max(abs(V) - mu/rho, 0);
    % Dual update
    Y = Y + rho*(X - Z);
    F_adpmm(k) = F(X);
    T_adpmm(k) = toc;
    if T_adpmm(k) >= time_limit, break; end
end
iter_adpmm = k;
fprintf('  done in %.1fs at iter %d, F=%.4e\n', T_adpmm(iter_adpmm), iter_adpmm, F_adpmm(iter_adpmm));

% ============================== ADPMM-SVD =======================================
fprintf('ADPMM-SVD...\n');
X = X0; Z = X0; Y = zeros(n, p);
F_adpmm_svd = zeros(1, N); F_adpmm_svd(1) = F(X);
T_adpmm_svd = zeros(1, N); T_adpmm_svd(1) = 0;
tic
for k = 2:N
    % X-update: orthogonalize L*X + H*X + rho*Z - Y via SVD
    B = H*X + rho*Z - Y;
    [U, ~, V] = svd(B, 'econ');
    X = U*V';
    % Z-update: soft-thresholding
    V = X + Y/rho;
    Z = sign(V) .* max(abs(V) - mu/rho, 0);
    % Dual update
    Y = Y + rho*(X - Z);
    F_adpmm_svd(k) = F(X);
    T_adpmm_svd(k) = toc;
    if T_adpmm_svd(k) >= time_limit, break; end
end
iter_adpmm_svd = k;
fprintf('  done in %.1fs at iter %d, F=%.4e\n', T_adpmm_svd(iter_adpmm_svd), iter_adpmm_svd, F_adpmm_svd(iter_adpmm_svd));

% ============================== ManPG ================================
% Repo function: f(X) = -tr(X'AX), so pass A = H/2 with type=1 to match our
% F(X) = -0.5*tr(X'HX). Effective prox threshold inside SSN is mu*t (correct).
fprintf('ManPG...\n');
option_manpg = struct( ...
    'n',          n, ...
    'r',          p, ...
    'mu',         mu, ...
    'maxiter',    N, ...
    'tol',        1e-8*n*p, ...
    'inner_iter', 100, ...
    'type',       1, ...
    'phi_init',   X0, ...
    'time_limit', time_limit);

[~, ~, ~, time_manpg, iter_manpg, flag_manpg, ~, ~, F_manpg] = ...
    manpg_orth_sparse(0.5*H, option_manpg);

T_manpg = time_manpg(1:iter_manpg);
total_manpg = T_manpg(end);

fprintf('  done in %.1fs at iter %d, F=%.4e (flag=%d)\n', ...
    total_manpg, iter_manpg, F_manpg(iter_manpg), flag_manpg);

% ============================== ManPG-Ada ============================
fprintf('ManPG-Ada...\n');
option_ada = struct( ...
    'n',          n, ...
    'r',          p, ...
    'mu',         mu, ...
    'maxiter',    N, ...
    'tol',        1e-8*n*p, ...
    'inner_iter', 10, ...
    'type',       1, ...
    'phi_init',   X0, ...
    'F_manpg',    -Inf, ...
    'time_limit', time_limit);

[~, ~, ~, time_ada, iter_manpg_ada, flag_ada, ~, ~, F_manpg_ada] = ...
    manpg_orth_sparse_adap(0.5*H, option_ada);

  T_manpg_ada = time_ada(1:iter_manpg_ada);
  total_ada = T_manpg_ada(end);

fprintf('  done in %.1fs at iter %d, F=%.4e (flag=%d)\n', ...
    total_ada, iter_manpg_ada, F_manpg_ada(iter_manpg_ada), flag_ada);


% ============================== RADMM =======================================
fprintf('RADMM...\n');
X = X0; Z = X0; Lambda = zeros(n, p);
gamma = 1e-8;
F_radmm = zeros(1, N); F_radmm(1) = F(X);
T_radmm = zeros(1, N); T_radmm(1) = 0;
tic
for k = 2:N
    % X step: one Riemannian gradient step
    gx = -H*X + Lambda + rho*(X - Z);
    rgx = proj(X, gx);
    X = retr(X, -eta*rgx);
    % Z step (with embedded Y soft-threshold)
    Yk = wthresh(X + Lambda/rho, 's', mu*(1+rho*gamma)/rho);
    Z = (Yk/gamma + Lambda + rho*X) / (1/gamma + rho);
    % Dual step
    Lambda = Lambda + rho*(X - Z);
    F_radmm(k) = F(X);
    T_radmm(k) = toc;
    if T_radmm(k) >= time_limit, break; end
end
iter_radmm = k;
fprintf('  done in %.1fs at iter %d, F=%.4e\n', T_radmm(iter_radmm), iter_radmm, F_radmm(iter_radmm));

% ============================== SOC =========================================
%fprintf('SOC...\n');
%X = X0; Y = X0;
%Lambda = zeros(size(X));
%F_soc = zeros(1, N); F_soc(1) = F(X);
%T_soc = zeros(1, N); T_soc(1) = 0;
%tic
%for k = 2:N
%    temp_F = @(X) -0.5*trace(X.'*H*X) + mu*sum(sum(abs(X))) + rho / 2 * norm(X - Y + Lambda, 'fro')^2;
%    % X step, proximal gradient method to
%    % solve f + g + quadratic term
%    for i = 1:100
%        grad_f = -H*X + rho * (X - Y + Lambda);
%        grad_map = (X - wthresh(X - eta*grad_f, 's', mu * eta)) / eta;
%        if norm(grad_map, 'fro') < 1e-8
%            break;
%        end
%        X = X - eta * grad_map;
%    end
%    % Y step: a projection step
%    [U,~,V] = svd(X + Lambda);
%    Y = U*eye(n,p)*V.';
%    % Lambda step
%    Lambda = Lambda + (X - Y);
%    F_soc(k) = F(Y);
%    T_soc(k) = toc;
%    if abs(F_soc(k) - F_soc(k-1)) <= 1e-8, break; end
%end
%iter_soc = k;
%fprintf('  done in %.1fs at iter %d, F=%.4e\n', T_soc(iter_soc), iter_soc, F_soc(iter_soc));

% ============================== MADMM =======================================
%fprintf('MADMM...\n');
%X = X0; Y = X0;
%Lambda = zeros(size(X));
%F_madmm = zeros(1, N); F_madmm(1) = F(X);
%T_madmm = zeros(1, N); T_madmm(1) = 0;
%tic
%for k = 2:N
%    % X step: a Riemannian gradient step
%    for i = 1:100
%        gx = -H*X + rho*(X - Y + Lambda);
%        rgx = proj(X, gx);
%        if norm(rgx, 'fro') < 1e-8
%            break;
%        end
%        X = retr(X, -eta*rgx);
%    end
%    % Y step
%    Y = wthresh(X + Lambda ,'s', mu/rho);
%    % Lambda step
%    Lambda = Lambda + (X - Y);
%    F_madmm(k) = F(X);
%    T_madmm(k) = toc;
%    if abs(F_madmm(k) - F_madmm(k-1)) <= 1e-8, break; end
%end
%iter_madmm = k;
%fprintf('  done in %.1fs at iter %d, F=%.4e\n', T_madmm(iter_madmm), iter_madmm, F_madmm(iter_madmm));

% ============================== ARADMM ======================================
fprintf('ARADMM...\n');
X = X0; Z = X0;
Lambda = zeros(size(X));
etak = 1/L; rhok = 5; beta0 = 50; crho = 1; cbeta = 50;
newcv = norm(Z - X, 'fro');
inicv = norm(Z - X, 'fro');
F_aradmm = zeros(1, N); F_aradmm(1) = F(X);
T_aradmm = zeros(1, N); T_aradmm(1) = 0;
tic
for k = 2:N
    oldcv = newcv;
    %etak = etak/k^(1/3);
    % Z step
    Z = wthresh(X + Lambda/rhok, 's', mu/rhok);
    % X step: a gradient step
    for i = 1:1
        gx = -H*X + Lambda + rhok*(X - Z);
        rgx = proj(X, gx);
        X = retr(X, -(etak)*rgx/(k^(1/3)));
    end
    % update beta and rho
    newcv = norm(Z - X, 'fro');
    if newcv > oldcv
        betak = min(beta0*(inicv*(log(2))^2)/(newcv*(k+1)^2 *log(k+2)), cbeta/(k^(1/3)*(log(k+1)^2)));
        rhok = crho*rhok*(k^(1/3));
    end
    % Lambda step
    Lambda = Lambda + betak*(X - Z);
    F_aradmm(k) = F(X);
    T_aradmm(k) = toc;
    if T_aradmm(k) >= time_limit, break; end
end
iter_aradmm = k;
fprintf('  done in %.1fs at iter %d, F=%.4e\n', T_aradmm(iter_aradmm), iter_aradmm, F_aradmm(iter_aradmm));

% ============================== OADMM =======================================
fprintf('OADMM...\n');
X = X0; Z = X0;
Lambda = zeros(size(X));
orho = 10*mu; sigma = 1.1; delta = 1e-3;
f_oadmm       = @(X) -0.5*trace(X.'*H*X);
g_oadmm       = @(Y) mu*sum(sum(abs(Y)));
g_gamma_oadmm = @(Z,gamma) mu*(g_oadmm(wthresh(Z,'s',gamma)) + 1/(2*gamma)*norm(wthresh(Z,'s',gamma) - Z,'fro')^2);
L_M           = @(X,Z,Lambda,gamma,rho) f_oadmm(X) + mu*g_gamma_oadmm(Z,gamma) + trace(Lambda.'*(X-Z)) + rho/2*norm(X-Z)^2;
F_oadmm = zeros(1, N); F_oadmm(1) = F(X);
T_oadmm = zeros(1, N); T_oadmm(1) = 0;
tic
for k = 2:N
    ogamma = 4/((2-sigma)*orho);
    Xbar = X;
    oeta = 1/(L + orho);
    % X step: a gradient step
    for i = 1:1
        gx = -H*Xbar + Lambda + orho*(Xbar - Z);
        rgx = proj(Xbar, gx);
        X = retr(Xbar, -(oeta)*rgx);
        Gra_norm = norm(rgx, 'fro');
        % line search
        ls_cut = 1;
        while L_M(X,Z,Lambda,ogamma,orho) > L_M(Xbar,Z,Lambda,ogamma,orho) - delta*oeta*Gra_norm^2 && ls_cut <= 10
            oeta = 0.5*oeta;
            X = retr(Xbar, -(oeta)*rgx);
            ls_cut = ls_cut + 1;
        end
    end
    % Z step (also update Y)
    Y = wthresh(X + Lambda/orho, 's', mu*(1+orho*ogamma)/orho);
    Z = (Y/ogamma + Lambda + orho*X) / (1/ogamma + orho);
    % Lambda step
    Lambda = Lambda + sigma*orho*(X - Z);
    orho = orho*(1+0.1*k^(1/3));
    F_oadmm(k) = F(X);
    T_oadmm(k) = toc;
    if T_oadmm(k) >= time_limit, break; end
end
iter_oadmm = k;
fprintf('  done in %.1fs at iter %d, F=%.4e\n', T_oadmm(iter_oadmm), iter_oadmm, F_oadmm(iter_oadmm));

% ============================== PLOT ========================================
algs = {'NS-ADPMM','SVD-ADPMM','ManPG','ManPG-Ada','RADMM','ARADMM','OADMM'};
Tc = {T_adpmm(1:iter_adpmm), T_adpmm_svd(1:iter_adpmm_svd), ...
      T_manpg(1:iter_manpg), T_manpg_ada(1:iter_manpg_ada), ...
      T_radmm(1:iter_radmm), T_aradmm(1:iter_aradmm), T_oadmm(1:iter_oadmm)};
Fc = {F_adpmm(1:iter_adpmm), F_adpmm_svd(1:iter_adpmm_svd), ...
      F_manpg(1:iter_manpg), F_manpg_ada(1:iter_manpg_ada), ...
      F_radmm(1:iter_radmm), F_aradmm(1:iter_aradmm), F_oadmm(1:iter_oadmm)};

if strcmp(x_mode, 'time')
    Xc = Tc;  xlbl = 'Time (s)';   xtag = 'time';
else
    Xc = cellfun(@(f) 1:numel(f), Fc, 'UniformOutput', false);
    xlbl = 'Iteration'; xtag = 'iter';
end

Fstar = min(cellfun(@min, Fc));
styles  = {'-',':','-',':',':','-','-'};   styles{2} = ':';

% figure('Visible','off'); hold on;
% h = gobjects(numel(Fc),1);
% order = [1 2 3 4 5 6 7];
% m = 1; % Show every mth point
% % for i = order
% %     semilogy(Xc{i}, Fc{i} - Fstar, ...
% %         'LineStyle', styles{i}, 'LineWidth', 2);
% % end
% for i = order
%     h(i) = plot(Xc{i}(1:m:end), Fc{i}(1:m:end), ...
%         'LineStyle', styles{i}, 'LineWidth', 2);
% end
% % set(gca,'YScale','log');
% %ylim([1e-17, inf]);
% % xlabel(xlbl); ylabel('F - F^\ast'); %log plot labels
% ylim([Fstar - 5, max(cellfun(@max, Fc)) + 5]);
% xlabel(xlbl); ylabel('F');
% ds_disp = strrep(dataset,'_','\_');
% % k = p: components, lambda = mu; sparsity coefficient - to match paper
% title(sprintf('sPCA %s (n=%d, k=%d, \\lambda=%g)', ds_disp, n, p, mu));
% drawnow % Need to save twice for legend to properly fit text
% legend('NS-ADPMM','SVD-ADPMM','ManPG','ManPG-Ada','RADMM','ARADMM','OADMM');
% saveas(gcf, sprintf('spca_%s_n%d_p%d_mu%.2f_subopt_%s.png', dataset, n, p, mu, xtag));
% legend('NS-ADPMM','SVD-ADPMM','ManPG','ManPG-Ada','RADMM','ARADMM','OADMM');
% saveas(gcf, sprintf('spca_%s_n%d_p%d_mu%.2f_subopt_%s.png', dataset, n, p, mu, xtag));
% fprintf('Saved: png\n');


figure('Visible','off'); hold on; grid on;
set(gca, 'FontSize', 16);              % tick labels
h = gobjects(numel(Fc),1);
order = [1 2 3 4 5 6 7];
m = 1;

for i = order
    h(i) = plot(Xc{i}(1:m:end), Fc{i}(1:m:end), ...
        'LineStyle', styles{i}, 'LineWidth', 2.5);
end

ylim([Fstar - 5, max(cellfun(@max, Fc)) + 5]);
xlabel(xlbl, 'FontSize', 20);
ylabel('F',  'FontSize', 20);

ds_disp = strrep(dataset,'_','\_');
title(sprintf('sPCA %s (n=%d, k=%d, \\lambda=%g)', ds_disp, n, p, mu), ...
      'FontSize', 20);

legend('NS-ADPMM','SVD-ADPMM','ManPG','ManPG-Ada','RADMM','ARADMM','OADMM');

fname = sprintf('spca_%s_n%d_p%d_mu%.2f_subopt_%s.png', dataset, n, p, mu, xtag);
exportgraphics(gcf, fname, 'Resolution', 300);
legend('NS-ADPMM','SVD-ADPMM','ManPG','ManPG-Ada','RADMM','ARADMM','OADMM');
exportgraphics(gcf, fname, 'Resolution', 300);
fprintf('Saved: %s\n', fname);
