% solvers.m
% Minimal sPCA solver comparison on a single dataset.
% Solvers ported from the trusted-author repo (compare_sPCA.m).
% Expects H, X0, n, p, mu, N, rho, eta, t, L, F to be in workspace.

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
    B = L*X + H*X + rho*Z - Y;
    nrmB = norm(B, 'fro');
    if nrmB < eps
        W = B;
    else
        W = B / (nrmB * 1.05);
        Ip = eye(p);
        for j = 1:5
            W = 0.5 * W * (3*Ip - W'*W);
        end
    end
    X = W;
    % Z-update: soft-thresholding
    V = X + Y/rho;
    Z = sign(V) .* max(abs(V) - mu/rho, 0);
    % Dual update
    Y = Y + rho*(X - Z);
    F_adpmm(k) = F(Z);
    T_adpmm(k) = toc;
    if abs(F_adpmm(k) - F_adpmm(k-1)) <= 1e-8, break; end
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
    B = L*X + H*X + rho*Z - Y;
    [U, ~, V] = svd(B, 'econ');
    X = U*V';
    % Z-update: soft-thresholding
    V = X + Y/rho;
    Z = sign(V) .* max(abs(V) - mu/rho, 0);
    % Dual update
    Y = Y + rho*(X - Z);
    F_adpmm_svd(k) = F(Z);
    T_adpmm_svd(k) = toc;
    if abs(F_adpmm_svd(k) - F_adpmm_svd(k-1)) <= 1e-8, break; end
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
    'phi_init',   X0);

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
    'inner_iter', 100, ...
    'type',       1, ...
    'phi_init',   X0, ...
    'F_manpg',    -Inf);

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
    if abs(F_radmm(k) - F_radmm(k-1)) <= 1e-8, break; end
end
iter_radmm = k;
fprintf('  done in %.1fs at iter %d, F=%.4e\n', T_radmm(iter_radmm), iter_radmm, F_radmm(iter_radmm));

% ============================== PLOT ========================================
Fstar = min([F_adpmm(1:iter_adpmm) F_adpmm_svd(1:iter_adpmm_svd) ...
             F_manpg(1:iter_manpg) F_manpg_ada(1:iter_manpg_ada) ...
             F_radmm(1:iter_radmm)]);
figure('Visible', 'off');
%semilogy(T_adpmm(1:iter_adpmm),         F_adpmm(1:iter_adpmm)         - Fstar + eps, 'LineWidth', 2); hold on;
%semilogy(T_adpmm_svd(1:iter_adpmm_svd), F_adpmm_svd(1:iter_adpmm_svd) - Fstar + eps, 'LineWidth', 2);
%semilogy(T_manpg(1:iter_manpg),         F_manpg(1:iter_manpg)         - Fstar + eps, 'LineWidth', 2);
%semilogy(T_manpg_ada(1:iter_manpg_ada), F_manpg_ada(1:iter_manpg_ada) - Fstar + eps, 'LineWidth', 2);
%semilogy(T_radmm(1:iter_radmm),         F_radmm(1:iter_radmm)         - Fstar + eps, 'LineWidth', 2);
plot(T_adpmm(1:iter_adpmm),         F_adpmm(1:iter_adpmm) + eps, 'LineWidth', 2); hold on;
plot(T_adpmm_svd(1:iter_adpmm_svd), F_adpmm_svd(1:iter_adpmm_svd) + eps, 'LineWidth', 2);
plot(T_manpg(1:iter_manpg),         F_manpg(1:iter_manpg) + eps, 'LineWidth', 2);
plot(T_manpg_ada(1:iter_manpg_ada), F_manpg_ada(1:iter_manpg_ada) + eps, 'LineWidth', 2);
plot(T_radmm(1:iter_radmm),         F_radmm(1:iter_radmm) + eps, 'LineWidth', 2);
xlabel('Time (s)'); ylabel('F');
legend('ADPMM','ADPMM-SVD','ManPG','ManPG-Ada','RADMM','Location','best','AutoUpdate','on');
title(sprintf('n=%d, p=%d, \\mu=%g, \\rho=%g', n, p, mu, rho));
grid on;
ylim([1e-2, 1e5]);
saveas(gcf, sprintf('spca-%d-%.2d.png', p, rho));
fprintf('Saved: png\n');
