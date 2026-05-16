% spca_solvers.m
% Minimal sPCA solver comparison on a single dataset.
% Loads X from a .mat file, builds H = X'*X, runs all solvers, plots curves.

% clear; clc; close all;
addpath('misc');
addpath('SSN_subproblem');


% ============================== ADPMM =======================================
fprintf('ADPMM...\n');
X = X0; Z = X0; Y = zeros(n, p);
F_adpmm = zeros(1, N); F_adpmm(1) = F(X);
tic
for k = 2:N
    B = L*X + H*X + rho*Z - Y;
    nrmB = norm(B);
    W = B / (nrmB * 1.05);
    for j = 1:5
        W = 0.5 * W * (3*eye(p) - W'*W);
    end
    X = W;
    V = X + Y/rho;
    Z = sign(V) .* max(abs(V) - mu/rho, 0);
    Y = Y + rho*(X - Z);
    F_adpmm(k) = F(X);
end
fprintf('  done in %.1fs, F=%.4e\n', toc, F_adpmm(end));

% ============================== ManPG =======================================
fprintf('ManPG...\n');
U = X0;
Dn = sparse(DuplicationM(p));
pDn = (Dn'*Dn) \ Dn';
prox = @(b,l,r) proximal_l1(b, l*mu, r);
Lam = zeros(p);
F_manpg = zeros(1, N); F_manpg(1) = F(U);
tic
for k = 2:N
    [PU, ~, Lam, ~, ~] = Semi_newton_matrix(n, p, U, t, U + t*(H*U), mu*t, ...
        1e-12, prox, 100, Lam, Dn, pDn);
    [T_, SIG, S_] = svd(PU'*PU);
    U = PU * (T_ * diag(1./sqrt(diag(SIG))) * S_');
    F_manpg(k) = F(U);
end
fprintf('  done in %.1fs, F=%.4e\n', toc, F_manpg(end));

% ============================== RADMM =======================================
fprintf('RADMM...\n');
X = X0; Z = X0; Lambda = zeros(n, p);
gamma = 1e-8;
F_radmm = zeros(1, N); F_radmm(1) = F(X);
tic
for k = 2:N
    gx = -H*X + Lambda + rho*(X - Z);
    rgx = gx - X * ((X'*gx + gx'*X)/2);
    X = X - eta*rgx;
    [U_,~,V_] = svd(X, 'econ'); X = U_*V_';
    Yk = sign(X + Lambda/rho) .* max(abs(X + Lambda/rho) - mu*(1+rho*gamma)/rho, 0);
    Z = (Yk/gamma + Lambda + rho*X) / (1/gamma + rho);
    Lambda = Lambda + rho*(X - Z);
    F_radmm(k) = F(X);
end
fprintf('  done in %.1fs, F=%.4e\n', toc, F_radmm(end));

% ============================== PLOT ========================================
% Shift by best objective seen across all methods (so y axis is F - F*)
Fstar = min([F_adpmm F_manpg F_radmm]);
figure('Visible', 'off');
semilogy(F_adpmm - Fstar + eps, 'LineWidth', 2); hold on;
semilogy(F_manpg - Fstar + eps, 'LineWidth', 2);
semilogy(F_radmm - Fstar + eps, 'LineWidth', 2);
xlabel('Iteration'); ylabel('F - F^*');
legend('ADPMM','ManPG','RADMM','Location','best');
title(sprintf('n=%d, p=%d, \\mu=%g, \\rho=%g', n, p, mu, rho));
grid on;
ylim([1e-2, 1e5]);
saveas(gcf, sprintf('spca-rho%.2f.png', rho));
fprintf('Saved: spca-rho%.2f.png\n', rho);
