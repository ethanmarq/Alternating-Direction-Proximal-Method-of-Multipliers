fprintf('ADPMM...\n');
X = X0; Z = X0; Y = zeros(n, p);
F_adpmm = zeros(1, N); F_adpmm(1) = F(X);
Ws = zeros(N-1, p);
tic
for k = 2:N
    % X-update: orthogonalize L*X + H*X + rho*Z - Y via Newton-Schulz
    B = L*X + H*X + rho*Z - Y;
    nrmB = norm(B);
    if nrmB < eps
        W = B;
    else
        W = B / (nrmB * 1.05);
        Ip = eye(p);
        for j = 1:5
            W = 0.5 * W * (3*Ip - W'*W);
        end
    end
    Ws(k-1, :) = svd(W, 0)';
    Wn(k-1) = mean(svd(W,0));
    % fprintf('W after NS iter: %.3f \n', norm(W,2));
    X = W;
    % Z-update: soft-thresholding
    V = X + Y/rho;
    Z = sign(V) .* max(abs(V) - mu/rho, 0);
    % Dual update
    Y = Y + rho*(X - Z);
    F_adpmm(k) = F(X);
    if abs(F_adpmm(k) - F_adpmm(k-1)) <= 1e-8, break; end
end
iter_adpmm = k;
fprintf('done in %.1fs at iter %d, F=%.4e\n', toc, iter_adpmm, F_adpmm(iter_adpmm));
if round(mean(mean(Ws)), 3) ~= 1; fprintf('NS DID NOT ORTHOGONALIZE\n'); end

fprintf('ADPMMv2...\n');
X = X0; Z = X0; Y = zeros(n, p);
F_adpmm2 = zeros(1, N); F_adpmm2(1) = F(X);
Ws2 = zeros(N-1, p);
Wn2 = zeros(1, N-1);
tic
for k = 2:N
    % X-update: orthogonalize L*X + H*X + rho*Z - Y via Newton-Schulz
    B = L*X + H*X + rho*Z - Y;
    nrmB = norm(B, 'fro');
    if nrmB < eps
        W = B;
    else
        W = B / (nrmB + eps);
        Ip = eye(p);
        for j = 1:5
            W = 0.5 * W * (3*Ip - W'*W);
        end
    end
    Ws2(k-1, :) = svd(W, 0)';
    Wn2(k-1) = mean(svd(W,0));
    X = W;
    % Z-update: soft-thresholding
    V = X + Y/rho;
    Z = sign(V) .* max(abs(V) - mu/rho, 0);
    % Dual update
    Y = Y + rho*(X - Z);
    F_adpmm2(k) = F(X);
    if abs(F_adpmm2(k) - F_adpmm2(k-1)) <= 1e-8, break; end
end
iter_adpmm2 = k;
fprintf('done in %.1fs at iter %d, F=%.4e\n', toc, iter_adpmm2, F_adpmm2(iter_adpmm2));
if round(mean(mean(Ws2)), 3) ~= 1; fprintf('NS DID NOT ORTHOGONALIZE\n'); end


fprintf('ADPMMv3...\n');
X = X0; Z = X0; Y = zeros(n, p);
F_adpmm3 = zeros(1, N); F_adpmm3(1) = F(X);
Ws3 = zeros(N-1, p);
Wn3 = zeros(1, N-1);
tic
for k = 2:N
    % X-update: orthogonalize L*X + H*X + rho*Z - Y via Newton-Schulz
    B = L*X + H*X + rho*Z - Y;
    [U, ~, V] = svd(B, 'econ');
    W = U*V';
    Ws3(k-1, :) = svd(W, 0)';
    Wn3(k-1) = mean(svd(W,0));
    X = W;
    % Z-update: soft-thresholding
    V = X + Y/rho;
    Z = sign(V) .* max(abs(V) - mu/rho, 0);
    % Dual update
    Y = Y + rho*(X - Z);
    F_adpmm3(k) = F(X);
    if abs(F_adpmm3(k) - F_adpmm3(k-1)) <= 1e-8, break; end
end
iter_adpmm3 = k;
fprintf('done in %.1fs at iter %d, F=%.4e\n', toc, iter_adpmm3, F_adpmm3(iter_adpmm3));

if round(mean(mean(Ws3)), 3) ~= 1; fprintf('NS DID NOT ORTHOGONALIZE\n'); end
figure('Visible', 'off'); hold on;
plot(1:iter_adpmm, F_adpmm(1:iter_adpmm), 'LineWidth', 2);
plot(1:iter_adpmm2, F_adpmm2(1:iter_adpmm2), 'LineWidth', 2);
plot(1:iter_adpmm3, F_adpmm3(1:iter_adpmm3), 'LineWidth', 2);
xlabel('Iteration'); ylabel('F(X)');
grid on;
legend('ADPMM', 'ADPMMv2', 'ADPMMv3', 'AutoUpdate', 'on');
saveas(gcf, sprintf('ADPMM-analysis.png'));
