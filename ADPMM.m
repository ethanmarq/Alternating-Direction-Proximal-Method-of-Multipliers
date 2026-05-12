clear variables;
close all;
addpath misc
rng(0);

lambda = 1;
rho = 2;
T = 20;

% --- Configuration for Sparse PCA ---
n = 5000;      % data dimension
p = 1000;      % number of features
k = 50;        % number of components

% --- Generate synthetic sparse PCA data ---
V_true = zeros(p, k);
V_true(1:10, 1)  = 1/sqrt(10);
V_true(11:20, 2) = 1/sqrt(10);
V_true(21:30, 3) = 1/sqrt(10);

Z_data = randn(n, k);
X_data = Z_data * V_true' + 0.5 * randn(n, p);
X_data = X_data - mean(X_data, 1);
X_data = X_data ./ sqrt(sum(X_data.^2, 1));

% covariance matrix
C = X_data' * X_data;

% minimize:  -0.5 * trace(X'*C*X) + lambda * ||Z||_1
% subject to X'*X = I, X = Z

% Initialize X on Stiefel (p x k)
[U_init, ~, V_init] = svd(randn(p));
X = U_init * V_init';
X = X(:,1:k);

Z = X;
Y = zeros(p, k);

obj = zeros([T,1]);
obj_ns = zeros([T,1]);

% SVD-based ADMM for sparse PCA
for i = 1:T
    obj(i) = -0.5 * trace(X' * C * X) + lambda * sum(abs(Z), "all");
    
    % X-update: projection onto Stiefel via SVD
    [U, ~, V] = svd(C*X + rho*Z - Y, 'econ');
    X = U * V';
    
    % Z-update: soft-thresholding (prox of lambda ||.||_1)
    Vk = X + Y./rho;
    Z = sign(Vk).*max(abs(Vk) - lambda/rho, 0);
    
    % dual update
    Y = Y + rho * (X - Z);
end

% reset for Newton–Schulz variant
X = U_init * V_init';
X = X(:,1:k);
Z = X;
Y = zeros(p, k);
C = X_data' * X_data;

for i = 1:T
    obj_ns(i) = -0.5 * trace(X' * C * X) + lambda * sum(abs(Z), "all");
    
    % X-update: projection onto Stiefel via Newton–Schulz approximation
    X = qwdh(C*X + rho*Z - Y);
    %newtonschulz5
    %zeropower_via_newtonschulz5
    %precond_newtonschulz
    %
    
    % Z-update: soft-thresholding
    Vk = X + Y./rho;
    Z = sign(Vk).*max(abs(Vk) - lambda/rho, 0);
    
    % dual update
    Y = Y + rho * (X - Z);
end

plot(obj, 'r');
hold on
plot(obj_ns, ':');
legend('obj', 'obj\_ns');
hold off;


function X = newtonschulz5(G, steps, eps_val)
    arguments
        G
        steps (1,1) {mustBeInteger} = 5
        eps_val (1,1) {mustBeNumeric} = 1e-7
    end

    X = G/(norm(G) + eps_val);
    [rows, cols] = size(G);
    if rows > cols
        X = X';
    end
    for k = 1:steps
        A = X * X';
        B = -4.7750 * A + 2.0315 * (A * A);
        X = 3.4445 * X + B * X;
    end
    if rows > cols
        X = X';
    end
end
