clear variables;
close all;
rng(0);
lambda = 1;
rho = 2;
T = 20;

% --- Configuration ---
n = 2000;           % Number of nodes (rows/cols)
k = 50;
density = 0.5;   % Probability of an edge existing (0 to 1)

R = rand(n) < density; 
R = double(R); % Convert logical to double

% Enforce symmetry for undirected graph: A = (R + R') / 2
% We use triu to take the upper triangle and mirror it to ensure perfect symmetry
A = triu(R, 1) + triu(R, 1)';

% Ensure no self-loops (diagonal must be zero)
A(1:n+1:end) = 0;

% --- 2. Calculate Degree Matrix (D) ---
% The degree of a node is the sum of the weights of connected edges
degrees = sum(A, 2); 
D = diag(degrees);

% --- 3. Compute Laplacian Matrix (L) ---
L = D - A;

[U_init, ~, V_init] = svd(randn(n));
X = U_init * V_init';

X = X(:,1:k);
Z = X;
Y = zeros(n, k);
obj = zeros([T,1]);
obj_ns = zeros([T,1]);
G = norm(L)*eye(n)-L;

for i = 1:T
    obj(i) = trace(X'*L*X)/2 + lambda * sum(abs(Z),"all");

    [U, ~, V] = svd(rho*Z-Y+G*X, 'econ');
    X = U * V';
    
    Vk = X + Y./rho;
    Z = sign(Vk).*max(abs(Vk) - lambda/rho, 0);
    Y = Y + rho * (X-Z);
end

X = U_init * V_init';
X = X(:,1:k);
Z = X;
Y = zeros(n, k);
for i = 1:T
    obj_ns(i) = trace(X'*L*X)/2 + lambda * sum(abs(Z),"all");

    X = newtonschulz5(rho*Z-Y+G*X);

    Vk = X + Y./rho;
    Z = sign(Vk).*max(abs(Vk) - lambda/rho, 0);
    Y = Y + rho * (X-Z);
end


plot(obj);
hold on
plot(obj_ns);
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
