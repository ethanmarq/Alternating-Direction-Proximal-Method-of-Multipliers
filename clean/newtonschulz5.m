function W = newtonschulz5(B, steps)
    if nargin < 2, steps = 10; end
    [~, k] = size(B);
    nrmB = norm(B, 'fro');
    if nrmB < eps, W = B; return; end
    W = B / nrmB;
    I = eye(k);
    for i = 1:steps
        WtW = W' * W;
        W = 0.5 * W * (3*I - WtW);
    end
end
