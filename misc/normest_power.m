function nrm = normest_power(A, tol, max_iter)
% NORMEST_POWER  Estimate the 2-norm of A using the power method.
    if nargin < 2, tol      = 1e-6; end
    if nargin < 3, max_iter = 20;   end

    n   = size(A, 2);
    x   = randn(n, 1);
    x   = x / norm(x);

    nrm_old = 0;
    for i = 1 : max_iter
        y       = A * x;
        nrm_new = norm(y);
        if nrm_new == 0
            nrm = 0;
            return;
        end
        x       = A.' * y / nrm_new;
        if abs(nrm_new - nrm_old) < tol * nrm_new
            break;
        end
        nrm_old = nrm_new;
    end
    nrm = nrm_new;
end