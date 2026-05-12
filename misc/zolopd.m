function [U, varargout] = zolopd(A, compute_hermitian, alpha, L)
% ZOLOPD  Polar decomposition via Zolotarev approximation.
%
%   [U, m_zol, it]          = zolopd(A)
%   [U, H, m_zol, it]       = zolopd(A, true)
%   [U, ...]                = zolopd(A, compute_hermitian, alpha, L)
%
%   Inputs:
%     A                 - (m x n) real matrix
%     compute_hermitian - (optional, default false) if true, also return H
%     alpha             - (optional) estimate of ||A||; computed if omitted
%     L                 - (optional) estimate of 1/cond(A); computed if omitted
%
%   Outputs (compute_hermitian = false):
%     U     - polar factor  (m x n)
%     m_zol - Zolotarev degree used
%     it    - number of iterations performed
%
%   Outputs (compute_hermitian = true):
%     U     - polar factor  (m x n)
%     H     - Hermitian factor (n x n)
%     m_zol - Zolotarev degree used
%     it    - number of iterations performed

    if nargin < 2 || isempty(compute_hermitian)
        compute_hermitian = false;
    end

    % Work in double precision throughout.
    A = double(A);
    [m, n] = size(A);

    % Check if A is (numerically) symmetric.
    if m == n && norm(A - A', 'fro') / norm(A, 'fro') < 1e-14
        symm = true;
    else
        symm = false;
    end

    % Estimate largest singular value if alpha not provided.
    if nargin < 3 || isempty(alpha)
        alpha = normest_power(A, 1e-6, 20);
    end

    % Scale A so that ||U|| ~ 1.
    U = A / alpha;

    % Estimate smallest singular value if L not provided.
    if nargin < 4 || isempty(L)
        Y = U;
        if m > n
            [~, R] = qr(U, 0);   % thin QR
            Y = R;
        end
        cond_Y  = cond(Y, 1);
        smin_est = norm(Y, 1) / cond_Y;
        L = smin_est / sqrt(n);
    end

    U   = U / L;      % now ||U|| ~ 1 and smallest sv ~ 1
    con = 1 / L;

    it    = 0;
    m_zol = choosem(con);
    itmax = 1 + (con >= 2);   % 1 if con < 2, else 2

    % ---- Main iteration loop ----
    while it < itmax
        it = it + 1;
        kp          = 1 / con;
        alpha_angle = acos(kp);          % angle for elliptic functions
        K           = mellipke(alpha_angle);
        m_zol       = choosem(con);

        % Build coefficient vector c of length 2*m_zol.
        c = zeros(2 * m_zol, 1);
        for ii = 1 : 2 * m_zol
            u_val      = ii * K / (2 * m_zol + 1);
            [sn, cn, ~] = mellipj(u_val, alpha_angle);
            c(ii)      = (sn / cn)^2;
        end

        % Rational correction.
        U = computeAA(U, c, it);

        % Normalize by ff(1).
        f1  = ff(1, c, m_zol);
        U   = U / f1;

        % Update condition number estimate.
        con = max(ff(con, c, m_zol) / f1, 1);

        if con < 2
            break;
        end

        if symm
            U = 0.5 * (U.' + U);   % enforce symmetry
        end
    end

    % One Newton-Schulz post-processing step.
    U = 1.5 * U - 0.5 * U * (U.' * U);

    % Return results.
    if compute_hermitian
        H = U.' * A;
        H = (H + H.') / 2;
        varargout = {H, m_zol, it};
    else
        varargout = {m_zol, it};
    end
end