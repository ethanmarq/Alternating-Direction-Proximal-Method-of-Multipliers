function AA = computeAA(A, c, it)
% COMPUTEAA  Apply the rational correction to A.
%
%   A  - current iterate (m x n)
%   c  - coefficient vector of length 2*m_zol (double)
%   it - current iteration index

    AA  = A;
    [m, n] = size(A);
    r   = length(c) / 2;    % m_zol

    for ii = 1 : r
        % Numerator product.
        enu = 1;
        for jj = 1 : r
            enu = enu * (c(2*ii - 1) - c(2*jj));
        end

        % Denominator product (exclude jj == ii).
        den = 1;
        for jj = 1 : r
            if jj ~= ii
                den = den * (c(2*ii - 1) - c(2*jj - 1));
            end
        end

        c_val = c(2*ii - 1);

        if it <= 1 && max(c(1:end-1)) > 1e2
            % QR-based branch (Householder).
            sqrt_c  = sqrt(c_val);
            I_n     = eye(n, class(A));
            stacked = [A; sqrt_c * I_n];
            [Q, ~]  = qr(stacked, 0);         % thin QR of (m+n) x n
            Q_top   = Q(1:m,   :);
            Q_bot   = Q(m+1:end, :);
            AA      = AA - (enu / den / sqrt_c) * (Q_top * Q_bot.');
        else
            % Cholesky-based branch.
            I_n  = eye(n, class(A));
            C    = chol(A.' * A + c_val * I_n);   % upper triangular R s.t. R'R = A'A + cI
            % Qtmp = A * (A'A + cI)^{-1}  via two triangular solves.
            Qtmp = (C \ (C.' \ A.')).' ;          % = A * (R'R)^{-1}
            AA   = AA - (enu / den) * Qtmp;
        end
    end
end