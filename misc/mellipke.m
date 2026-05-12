function [K, E] = mellipke(alpha)
% MELLIPKE  Complete elliptic integrals K and E via AGM.
%   alpha : modular angle in radians  (k = sin(alpha))
    tol   = eps('double');
    m_val = sin(alpha)^2;
    a0    = 1;
    b0    = cos(alpha);
    s0    = m_val;
    i1    = 0;
    mm    = 1;

    while mm > tol
        a1  = 0.5 * (a0 + b0);
        b1  = sqrt(a0 * b0);
        c1  = 0.5 * (a0 - b0);
        i1  = i1 + 1;
        w1  = 2^i1 * c1^2;
        mm  = w1;
        s0  = s0 + w1;
        a0  = a1;
        b0  = b1;
    end

    K = pi / (2 * a1);
    E = K * (1 - s0 / 2);
end