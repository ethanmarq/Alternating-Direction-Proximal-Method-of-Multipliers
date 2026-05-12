function [sn, cn, dn] = mellipj(u, alpha)
% MELLIPJ  Jacobi elliptic functions via AGM backward recurrence.
%   u     : scalar argument
%   alpha : modular angle in radians  (k = sin(alpha))
    tol   = eps('double');
    m_val = sin(alpha)^2;

    % Forward AGM sequence.
    a_vals = 1;
    b_vals = cos(alpha);
    c_vals = sin(alpha);
    i = 1;

    while abs(c_vals(i)) > tol && i < 1000
        a_next = 0.5 * (a_vals(i) + b_vals(i));
        b_next = sqrt(a_vals(i) * b_vals(i));
        c_next = 0.5 * (a_vals(i) - b_vals(i));
        a_vals(end+1) = a_next; %#ok<AGROW>
        b_vals(end+1) = b_next; %#ok<AGROW>
        c_vals(end+1) = c_next; %#ok<AGROW>
        i = i + 1;
    end
    N = i - 1;          % number of AGM steps taken

    % Amplitude at the finest level.
    phi = 2^N * a_vals(end) * u;

    % Backward recurrence for phi.
    for j = N : -1 : 1
        temp = c_vals(j+1) * sin(phi) / a_vals(j+1);
        temp = max(-1, min(1, temp));   % clamp for numerical safety
        phi  = 0.5 * (asin(temp) + phi);
    end

    sn = sin(phi);
    cn = cos(phi);
    dn = sqrt(1 - m_val * sn^2);
end