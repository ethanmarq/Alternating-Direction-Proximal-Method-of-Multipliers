function val = ff(x, c, m_zol)
% FF  Evaluate the Zolotarev rational function at scalar x.
    val = x;
    for j = 1 : m_zol
        val = val * (x^2 + c(2*j)) / (x^2 + c(2*j - 1));
    end
end
