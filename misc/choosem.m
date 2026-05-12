function m = choosem(con)
% CHOOSEM  Select Zolotarev degree m based on estimated condition number.
    if     con < 1.001,  m = 2;
    elseif con <= 1.01,  m = 3;
    elseif con <= 1.1,   m = 4;
    elseif con <= 1.2,   m = 5;
    elseif con <= 1.5,   m = 6;
    elseif con <= 2,     m = 8;
    elseif con < 6.5,    m = 2;
    elseif con < 180,    m = 3;
    elseif con < 1.5e4,  m = 4;
    elseif con < 2e6,    m = 5;
    elseif con < 1e9,    m = 6;
    elseif con < 3e12,   m = 7;
    else,                m = 8;
    end
end