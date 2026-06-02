dataset = 'news20';
load_spca;
N = 5000;
time_limit = 360;
%p_list = [100, 1000, 10000, 100000];
mu_list = [0.01, 0.1, 1, 5, 10, 20, 50, 100];

for i = 1:numel(mu_list)
    mu = mu_list(i);
    fprintf('\n=========== mu = %.2f ===========\n', mu);
    try
        solvers;   % runs your existing script with current N, p
    catch ME
        fprintf(2, 'mu=%.2f failed: %s\n', mu, ME.message);
    end
end
