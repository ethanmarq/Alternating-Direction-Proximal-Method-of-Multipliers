load_h;
N = 1000;
p_list = [100, 1000, 10000, 100000];

results = struct();
for i = 1:numel(p_list)
    p = p_list(i);
    fprintf('\n=========== p = %d ===========\n', p);
    try
        solvers;   % runs your existing script with current N, p
        results(i).p           = p;
        results(i).iter_adpmm  = iter_adpmm;
        results(i).iter_manpg  = iter_manpg;
        results(i).F_adpmm     = F_adpmm(1:iter_adpmm);
        results(i).F_manpg     = F_manpg(1:iter_manpg);
        % ...add whatever else you want to keep
        save(sprintf('sweep_p%d.mat', p), '-v7.3');
    catch ME
        fprintf(2, 'p=%d failed: %s\n', p, ME.message);
        results(i).error = ME.message;
    end
end
save('sweep_all.mat', 'results', '-v7.3');
