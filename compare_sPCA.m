%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test on the sPCA problem
% Manifold: Stiefel manifold St(n, p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear
addpath misc
addpath SSN_subproblem
%% Problem Generating
n_list = 400; p_list = 50;
mu_list = 0.01;
for n=n_list
    for p=p_list
        for mu=mu_list
            % problem
            disp("test on n="+ n +" p="+ p+" mu="+mu);
            K = n;
            A = randn(n, K); A = orth(A);
            S = diag(abs(randn(K,1)));
            H = A*S*A.';
            f = @(X) -0.5*trace(X.'*H*X);
            nabla_f = @(X) -H*X;
            g = @(Y) mu*sum(sum(abs(Y)));
            g_gamma = @(Z,gamma) mu*(g(wthresh(Z,'s',gamma))+1/(2*gamma)*norm(wthresh(Z,'s',gamma) - Z,'fro')^2);
            F = @(X) f(X) + g(X);
            sub_F = @(X) - H*X + mu*sign(X);
            L_M = @(X,Z,Lambda,gamma,rho) f(X) + mu*g_gamma(Z,gamma) + trace(Lambda.'*(X-Z)) + rho/2*norm(X-Z)^2;
            %% Algorithm
            % initialization
            N = 500;
            iter = 1;
            avg = 1; % averaged results across repeated experiments
            F_val_soc_avg = zeros([1,500]);
            F_val_madmm_avg = zeros([1,500]);
            F_val_radmm_avg = zeros([1,500]);
            F_val_oadmm_avg = zeros([1,500]);
            F_val_aradmm_avg = zeros([1,500]);
            
            F_val_manpg_avg = zeros([1,500]);

            F_val_adpmm_avg = zeros([1,500]);

            cpu_time_soc = zeros([avg,500]); cpu_time_soc(1) = eps;
            cpu_time_madmm = zeros([avg,500]); cpu_time_madmm(1) = eps;
            cpu_time_radmm = zeros([avg,500]); cpu_time_radmm(1) = eps;
            cpu_time_oadmm = zeros([avg,500]); cpu_time_oadmm(1) = eps;
            cpu_time_aradmm = zeros([avg,500]); cpu_time_aradmm(1) = eps;
            cpu_time_manpg = zeros([avg,500]); cpu_time_manpg(1) = eps;

            cpu_time_adpmm = zeros([avg,500]); cpu_time_adpmm(1) = eps;

            iter_soc = N; iter_madmm = N; iter_radmm = N; iter_aradmm = N; iter_oadmm = N;
            
            iter_manpg = N;
            % Parameters for ManPG subproblem
            L = abs(eigs(full(H),1)); % Lipschitz constant
            t = 1/L;
            inner_iter = 100;
            prox_fun = @(b,l,r) proximal_l1(b,l*mu,r);
            t_min = 1e-4; % minimum stepsize
            Dn = sparse(DuplicationM(p)); % vectorization for SSN
            pDn = (Dn'*Dn)\Dn'; % for SSN
            nu = mu; % match mu of other algorithms
            alpha = 1; % stepsize for ManPG
            tol = 1e-8*n*p;

            iter_adpmm = N;
            rho_adpmm = 2;

            disp("Total repitition " + avg);
            avg_min_among_all = 0;
            for k = 1:avg
                % random initialize
                X0 = randn(n, p);
                X0 = orth(X0);
                
                %% SOC
                X = X0; Y = X0;
                Lambda = zeros(size(X));
                eta = 1e-2; rho = 5e1;
                
                for iter=2:N
                    temp_F = @(X) -0.5*trace(X.'*H*X) + mu*sum(sum(abs(X))) + rho / 2 * norm(X - Y + Lambda, 'fro')^2;
                    admm_start = tic;
                    % X step, proximal gradient method to 
                    % solve f + g + quadratic term
                    for i=1:100
                        grad_f = -H*X + rho * (X - Y + Lambda);
                        grad_map = (X - wthresh(X - eta*grad_f, 's', mu * eta)) / eta;
                        if norm(grad_map, 'fro') < 1e-8
                            break;
                        end
                        X = X - eta * grad_map;
                    end
                    
                    % Y step: a projection step
                    [U,~,V] = svd(X + Lambda);
                    Y = U*eye(n,p)*V.';
                    % Lambda step
                    Lambda = Lambda + (X - Y);
                    elapsed_time = toc(admm_start);
                    % Value update
                    F_val_soc(iter) = F(Y);
                    if abs(F_val_soc(iter) - F_val_soc(iter-1)) <= 1e-8
                        break
                    end
                    cpu_time_soc(k,iter) = cpu_time_soc(k,iter) + elapsed_time;
                    if iter <= iter_soc
                        cpu_time_soc(k,iter+1) = cpu_time_soc(k,iter);
                    end
                    
                end
                iter_soc = min(iter, iter_soc);
                
                %% MADMM
                X = X0; Y = X0;
                Lambda = zeros(size(X));
                eta = 1e-2;  rho = 100;
                
                for iter=2:N
                    admm_start = tic;
                    % X step: a Riemannian gradient step
                    for i=1:100
                        gx = -H*X + rho*(X - Y + Lambda);
                        rgx = proj(X, gx);
                        if norm(rgx, 'fro') < 1e-8
                            break;
                        end
                        X = retr(X, -eta*rgx);
                    end
                    % Y step
                    Y = wthresh(X + Lambda ,'s', mu/rho);
                    % Lambda step
                    Lambda = Lambda + (X - Y);
                    elapsed_time = toc(admm_start);
                    % Value update
                    F_val_madmm(iter) = F(X);
                    if abs(F_val_madmm(iter) - F_val_madmm(iter-1)) <= 1e-8
                        break
                    end
                    cpu_time_madmm(k,iter) = cpu_time_madmm(k,iter) + elapsed_time;
                    if iter <= iter_madmm
                        cpu_time_madmm(k,iter+1) = cpu_time_madmm(k,iter);
                    end
                end
                iter_madmm = min(iter, iter_madmm);
                %% RADMM
                X = X0; Z = X0;
                Lambda = zeros(size(X)); 
                eta = 1e-2; gamma = 1e-8;rho = 100;
                
                for iter=2:N
                    admm_start = tic;
                    % X step: a gradient step
                    for i=1:1
                        gx = -H*X + Lambda + rho*(X - Z);
                        rgx = proj(X, gx);
                        X = retr(X, -(eta)*rgx);
                    end
                    % Z step (also update Y)
                    Y = wthresh(X + Lambda/rho,'s',mu*(1+rho*gamma)/rho);
                    Z = (Y/gamma + Lambda + rho*X) / (1/gamma + rho);
                    % Lambda step
                    Lambda = Lambda + rho*(X - Z);
                    elapsed_time = toc(admm_start);
                    % Value update
                    F_val_radmm(iter) = F(X);
                    if abs(F_val_radmm(iter) - F_val_radmm(iter-1)) <= 1e-8
                        break
                    end
                    cpu_time_radmm(k,iter) = cpu_time_radmm(k,iter) + elapsed_time;
                    if iter <= iter_radmm
                        cpu_time_radmm(k,iter+1) = cpu_time_radmm(k,iter);
                    end
                end
                iter_radmm = min(iter, iter_radmm);
                %% ARADMM
                X = X0; Z = X0;
                Lambda = zeros(size(X)); 
                etak = 2e-1; rhok = 5; beta0= 50; crho=1; cbeta=50;
                newcv=norm(Z - X, 'fro');
                inicv=norm(Z - X, 'fro');
                for iter=2:N
                    admm_start = tic;
                    oldcv=newcv;
                    % Z step 
                    Z = wthresh(X + Lambda/rhok,'s',mu/rhok);
                    % X step: a gradient step
                    for i=1:1
                        gx = -H*X + Lambda + rhok*(X - Z);
                        rgx = proj(X, gx);
                        X = retr(X, -(etak)*rgx/(iter^(1/3)));
                    end
                    % update beta and rho
                    newcv=norm(Z - X, 'fro');
                    if newcv>oldcv
                        betak=min(beta0*(inicv*(log(2))^2)/(newcv*(iter+1)^2 *log(iter+2)),cbeta/(iter^(1/3)*(log(iter+1)^2)));
                        rhok = crho*rhok*(iter^(1/3));  
                    end
                    % Lambda step
                    Lambda = Lambda + betak*(X - Z);
                    elapsed_time = toc(admm_start);
                    % Value update
                    F_val_aradmm(iter) = F(X);
                    if abs(F_val_aradmm(iter) - F_val_aradmm(iter-1)) <= 1e-8
                        break
                    end
                    cpu_time_aradmm(k,iter) = cpu_time_aradmm(k,iter) + elapsed_time;
                    if iter <= iter_aradmm
                        cpu_time_aradmm(k,iter+1) = cpu_time_aradmm(k,iter);
                    end
                end
                iter_aradmm = min(iter, iter_aradmm);
                %% OADMM
                X = X0; Z = X0;
                Lambda = zeros(size(X)); 
                orho = 10*mu; sigma=1.1; delta=1e-3;
                
                for iter=2:N
                    admm_start = tic;
                    ogamma=4/((2-sigma)*orho);
                    Xbar=X;
                    oeta = 1/orho;
                    % X step: a gradient step
                    for i=1:1
                        gx = -H*Xbar + Lambda + orho*(Xbar - Z);
                        rgx = proj(Xbar, gx);
                        X = retr(Xbar, -(oeta)*rgx);
                        Gra_norm=norm(rgx, 'fro') ;
                        % line search
                        ls_cut=1;
                        while L_M(X,Z,Lambda,ogamma,orho) > L_M(Xbar,Z,Lambda,ogamma,orho)-delta*oeta*Gra_norm^2 && ls_cut <= 10
                            oeta = 0.5*oeta;
                            X = retr(Xbar, -(oeta)*rgx);
                            ls_cut=ls_cut+1;
                        end
                    end
                    % Z step (also update Y)
                    Y = wthresh(X + Lambda/orho,'s',mu*(1+orho*ogamma)/orho);
                    Z = (Y/ogamma + Lambda + orho*X) / (1/ogamma + orho);
                    % Lambda step
                    Lambda = Lambda + sigma*orho*(X - Z);
                    orho=orho*(1+0.1*iter^(1/3));
                    elapsed_time = toc(admm_start);
                    % Value update
                    F_val_oadmm(iter) = F(X);
                    if abs(F_val_oadmm(iter) - F_val_oadmm(iter-1)) <= 1e-8 && F_val_oadmm(iter) <= min(F_val_aradmm)
                        break
                    end
                    cpu_time_oadmm(k,iter) = cpu_time_oadmm(k,iter) + elapsed_time;
                    if iter <= iter_oadmm
                        cpu_time_oadmm(k,iter+1) = cpu_time_oadmm(k,iter);
                    end
                end
                iter_oadmm = min(iter, iter_oadmm);
                %% ManPG
                U = X0;
                %F_val_manpg(1) = F(U);
                num_inexact = 0; inner_flag = 0;
                for iter=2:N
                    manpg_start = tic;
                    neg_pg = H*U;
                    if alpha < t_min || num_inexact > 10
                        inner_tol = max(5e-16, min(1e-14,1e-5*tol*t^2)); % subproblem inexact;
                    else
                        inner_tol = max(1e-13, min(1e-11,1e-3*tol*t^2));
                    end
                    % The subproblem
                    if iter == 2
                         [ PU,num_inner_x(iter),Lam_x, opt_sub_x(iter),in_flag] = Semi_newton_matrix(n,p,U,t,U + t*neg_pg,nu*t,inner_tol,prox_fun,inner_iter,zeros(p),Dn,pDn);
                    else
                         [ PU,num_inner_x(iter),Lam_x, opt_sub_x(iter),in_flag] = Semi_newton_matrix(n,p,U,t,U + t*neg_pg,nu*t,inner_tol,prox_fun,inner_iter,Lam_x,Dn,pDn);
                    end
                    if in_flag == 1   % subprolem not exact.
                        inner_flag = 1 + inner_flag;
                    end
                    V = PU-U; % The V solved from SSN
                    %%% Without linesearch
                    PU = U+alpha*V;
                    % projection onto the Stiefel manifold
                    [T, SIGMA, S] = svd(PU'*PU);   SIGMA =diag(SIGMA);   
                    U_temp = PU*(T*diag(sqrt(1./SIGMA))*S');
                    U = U_temp; % update
                    
                    elapsed_time_manpg = toc(manpg_start);
                    % Value update
                    F_val_manpg(iter) = F(U);
                    if abs(F_val_manpg(iter) - F_val_manpg(iter-1)) <= 1e-8
                        break
                    end
                    cpu_time_manpg(k,iter) = cpu_time_manpg(k,iter) + elapsed_time_manpg;
                    if iter <= iter_manpg
                        cpu_time_manpg(k,iter+1) = cpu_time_manpg(k,iter);
                    end
                end
                iter_manpg = min(iter, iter_manpg);

                %% ADPMM (Newton-Schulz variant)
                X = X0; Z = X0;
                Y_dual = zeros(size(X));
                for iter=2:N
                    adpmm_start = tic;
                    
                    % X-update: projection onto Stiefel via Newton-Schulz approximation
                    X = newtonschulz5(H*X + rho_adpmm*Z - Y_dual);
                    
                    % Z-update: soft-thresholding
                    Vk = X + Y_dual./rho_adpmm;
                    Z = sign(Vk).*max(abs(Vk) - mu/rho_adpmm, 0);
                    
                    % dual update
                    Y_dual = Y_dual + rho_adpmm * (X - Z);
                    
                    elapsed_time_adpmm = toc(adpmm_start);
                    
                    % Value update
                    F_val_adpmm(iter) = F(X);
                    if abs(F_val_adpmm(iter) - F_val_adpmm(iter-1)) <= 1e-8
                        break
                    end
                    cpu_time_adpmm(k,iter) = cpu_time_adpmm(k,iter) + elapsed_time_adpmm;
                    if iter <= iter_adpmm
                        cpu_time_adpmm(k,iter+1) = cpu_time_adpmm(k,iter);
                    end
                end
                iter_adpmm = min(iter, iter_adpmm);

                % minimun
                min_among_all = min([min(F_val_soc), min(F_val_madmm), min(F_val_radmm), min(F_val_oadmm), min(F_val_aradmm)]);
                
                min_among_all = min([min_among_all, min(F_val_manpg)]);

                if min(F_val_adpmm) - 1e-10 < min_among_all
                    min_among_all = min(F_val_adpmm);
                end

                l = size(F_val_soc);
                for i=1:l(2)
                    F_val_soc(i) = F_val_soc(i) - min_among_all;
                end
                l = size(F_val_madmm);
                for i=1:l(2)
                    F_val_madmm(i) = F_val_madmm(i) - min_among_all;
                end
                l = size(F_val_radmm);
                for i=1:l(2)
                    F_val_radmm(i) = F_val_radmm(i) - min_among_all;
                end
                l = size(F_val_oadmm);
                for i=1:l(2)
                    F_val_oadmm(i) = F_val_oadmm(i) - min_among_all;
                end
                l = size(F_val_aradmm);
                for i=1:l(2)
                    F_val_aradmm(i) = F_val_aradmm(i) - min_among_all;
                end
                
                l = size(F_val_manpg);
                for i=1:l(2)
                    F_val_manpg(i) = F_val_manpg(i) - min_among_all;
                end

                l = size(F_val_adpmm);
                for i=1:l(2)
                    F_val_adpmm(i) = F_val_adpmm(i) - min_among_all;
                end

                F_val_soc_avg = F_val_soc_avg + F_val_soc ;
                F_val_madmm_avg = F_val_madmm_avg + F_val_madmm ;
                F_val_radmm_avg = F_val_radmm_avg + F_val_radmm ;
                F_val_oadmm_avg = F_val_oadmm_avg + F_val_oadmm ;
                F_val_aradmm_avg = F_val_aradmm_avg + F_val_aradmm ;
                
                F_val_manpg_avg = F_val_manpg_avg + F_val_manpg ;

                F_val_adpmm_avg = F_val_adpmm_avg + F_val_adpmm ;

                avg_min_among_all = avg_min_among_all + min_among_all;
            end
            avg_min_among_all = avg_min_among_all / avg;
            F_val_soc_avg = (F_val_soc_avg/avg);
            F_val_madmm_avg = (F_val_madmm_avg/avg);
            F_val_radmm_avg = (F_val_radmm_avg/avg);
            F_val_oadmm_avg = (F_val_oadmm_avg/avg);
            F_val_aradmm_avg = (F_val_aradmm_avg/avg);
            
            F_val_manpg_avg = (F_val_manpg_avg/avg);

            F_val_adpmm_avg = (F_val_adpmm_avg/avg);

            cpu_time_soc = sum(cpu_time_soc,1)/avg;
            cpu_time_madmm = sum(cpu_time_madmm,1)/avg;
            cpu_time_radmm = sum(cpu_time_radmm,1)/avg;
            cpu_time_oadmm = sum(cpu_time_oadmm,1)/avg;
            cpu_time_aradmm = sum(cpu_time_aradmm,1)/avg;
            cpu_time_manpg = sum(cpu_time_manpg,1)/avg;

            cpu_time_adpmm = sum(cpu_time_adpmm,1)/avg;

            %% Plots
            figure0 = figure(1);
            clf
            semilogy(F_val_soc_avg(1:iter_soc), '-.','LineWidth',2); hold on;
            semilogy(F_val_madmm_avg(1:iter_madmm), '-.','LineWidth',2); hold on;
            semilogy(F_val_radmm_avg(1:iter_radmm), '-.','LineWidth',2); hold on;
            semilogy(F_val_aradmm_avg(1:iter_aradmm),'LineWidth',2); hold on;
            semilogy(F_val_oadmm_avg(1:iter_oadmm),'LineWidth',2); hold on;
            
            semilogy(F_val_manpg_avg(1:iter_manpg), 'LineWidth',2); hold on;

            semilogy(F_val_adpmm_avg(1:iter_adpmm), 'LineWidth',2); hold on;

            xlabel('Iteration','interpreter','latex','FontSize',18); 
            ylabel('$f(x)-f^*$','interpreter','latex','FontSize',18);

            legend('SOC', 'MADMM', 'RADMM', 'ARADMM', 'OADMM', 'ManPG', 'ADPMM');
            legend('Location','best','FontSize',18);

            figure1 = figure(2);
            clf
            loglog(cpu_time_soc(1:iter_soc), F_val_soc_avg(1:iter_soc), '-.','LineWidth',2); hold on;
            loglog(cpu_time_madmm(1:iter_madmm), F_val_madmm_avg(1:iter_madmm), '-.','LineWidth',2); hold on;
            loglog(cpu_time_radmm(1:iter_radmm), F_val_radmm_avg(1:iter_radmm), '-.','LineWidth',2); hold on;
            loglog(cpu_time_aradmm(1:iter_aradmm), F_val_aradmm_avg(1:iter_aradmm),'LineWidth',2); hold on;
            loglog(cpu_time_oadmm(1:iter_oadmm), F_val_oadmm_avg(1:iter_oadmm),'LineWidth',2); hold on;
            loglog(cpu_time_manpg(1:iter_manpg), F_val_manpg_avg(1:iter_manpg), 'LineWidth',2); hold on;
            loglog(cpu_time_adpmm(1:iter_adpmm), F_val_adpmm_avg(1:iter_adpmm), 'LineWidth',2); hold on;

            xlabel('CPU time','interpreter','latex','FontSize',18); 
            ylabel('$f(x)-f^*$','interpreter','latex','FontSize',18);
            legend('SOC', 'MADMM', 'RADMM', 'ARADMM', 'OADMM', 'ManPG', 'ADPMM');
            legend('Location','best','FontSize',18);
        end
    end
end

function W = newtonschulz5(B, steps)
    arguments
        B
        steps (1,1) {mustBeInteger} = 10
    end

    [p, k] = size(B);
    nrmB = norm(B, 'fro');

    if nrmB < eps
        W = B;
        return;
    end

    W = B / nrmB;
    I = eye(k);

    for i = 1:steps
        WtW = W' * W;
        W = 0.5 * W * (3 * I - WtW);
    end
end
