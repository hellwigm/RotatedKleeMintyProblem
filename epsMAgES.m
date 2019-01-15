function [out,global_best]=epsMAgES(problem,budget,lower_bounds,upper_bounds,input)

    dim         = input.dim;
    sigma       = input.sigma;
    mu          = input.mu;
    lambda      = input.lambda;
    newpop.y    = zeros(dim,lambda);    % initialize new population matrix (n times NP)
	newpop.f    = 0;
	newpop.conv = 0;
	evals.fun   = 0;

    g           = 0;
    termination = 0;
    
    % Initialize dynamic (internal) strategy parameters and constants
    ps      = zeros(dim,1);                                 % evolution paths for sigma
    MM      = eye(dim);                                     % initial transformation matrix
    sqrt_s  = sqrt(input.cs*(2-input.cs)*input.mueff);      % factor in path update
    
    fOpt        = problem.fOpt;
    
%% HEMI     constraint_number = problem.gn(CEC_fun_no) + problem.hn(CEC_fun_no);
    % Initialize random population of lambda candidate solutions
    for k=1:lambda
        newpop.y(:,k)   =  lower_bounds...
                            +(upper_bounds- lower_bounds).*rand(dim,1); 
        [fval, gv ]     = feval(problem.Fname,newpop.y(:,k),problem);
        newpop.f(k)     = fval;                             % fitness vector 
        newpop.conv(k)  = sum(gv.*(gv>0));  %  constraint violations
		evals.fun       = evals.fun + 1; 
%         evals.con       = input.dim ; % count objective function evaluations
    end

     nic=1;
    % Initial parameter for epsilon level ordering
    TC=1000;                                                
    n=ceil(0.9*size(newpop.conv,2));                        
    index=eps_sort(newpop.f,newpop.conv,0);
    EPSILON= mean(newpop.conv(index(1:n))); %% HEMI %   0;                
    Epsilon= EPSILON;
    CP=max(3,(-5-log(EPSILON))/log(0.05));                  
    
    % Rank initial population 
    [ranking]           = eps_sort(newpop.f,newpop.conv,Epsilon);   % epsilon Ranking
    ParentPop           = newpop.y(:,ranking(1:mu));
    yParent             = sum(ParentPop,2)./mu;
    
    % Best individual of current population
    best_ind            = ranking(1);
    best_val            = newpop.f(best_ind);       % best fitness value of current population
    best_y              = newpop.y(:,best_ind);
    best_conv           = newpop.conv(:,best_ind);
          
    % Best solution found so far
    global_best.y       = best_y; 				% best solution found so far
	global_best.val     = best_val;
    global_best.conv    = best_conv;
    
    % Upper mutation strength bound
    sigmaMAX            = (max(problem.upper_bounds)-min(problem.lower_bounds))/2;
    flag1               = 0;
    flag2               = 0;
    
    while ~termination
        % Compute pseudo inverse of transformation matrix MM
        piM       = pinv(MM,1e-12);
        repi      = zeros(1,lambda);   
        
        % Sample lambda offspring distributed around yParent
        newpop.z  = randn(dim,lambda);
        newpop.d  = MM*newpop.z;
        newpopy   = repmat(yParent,1,lambda) + sigma.*newpop.d;
        
        for k=1:lambda 
                                        
            % ensure box constraint satisfaction 
            newpop.y(:,k)   = keep_range(newpopy(:,k),problem.lower_bounds,problem.upper_bounds);
            % log whether an offspring was repaired
            repi(k) = (sum(newpop.y(:,k)~=newpopy(:,k)) > 0);
            
            % evaluation  
            [fval, gv]      = feval(problem.Fname,newpop.y(:,k),problem);
            evals.fun       = evals.fun + 1;
            fitval          = fval;                                                                 % fitness vector 
            convio          = sum(gv.*(gv>0));% vector of the corresponding constraint violations)
            
            h=1;            
            % gradient-based repair step
            if mod(g,dim)==0 && rand(1) <=0.2
                while convio > 0 && h <= 3
                    new_mutant      = gradientMutation(problem,newpop.y(:,k),gv);
                    new_mutant      = keep_range(new_mutant,problem.lower_bounds,problem.upper_bounds);
                    [fval, gv]      = feval(problem.Fname,new_mutant,problem);
                    fitval          = fval;                                                                 % fitness vector 
                    convio          = sum(gv.*(gv>0));
                    evals.fun       = evals.fun + dim + 1;     % count function evaluations
                    if evals.fun>=budget              % check termination criterion
                        out = global_best.val;
                        break
                    end

                    h=h+1;
                    newpop.y(:,k) = new_mutant;
                    repi(k)= repi(k)+1;                     % log repaired offspring individuals
               end 
            end
            newpop.f(k)     = fitval;                                                                 % fitness vector 
            newpop.conv(k)  = convio;
            
            % Readjust mutation vectors:
            % iff repaired or components adjusted w.r.t box constraints
            if repi(k) > 0
                newpop.d(:,k) = (newpop.y(:,k)    - yParent)./sigma;
                newpop.z(:,k) = piM*newpop.d(:,k);
            end
        end
           
        % Rank current population w.r.t. recent epsilon level        
        [ranking]   = eps_sort(newpop.f,newpop.conv,Epsilon);
        
        % Best individual of current population
        best_ind    = ranking(1);
        best_val    = newpop.f(best_ind);            
        best_y      = newpop.y(:,best_ind);          
        best_conv   = newpop.conv(:,best_ind);
        
        % Recombination of mutation vectors and centroid update
        parent_z = newpop.z(:,ranking(1:mu)) * input.weights';  
        parent_d = newpop.d(:,ranking(1:mu)) * input.weights';  
        yParent  = yParent + sigma * parent_d; 
                        
        % Update evolution path and transformation matrix
        ps = (1-input.cs) * ps + sqrt_s * parent_z; 
        MM = (1 - 0.5*input.c1 - 0.5*input.cmu) * MM + (0.5*input.c1)*(MM*ps)*ps';
        for m = 1:mu
            MM = MM + ((0.5*input.cmu*input.weights(m))*newpop.d(:,ranking(m)))...
                *newpop.z(:,ranking(m))';
        end    
        
        % Regularization step to prevent unstable pseudo inverse calculations
        liMM = MM > 1e+12;
        siMM = MM < -1e+12;
        if sum(sum(liMM))>1 || sum(sum(siMM))>1
            MM = eye(input.dim);
	    ps = ones(input.dim,1);
        end
        
        % Adapt the mutation strength 
        sigma = min(sigma  * exp((input.cs/2)*(norm(ps)^2/input.dim - 1)),sigmaMAX); 
       
        % update best solution found so far                    
        if (best_conv==0 && global_best.conv==0 && best_val <= global_best.val) ||...
                (best_conv==global_best.conv && best_val <= global_best.val) || best_conv<global_best.conv
            global_best.y   = best_y; 				
            global_best.val = best_val;
            global_best.conv = best_conv;
            global_best.fevs = evals.fun;
                              nic=1;
        else
            nic=nic+1;
        end

         % termination
       if (abs(global_best.val-fOpt) <= 1e-8 && global_best.conv==0 ) || (nic > max(100,floor(budget/100))) || evals.fun>=budget %%%%%%%%%%%%%%%%%%
            termination = 1;
%             disp('Maximal budget of evaluations exceeded!')
        end
                
        % Update epsilon value 
        g=g+1;
        if(g>1 && g<TC)
          Epsilon=EPSILON*((1-g/TC)^CP);
        elseif(g+1>=TC)
          Epsilon=0;
        end   
        
     % log global best after having used 10%, and 50% of the evaluation budget
%         if evals.fun>=input.budget*10/100 && flag1==0
%             fit10=global_best.val;
%             con10=global_best.conv;
%             [ff,gg,hh]=feval(problem.constr_fun_name,global_best.y',CEC_fun_no);
%             c10_1    = sum(gg>1)                  + sum(abs(hh)>1);
%             c10_2    = sum((gg>0.01) & (gg<1))    + sum(abs(hh)>0.01 & abs(hh)<1);
%             c10_3    = sum((gg>0.0001)&(gg<0.01)) + sum(abs(hh)>0.0001 & abs(hh)<0.01);  
%             flag1=1;
%         elseif evals.fun>=input.budget*50/100 && flag2==0
%             fit50=global_best.val;
%             con50=global_best.conv;
%             [ff,gg,hh]=feval(problem.constr_fun_name,global_best.y',CEC_fun_no);
%             c50_1    = sum(gg>1)                  + sum(abs(hh)>1);
%             c50_2    = sum((gg>0.01)&(gg<1))      + sum(abs(hh)>0.01 & abs(hh)<1);
%             c50_3    = sum((gg>0.0001)&(gg<0.01)) + sum(abs(hh)>0.0001 & abs(hh)<0.01)  ;
%             flag2=1;
%         end

    end
    
    % log final global best solution
    fit100=global_best.val;
%     con100=global_best.conv;
%     [ff,gg,hh]=feval(problem.constr_fun_name,global_best.y',CEC_fun_no);
%     c100_1    = sum(gg>1)                   + sum(abs(hh)>1);
%     c100_2    = sum((gg>0.01)&(gg<1))       + sum(abs(hh)>0.01 & abs(hh)<1);
%     c100_3    = sum((gg>0.0001)&(gg<0.01))  + sum(abs(hh)>0.0001 &abs(hh)<0.01);

%     out = [fit10 con10 c10_1 c10_2 c10_3;
%              fit50 con50 c50_1 c50_2 c50_3;
      out =    fit100; %con100 c100_1 c100_2 c100_3];
    
end
