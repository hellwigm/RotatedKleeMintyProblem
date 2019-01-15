function [out,global_best, dyn]=epsDEga3(problem,budget,lower_bounds,upper_bounds,input)

    NP          = input.population_size;			% population size
    n           = input.dim;                    % senewpop space parameter dimension
    newpop.y    = zeros(n,NP);					% initialize new population matrix (n times NP)
    M           = 100*n;
	newpop.f    = 0;
	newpop.conv = 0;
	evals.fun   = 0;
	
    g           = 0;
    termination = 0;
    Pg          = 0.2;
    Rg          = 3;                            % number of consecutive gradient approximations
                               % initial epsilon adaptation parameter
    CR          = 0.9;
	TC          = 1000;
    
    % Initialize population uniformly distributed within the bounds
    

        newpop.y   =  lower_bounds...
                            +( upper_bounds- lower_bounds).*rand(n,M);  
    for k=1:M
        % population of uniformly distributed vectors [0,1]^n 
        [Aval, Agv]    = feval(problem.Fname,newpop.y(:,k),problem);
        newpop.f(k)     = Aval;	  % fitness vector 
        newpop.conv(k)  = sum(Agv'*(Agv>0));                  % vector of the corresponding constraint violations)
    end
    evals.sum=M;
    
    nn=ceil(0.9*length(newpop.conv));                 % initial epsilon value determination
    lex_rank=eps_sort(newpop.f,newpop.conv,0);
    EPSILON=mean(newpop.conv(lex_rank(1:nn)));        %%%%%%%%%%%%%%%%%%%%%%%%%%% MEAN constraint violation of the best 90 % individuals
    Epsilon=EPSILON;
    [ranking]=eps_sort(newpop.f,newpop.conv,Epsilon);
    
    cp = max(3,(-5-log(EPSILON))/log(0.05));
    
    best_ind    = ranking(1);
    best_val    = newpop.f(best_ind);				% best fitness value of current population
    best_y      = newpop.y(:,best_ind);
    best_conv   = newpop.conv(:,best_ind);
    
    newpop.y    = newpop.y(:,ranking);
    newpop.f    = newpop.f(:,ranking);
    newpop.conv = newpop.conv(:,ranking);
    
    global_best.y       = best_y; 				% best solution found so far
    global_best.val     = best_val;
    global_best.conv    = best_conv;

%     dyn.gen(g+1)        = g;
%     dyn.fev(g+1)        = evals.sum;
%     dyn.fit(g+1)        = global_best.val;
%     dyn.conv(g+1)       = global_best.conv;
%     dyn.F(g+1)          = input.F;
%     dyn.Epsilon(g+1) =    EPSILON;
    
   % [inertia,centroid]  = PopDivMoi(newpop.y);
   % dyn.PopInertia(g+1) = inertia;
   % dyn.PopMean(:,g+1)  = centroid;    
    
    ll=0;
    lll=0;
     fOpt        = problem.fOpt;
     
    nic=1; 
    while termination ~= 1
			
	
        df=input.F;
        if (g+1 >= 0.95*TC) && (g+1 <TC)
            cp = max(3,0.3*cp+0.7*3);
            dF = 0.3*input.F + 0.7;
        end
        if rand<0.05
                dF=min(1+abs(0.05*randn(1)),1.1);
        else
                dF=input.F;
        end
        iseq=ones(1,NP);
        seq = [1:NP];
        for k=1:2
            
            seq=seq(iseq==1);
%             length(seq)
            for jj=1:length(seq)
                
                l=seq(jj);
                
                idx    = randperm(NP,3);
                target.y = newpop.y(:,l);
                target.f = newpop.f(l);
                target.c = newpop.conv(l);
                
                xp1=newpop.y(:,idx(1));
                xp2=newpop.y(:,idx(2));
                
                if rand<0.05
                    pa  = idx(3);
                else
                    if rand < NP/M
                        rmn = 0;
                    else
                        rmn = max(NP,randi(M-NP));
                    end
                    pa  = idx(3)+rmn;
                end    
                xp3=newpop.y(:,pa);
                
                % generate the base and difference vectors according to the DE/rand/1/bin

                base_y = xp1;
                difA = xp2 - xp3;

                % and apply mutation
                ui_y = base_y + dF*difA;

                % recombination of ui_pop and oldpop via binary crossover 
                mui = ExpCrossover(n,1,input.CR);
                mop = mui < 0.5;
                ui_y = mop.*target.y + mui.*ui_y;
                
                % First check ui_pop for bound constraint violations
                for j=1:n
                    if ui_y(j) >  upper_bounds(j)
                        exceed = ui_y(j)- upper_bounds(j);
                        if exceed >=  upper_bounds(j) -  lower_bounds(j)
                            exceed = exceed - floor(exceed/( upper_bounds(j) -  lower_bounds(j)))*( upper_bounds(j) -  lower_bounds(j));
                        end
                        ui_y(j) =  upper_bounds(j) - exceed; %%%%%%%%%%%%%% reflection %%%%%%%%%%% !!!
                    elseif ui_y(j) <  lower_bounds(j)
                        exceed =  lower_bounds(j)-ui_y(j);
                        if exceed >=  upper_bounds(j) -  lower_bounds(j)                       
                            exceed = exceed - floor(exceed/( upper_bounds(j) -  lower_bounds(j)))*( upper_bounds(j) -  lower_bounds(j));
                        end
                        ui_y(j) =  lower_bounds(j) + exceed; %%%%%%%%%%%%%% reflection %%%%%%%%%%% !!!
                    end
                end
                
                % Gradient repair
                if rem(g+1,n)==0 && rand < Pg

                    h=1;
                    [ui_val, ui_g ]= feval(problem.Fname,ui_y,problem);;
                    ui_conv             = sum(ui_g'*(ui_g>0)); 
                    evals.fun               = evals.fun + 1;
                    
                    nusc                =sum(ui_g>0)>10^-4; % skip move
                    
                    if nusc==1 && rand <0.5
                        
                    else
                        while (h<=Rg && ui_conv>0)
                            ui_y=gradientMutation(problem,ui_y,ui_g);
                            for j=1:n
                                if ui_y(j)<lower_bounds(j)
                                    exceed = lower_bounds(j)-ui_y(j);
                                        if exceed >= upper_bounds(j) - lower_bounds(j)                       
                                            exceed = exceed - floor(exceed/(upper_bounds(j) - lower_bounds(j)))*(upper_bounds(j) - lower_bounds(j));
                                        end
                                    ui_y(j) = problem.lower_bounds(j) + exceed;
                                 
                                elseif ui_y(j) > upper_bounds(j)
                                    exceed = ui_y(j)-upper_bounds(j);
                                        if exceed >= upper_bounds(j) -lower_bounds(j)
                                            exceed = exceed - floor(exceed/(upper_bounds(j) - lower_bounds(j)))*(upper_bounds(j) - lower_bounds(j));
                                        end
                                    ui_y(j) = upper_bounds(j) - exceed; 
                                  
                                end       
                            end
                            h=h+1;
                    [ui_val, ui_g ]  		    = feval(problem.Fname,ui_y,problem);
                            ui_conv                 = sum(ui_g'*(ui_g>0));                  % vector of the corresponding constraint violations)
                            
                            evals.fun               = evals.fun + n +1;
                        end
                    end
         
                else % no gradient repair
                    
                    [ui_val, ui_g ]  		    = feval(problem.Fname,ui_y,problem);
                            ui_conv                 = sum(ui_g'*(ui_g>0)); 
                    evals.fun           = evals.fun +1;
                end
                
                %% Implementation of epsilon constraint ranking (EC)
                % EPSILON feasible solutions are compared based on their
                % fitness values
                if (ui_conv <= Epsilon && target.c <= Epsilon) || (ui_conv == target.c)
                    if ui_val < target.f
                        newpop.y(:,l) = ui_y;
                        newpop.f(l)   = ui_val;
                        newpop.conv(l) = ui_conv;
                        iseq(jj)=0;
                    else
                        RI=NP+randi(M-NP);
                        newpop.y(:,RI)  = ui_y;
                        newpop.f(RI)    = ui_val;
                        newpop.conv(RI) = ui_conv;    
                        iseq(jj)=1;
                    end
                % non-EPSILON feasible solutions are compared based on their
                % constraint violations
                else
                    if ui_conv < target.c
                        newpop.y(:,l)  = ui_y;
                        newpop.f(l)    = ui_val;
                        newpop.conv(l) = ui_conv;
                        iseq(jj)=0;
                    else
                        RI=NP+randi(M-NP);
                        newpop.y(:,RI)  = ui_y;
                        newpop.f(RI)    = ui_val;
                        newpop.conv(RI) = ui_conv;
                        iseq(jj)=1;
                    end
                end
                           
            end            
        end
    
        ranking  = eps_sort(newpop.f,newpop.conv,Epsilon);
        best_ind = ranking(1);
        best_y   = newpop.y(:,best_ind);
        best_conv= newpop.conv(best_ind);
        best_val = newpop.f(best_ind);
        
        if (best_conv<=Epsilon && global_best.conv<=Epsilon && best_val < global_best.val) ||...
                (best_conv==global_best.conv && best_val < global_best.val) || best_conv<global_best.conv
            global_best.y   = best_y; 				% best solution found so far
            global_best.val = best_val;
            global_best.conv = best_conv;
                                   nic=1;
        else
            nic=nic+1;
        end
        evals.sum = evals.fun;
        
         if (abs(global_best.val-fOpt) <= 1e-8 && global_best.conv==0 ) || (nic > max(100,floor(budget/100))) || evals.fun>=budget %%%%%%%%%%%%%%%%%%
            termination = 1;
%             disp('Maximal budget of evaluations exceeded!')
        end
        g=g+1;
        
        if g<TC           
            Epsilon=EPSILON*((1-(g)/TC)^cp);
        else 
            Epsilon=0;
        end
%         dyn.gen(g+1)        = g;
%         dyn.fev(g+1)        = evals.sum;
%         dyn.fit(g+1)        = global_best.val;
%         dyn.conv(g+1)       = global_best.conv;
%         dyn.F(g+1)          = dF;
%         dyn.Epsilon(g+1)    = Epsilon;
    
      %  [inertia,centroid]  = PopDivMoi(newpop.y);
      %  dyn.PopInertia(g+1) = inertia;
      %  dyn.PopMean(:,g+1)  = centroid; 
        
   %     if evals.sum>=input.budget*10/100 && ll==0
   %         fit10=global_best.val;
   %         con10=global_best.conv;
   %         [ff,gg,hh]=feval(problem.constr_fun_name,global_best.y',CEC_fun_no);
            
   %             c10_1    = sum(gg>1) + sum(abs(hh)>1);
   %             c10_2    = sum((gg>0.01) & (gg<1)) + sum(abs(hh)>0.01 & abs(hh)<1);
   %             c10_3    = sum((gg>0.0001)&(gg<0.01)) + sum(abs(hh)>0.0001 & abs(hh)<0.01);          
   %         ll=1;
   %     elseif evals.sum>=input.budget*50/100 && lll==0
   %         fit50=global_best.val;
   %         con50=global_best.conv;
  %         [ff,gg,hh]=feval(problem.constr_fun_name,global_best.y',CEC_fun_no);
   %           
   %             c50_1    = sum(gg>1) + sum(abs(hh)>1);
   %             c50_2    = sum((gg>0.01)&(gg<1)) + sum(abs(hh)>0.01 & abs(hh)<1);
   %             c50_3    = sum((gg>0.0001)&(gg<0.01)) + sum(abs(hh)>0.0001 & abs(hh)<0.01)  ;
   %         lll=1;
   %     end 
        
    end
    
    fit100=global_best.val;
    %con100=global_best.conv;
    
   % [ff,gg,hh]=feval(problem.constr_fun_name,global_best.y',CEC_fun_no);
    %    c100_1    = sum(gg>1) + sum(abs(hh)>1);
    %    c100_2    = sum((gg>0.01)&(gg<1)) + sum(abs(hh)>0.01 & abs(hh)<1);
     %   c100_3    = sum((gg>0.0001)&(gg<0.01)) + sum(abs(hh)>0.0001 &abs(hh)<0.01);
    
       
    out= fit100;%[fit10 con10 c10_1 c10_2 c10_3; fit50 con50 c50_1 c50_2 c50_3; fit100, con100 c100_1 c100_2 c100_3];
    
end
