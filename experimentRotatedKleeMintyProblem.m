%% Example experimental 
%
% Benchmarking Random Search (RS) on the Rotated Klee-Minty Problem
%
  clear all, clc
%

% Definition of global variable names for performance logging 
global target_flag
global T
global Targets
global consumed 
global best

% Range of problem dimensions [default: 2, 3, 5, 10, 20, 40 , (100)]
DIMENSION=[3]; %,3,5,10,20,40];

for j=1:length(DIMENSION)
    
    %% Problem initialization
    target_flag         = 0;
    ListT               = [];
    input.dim           = DIMENSION(j);
    
    % Creation of the Rotated Klee-Minty Problem in dimension 'input.dim'
    problem             = createRotatedKleeMintyCubeConstraintSystem(input.dim,2);
      
    %% USER specifications
  
    % Specify the number of independent runs per constrained problem
    problem.number_of_runs = 3;  % default #runs = 15
    
    % Set budget_multiplier 
    % Determintes the maximal number of constrained function evaluations
    % DEFAULT: budget = budget_multiplier * dimension
    problem.budget_multiplier   = 2*10^4;   % default: 2*10^4
    problem.budget              = input.dim*problem.budget_multiplier; 
    
    % Name of the solver to be benchmarked
    input.strategy      = 'RandS';
        
    for k=1:problem.number_of_runs
        target_flag     = 1;
        best            = [];
        consumed        = 0;
        eval(['[out, global_best]=' input.strategy '(problem,problem.budget,problem.lower_bounds,problem.upper_bounds,input);']);
        List(k,:)       = [input.dim, global_best.val, global_best.conv, norm(global_best.y-problem.t) consumed];
        GB{k}           = global_best;
        ListT{k}        = T;
        if k==1
            FEperTarget = Targets(:,1:3);
        else
            FEperTarget = [FEperTarget,Targets(:,3)];
        end
            
    end

    % Gather statistics for algorithm assessment
    rankingL    = lex_sort(List(:,2),List(:,3));
    
    besti       = rankingL(1);
    worsti      = rankingL(problem.number_of_runs);
    cmedi       = rankingL(floor((problem.number_of_runs+1)/2));
    med         = List(cmedi,2);
    cmed        = List(cmedi,3);
    
    if List(cmedi,3) == 0
        nmed  = List(cmedi,4);
    else
        nmed  = NaN;
    end
    if List(besti,3) == 0
        nbest = List(besti,4);
    else
        nbest = NaN;
    end
    
    median_y    = GB{cmedi};
    ncon        = sum(List(:,3 )== 0);
    FR          = ncon/problem.number_of_runs;
    FOPT        = problem.fOpt;
    
    if FR==1
        nmean = mean(List(:,4)); 
        nstd  = std(List(:,4));
    elseif FR == 0
        nmean = NaN; 
        nstd  = NaN;
    else
        idf = find(List(:,3)==0);
        nmean = mean(List(idf,4)); 
        nstd  = std(List(idf,4));
    end
    
    % Vector of gathered statistics from all independent runs
    eval(['StatsN' num2str(input.dim) '= [input.dim  FOPT List(besti,2) med cmed (FOPT-med) FR mean(List(:,4)) mean(List(:,5)) std(List(:,4))];']);

    problem.number_of_runs = 1000;
    FEperTargetBootstrapped = bootstrap(problem, ...
                                        FEperTarget, ...
                                        problem.number_of_runs, ...
                                        1000, ...
                                        problem.budget);

    % Derive ECDF data from all observed runs
    ecdf_data{j} = assessRotatedKleeMintyPerformance(problem, ...
                                                     FEperTargetBootstrapped);


    clear FES
    clear FIT
    clear CON
end

% Create ECDF plots and Tables according to the recommendations
pprocRotatedKleeMintyProblem()
