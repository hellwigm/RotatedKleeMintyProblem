%  experimental setup

clear all, clc

% 2,3,5,10,20,40
DIMENSION=[2,3,5,10,20,40];


global target_flag
global T
global Targets
global consumed 
global best

for j=1:length(DIMENSION)
    target_flag = 0;
        
    %% USER specified budget of constrained function evaluations
    problem.budget_multiplier   = 2*10^4;
        
    % create the rotated Klee-Minty problem in dimension 'dim'
    dim                 = DIMENSION(j);
    budget              = dim*problem.budget_multiplier;
    problem             = createRotatedKleeMintyCubeConstraintSystem(dim);
    problem.budget      = budget;

    input.dim = dim;

    %% USER specified INPUT parameter of the individual SOLVER
    % specify SOLVER
    input.strategy      = 'RandS';
    
    %% start EXPERIMENTAL runs
    number_of_runs = 15;
    ListT       = [];
    for k=1:number_of_runs
        k
        target_flag = 1;
        
        best        = [];
        consumed    = 0;
        eval(['[out, global_best]=' input.strategy '(problem,problem.budget,problem.lower_bounds,problem.upper_bounds,input);']);
        dyn.fev = [consumed];
        dyn.fit = [global_best.val];
        dyn.conv = [global_best.conv];
        List(k,:)=[dim, global_best.val, global_best.conv, norm(global_best.y-problem.t), dyn.fev(end)];
        
        GB{k}=global_best.y; 
        eval(['Dyn' num2str(dim) '{k}=dyn;']);
        ListT{k}=T;
        ListTargets{k}=Targets;
        Dyn{k}=dyn;
        if k==1
            FEperTarget=Targets(:,1:3);
        else
            FEperTarget=[FEperTarget,Targets(:,3)];
        end
            
    end
    
    %[ecdf_data] = assessRotatedKleeMintyPerformance(problem,number_of_runs,FEperTarget,ListTargets);
    
    rankingL=lex_sort(List(:,2),List(:,3));
    
    besti = rankingL(1);
    worsti= rankingL(15);
    cmedi = rankingL(8);
    med   = List(cmedi,2);
    cmed  = List(cmedi,3);
    
    nmed  = List(cmedi,4);
    nbest = List(besti,4);
    
    median_y = GB{cmedi};
    ncon  = sum(List(:,3 )== 0);
    FR = ncon/15;
    FOPT = problem.fOpt;
    %Stats(:,j) = [dim  FOPT List(besti,2) med cmed abs(FOPT-med) FR mean(List(:,2)) mean(List(:,3)) mean(List(:,5)) ecdf_data.aRT/dim nbest nmed mean(List(:,4)) std(List(:,4))];
    Stats(:,j) = [dim  FOPT List(besti,2) med cmed abs(FOPT-med) FR mean(List(:,2)) mean(List(:,3)) mean(List(:,5)) 0 nbest nmed mean(List(:,4)) std(List(:,4))];
        
    %eval(['StatsN' num2str(dim) '= [dim  FOPT List(besti,2) med cmed (FOPT-med) FR mean(List(:,2)) mean(List(:,3)) mean(List(:,5)) ecdf_data.aRT/dim nbest nmed mean(List(:,4)) std(List(:,4))]']);
    eval(['StatsN' num2str(dim) '= [dim  FOPT List(besti,2) med cmed (FOPT-med) FR mean(List(:,2)) mean(List(:,3)) mean(List(:,5)) 0 nbest nmed mean(List(:,4)) std(List(:,4))]']);
    
    for k=1:15
        LL(k)=length(Dyn{k}.fev);
    end
    len=min(LL);
    
    FES(:,j) = Dyn{1}.fev(1:len);
    FIT(:,j) = Dyn{1}.fit(1:len);
    CON(:,j) = Dyn{1}.conv(1:len);
    for k=2:15
        FES(:,j) = FES(:,j) + Dyn{k}.fev(1:len)';
        FIT(:,j) = FIT(:,j) + Dyn{k}.fit(1:len)';
        CON(:,j) = CON(:,j) + Dyn{k}.conv(1:len)';
    end
    FES(:,j) = FES(:,j) ./ 15;
    FIT(:,j) = FIT(:,j) ./ 15;
    CON(:,j) = CON(:,j) ./ 15;

    problem.number_of_runs = 1000;
    FEperTargetBootstrapped = bootstrap(problem, ...
                                        FEperTarget, ...
                                        problem.number_of_runs, ...
                                        1000, ...
                                        problem.budget);
    ecdf_data = assessRotatedKleeMintyPerformance(problem, ...
                                                  FEperTargetBootstrapped);

    foldername = [date '_linspaced_rotKM_RS_' input.strategy];
        efn = exist(foldername);
        if efn ~= 7
            mkdir(foldername);
        end
    save([foldername '/rotKM_' input.strategy '_Dim' num2str(dim) '.mat'],'problem','input',['Dyn' num2str(dim)],'FES','FIT','CON','ListT','ListTargets','ecdf_data',['StatsN' num2str(dim)] ,'-v7')
    
     clear FES
     clear FIT
     clear CON
end

 %save(['2018_RotationAddC_KleeMinty_' input.strategy 'Rep.mat'],'problem','input','Stats','-v7')
