function [f,g] = evaluateRotatedKleeMintyProblem(x,problem)
    global target_flag
    global consumed
    
    y = problem.R*(x-problem.t);
    f = sum(problem.c.*x); %% y
    g = problem.A*y-problem.b;
    
    sz = 2*problem.dim;
    tz = length(g)./sz;
    gg = reshape(g,sz,tz);
     
    if sum(sum(gg<=0,1)==4)==1 
        g=g.*0;
    else
    end      
    
    if target_flag > 0
        consumed = consumed+1;
        logRotatedKleeMintyEvals(problem,f,g);
    end
    
end
