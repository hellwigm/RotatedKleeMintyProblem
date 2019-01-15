function [y]=gradientMutation(problem,x,gg)
    eta     = 1e-4;                         % hard coded deviation for determining finite differences               
    n       = length(x);
    Dd      = eye(n);
    dx      = repmat(x,1,n)+eta.*Dd;
    for i=1:n
        [fff, gv] = feval(problem.Fname,dx(:,i),problem);
	    dCx(:,i)  = gv;
    end
    deltaG  = max(0,gg);
    Cx= [gg];

    nabC    = 1/eta.*( dCx - repmat(Cx,1,n));
    delC    = [deltaG];
    inv_nabC= pinv(nabC,1e-12);            % Moore-Penrose inverse of nabC
    deltaX  = -inv_nabC*delC;
    y       = (x+deltaX);
end
