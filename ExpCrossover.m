function [MUI]=ExpCrossover(D,NP,CR)
%%
% create a NPxD matrix for exponential crossover	
%
	MUI=zeros(D,NP);
	for l=1:NP
		k=1;
		j=randi(D);
		while rand < CR && k<=D
                MUI(j,l)=1;    
    			j=mod(j,D)+1;
			k=k+1;		
		end
	end
end
