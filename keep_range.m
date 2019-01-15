function [ui_y]=keep_range(ui_y,lower_bounds,upper_bounds)
bwidth =  upper_bounds -  lower_bounds;         % vector of box widths
n=length(ui_y);                                 % dimenson
	for j=1:n
        if ui_y(j)< lower_bounds(j)                                     % component deceeds box
            exceed =  lower_bounds(j)-ui_y(j);  
			if exceed >= bwidth(j)
			    exceed = exceed - floor(exceed/bwidth(j))*bwidth(j);    % determine reflection width
			end
		    ui_y(j) =  lower_bounds(j) + exceed;                        % repair step
		    
		elseif ui_y(j) >  upper_bounds(j)                               % component exceeds box
		    exceed = ui_y(j)-upper_bounds(j);
			if exceed >= bwidth(j)
			    exceed = exceed - floor(exceed/bwidth(j))*bwidth(j);    % determine reflection width
			end
		    ui_y(j) = upper_bounds(j) - exceed;                         % repair step
		    
		end       
	end
end
