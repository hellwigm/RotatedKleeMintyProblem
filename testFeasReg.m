%% Empirically assess feasible regions
clear all, clc
problem=createRotatedKleeMintyCubeConstraintSystem(2,2)

xx=problem.dim^3 + [-4:0.01:4];
yy=problem.dim^3 + [-4:0.01:4];

for j=1:length(xx)
    for i=1:length(yy)
        x = [xx(i);yy(j)];
        [f,g] = evaluateRotatedKleeMintyProblem(x,problem);
        if sum(g) == 0;
            Z(j,i) = 0;
        else
            Z(j,i) = sum((g>0).*g);
        end
    end
end

figure
contour(xx,yy,Z,200)

% figure
% contour(xx,yy,Z,'g','LineWidth',1)
% hold on
% for j=1:10:length(xx)
%     for i=1:10:length(yy)
%         if Z(j,i)==1
%             plot(xx(i),yy(j),'go');
%         else
%             plot(xx(i),yy(j),'r+');
%         end
%     end
% end
