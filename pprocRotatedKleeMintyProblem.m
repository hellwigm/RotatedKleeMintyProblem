clear all;
clc;

foldername = [date '_rotKM_RS_comparison'];
efn = exist(foldername);
if efn ~= 7
  mkdir(foldername);
end

strategies = cell(0, 1);

strategy = struct();
strategy.name = 'RandS';
strategy.path = ...
fullfile('.', ...
         ['10-Jan-2019_linspaced_rotKM_RS_', ...
          strategy.name]);
strategies{end + 1, 1} = strategy;

nStrategies = size(strategies, 1);

dimensions = [2, 3, 5, 10, 20, 40];
nDimensions = length(dimensions);
for k = 1:nStrategies
  strategies{k, 1}.Eint = cell(nDimensions, 1);
  strategies{k, 1}.Etar = cell(nDimensions, 1);
  strategies{k, 1}.Stats = zeros(nDimensions, 15);
  for j = 1:nDimensions
    dim = dimensions(j);
    load(fullfile(strategies{k, 1}.path, ...
                  ['rotKM_', strategies{k, 1}.name, ...
                   '_Dim', num2str(dim), '.mat']));
    eval(['stats = StatsN', num2str(dim), ';']);
    strategies{k, 1}.Eint{j, 1} = ecdf_data.IntervalFevals ./ dim;
    strategies{k, 1}.Etar{j, 1} = ecdf_data.TargetsPerFevals ./ ...
                                  (1000 * problem.TarNum);
    strategies{k, 1}.Stats(j, :) = stats;
  end
end

problem.budget_multiplier=problem.budget/problem.dim;

for k = 1:nStrategies
  p1=figure;
  axes('Parent', p1, ...
       'Position', [0.15 0.13 0.75 0.70], ...
       'LineWidth', 1, ...
       'FontSize', 14);
  set(gcf,'color','w');
  set(gcf,'position',[1 600 800 600]);
  for j = 1:nDimensions
    dim = dimensions(j);
    dis  = 'DisplayName';
    name = ['N' num2str(dim)];
    plot(log(strategies{k, 1}.Eint{j, 1})/log(10),...
         strategies{k, 1}.Etar{j, 1},dis,name);
    hold on;
  end
  hleg = legend('show');
  legend boxoff;
  title(strategies{k, 1}.name);
  plot([log(problem.budget/dim)/log(10), ...
        log(problem.budget/dim)/log(10)],[0 1],'k','HandleVisibility','off')
  plot([0 ceil(log(problem.budget/dim)/log(10))+0.5],...
       [(problem.ConTarNum+1)/(problem.ConTarNum+problem.FunTarNum+1)...
                                (problem.ConTarNum+1)/...
        (problem.ConTarNum+problem.FunTarNum+1)],'--k',...
       'HandleVisibility','off');
  xlim([0 ceil(log(problem.budget/dim)/log(10))+0.5]);
  ylim([0 1]);
  xlabel('log(#fevals/dimension)');
  ylabel('target ratio');
  set(hleg,'FontSize',14,'Location','bestoutside');
  grid on;
  print(gcf, ...
        fullfile(foldername, ['ecdf_' strategies{k, 1}.name '.pdf']), ...
        '-dpdf','-r600','-bestfit');
end

p3=figure;
ax1=axes('Parent',p3, ...
         'Position',[0.15 0.13 0.75 0.70], ...
         'LineWidth',1,'FontSize',14);
set(gcf,'color','w');
set(gcf,'position',[1 600 800 600]);
for j = 1:nDimensions
  subplot(2,3,j);
  dim = dimensions(j);
  hold on;
  for k = 1:nStrategies
    dis1  = 'DisplayName';
    name1 = strategies{k, 1}.name;
    plot(log(strategies{k, 1}.Eint{j, 1})/log(10), ...
         strategies{k, 1}.Etar{j, 1},dis1,name1);
  end
  plot([log(problem.budget_multiplier)/log(10), ...
        log(problem.budget_multiplier)/log(10)],[0 1], ...
       'k','HandleVisibility','off');
  title(['N=' num2str(dim)]);
  plot([0 ceil(log(problem.budget_multiplier)/log(10))+0.5],...
       [(problem.ConTarNum+1)/(problem.ConTarNum+problem.FunTarNum+1)...
                                (problem.ConTarNum+1)/...
        (problem.ConTarNum+problem.FunTarNum+1)],'--k',...
       'HandleVisibility','off');
  xlim([0 ceil(log(problem.budget_multiplier)/log(10))+0.5]);
  ylim([0 1]);
  xlabel('log(#fevals/dimension)');
  ylabel('target ratio');
  grid on;
  hleg = legend('show');
  legend boxoff;
  set(hleg,'FontSize',8,'Location','southoutside');
end
print(gcf, ...
      fullfile(foldername, 'ecdf_comparison.pdf'), ...
      '-dpdf','-r600','-bestfit');

p4=figure;
ax2=axes('Parent',p4,...
         'Position',[0.15 0.13 0.75 0.70],...
         'LineWidth',1,'FontSize',16);
set(gcf,'color','w');
set(gcf,'position',[1 600 800 600]);
hold on;
for k = 1:nStrategies
  loglog(dimensions,abs(strategies{k, 1}.Stats(:,6)), ...
         'DisplayName',strategies{k, 1}.name,...
         'LineWidth',1,'MarkerSize',8,'LineWidth',1.5);
end
plot([1 50],[10^-8 10^-8],'--k','DisplayName','final target');
xlim([1.5 max(dimensions)+10]);
%ylim([10^-9 10^-0]);
xlabel('dimension');
ylabel('median obj. function error');
grid on;
hleg = legend('show');
legend boxoff;
set(hleg,'FontSize',16,'Location','NorthWest');
set(ax2,'FontSize',20,'MinorGridlines','none');
xticks(dimensions);
print(gcf,fullfile(foldername, 'medianFitError_comparison.pdf'), ...
      '-r600','-dpdf','-bestfit');

p3=figure;
ax3=axes('Parent',p3,...
         'Position',[0.15 0.13 0.75 0.70],'LineWidth',1,'FontSize',16);
set(gcf,'color','w');
set(gcf,'position',[1 600 800 600]);
hold on;
for k = 1:nStrategies
  errorbar(dimensions,strategies{k, 1}.Stats(1:length(dimensions),14),...
           strategies{k, 1}.Stats(1:length(dimensions),15),...
           'DisplayName', strategies{k, 1}.name,'MarkerSize',8,'LineWidth',1.5);
end
xlim([1.5 max(dimensions)+10]);
%ylim([10^-9 10^-3]);
xlabel('dimension');
ylabel('mean norm error');
grid on;
hleg = legend('show');
legend boxoff;
set(hleg,'FontSize',16,'Location','NorthWest');
set(ax3,'FontSize',20,'MinorGridlines','none');
xticks(dimensions);
%yticks([1e-9 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3]);
print(gcf,fullfile(foldername, 'meanNormError_comparison.pdf'), ...
      '-r600','-dpdf','-bestfit');

for k = 1:nStrategies
  filename = strcat(strategies{k, 1}.name, '_RS_on_RotKM_Table.txt');
  fileID = fopen(fullfile(foldername, filename),'w');
  fprintf(fileID, ...
          '%14s\t %14s\t %14s\t %14s\t %14s\t %14s\t %14s\t %14s\t %14s\n', ...
          'Dim','Fopt','Fbest','Fmed','\nu', ...
          'medianErr','FR','meanNormErr','Fevals');
  for j=1:length(dimensions)
    fprintf(fileID,['%14.8g\t %14.8g\t %14.8g\t %14.8g\t ', ...
                    '%14.8g\t %14.8g\t %14.8g\t %14.8g\t %14.8g\n'],...
            strategies{k, 1}.Stats(j,[1:7, 14, 10]));
  end
  fclose(fileID);
end

