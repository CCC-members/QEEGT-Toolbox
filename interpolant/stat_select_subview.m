%%
%% Plotting stat by selected frequencies
%%
if(~isfolder('Figures')); mkdir('Figures');end
freq_range = 0.78:0.39:19.14;
delta = find(freq_range >= 0 & freq_range <= 4);      % delta 0 - 4 Hz
theta = find(freq_range >= 4 & freq_range <= 8);      % theta 4 - 8 Hz
alpha = find(freq_range >= 8 & freq_range <= 12.5);   % alpha 8 - 12.5 Hz
beta  = find(freq_range >= 12.5 & freq_range <= 20);  % beta 12.5 - 20 Hz
views = [90 0; -90 0; 90 90; -90 -90];
%% Figure stats
for met = 1:length(metric_list)
    mtname = metric_list{met};
    mtname = strcat(mtname,'_interp');
    metric_interp = tstatps.(mtname);
    %    
    select_freqs = [2 5 11 24 25 27 41 43 46 47];
    for freq = 1:size(metric_interp,2)
        if(~isempty(find(select_freqs == freq, 1)))
            if(~isempty(find(delta == freq, 1))); band = 'delta';
            elseif(~isempty(find(theta == freq, 1))); band = 'theta';
            elseif(~isempty(find(alpha == freq, 1))); band = 'alpha';
            else; band = 'beta';
            end
            figname = strcat(metric_list{met},'-',band,'(',num2str(freq_range(freq)),')');
            fig = figure("Name",figname,"NumberTitle","off","Color",'w','Position',[398 103 1300 900]);
            axis tight; 
            title(figname,"FontSize",16,"FontWeight","bold","Visible","on")
            metric_tmpe = metric_interp(:,freq);   
            for v=1:size(views,1)
                template                = load('axes.mat');
                currentAxes         = template.axes;
                patch(currentAxes,...
                    'Faces',Faces, ...
                    'Vertices',Vertices, ...
                    'FaceVertexCData',metric_tmpe, ...
                    'FaceColor','interp', ...
                    'EdgeColor','none', ...
                    'AlphaDataMapping', 'none', ...
                    'EdgeColor',        'none', ...
                    'EdgeAlpha',        1, ...
                    'BackfaceLighting', 'lit', ...
                    'AmbientStrength',  0.5, ...
                    'DiffuseStrength',  0.5, ...
                    'SpecularStrength', 0.2, ...
                    'SpecularExponent', 1, ...
                    'SpecularColorReflectance', 0.5, ...
                    'FaceLighting',     'gouraud', ...
                    'EdgeLighting',     'gouraud', ...
                    'FaceAlpha',.99);
                axis tight;
                axis off;
                rotate3d on;
                max_val = max(abs(metric_tmpe));
                currentAxes.CLim = [(-max_val-0.01) (max_val+0.01)];
                colormap(bipolar(201, 0.3))    
                axis(currentAxes,"tight");                
                currentAxes.View = views(v,:);               
                plot_tmpe = subplot(2,2,v,currentAxes);
            end            
            colorbar(currentAxes,'Position',[0.93 0.168 0.022 0.7]);           
            filename = strcat('Figure_',metric_list{met},'_',band,'_',num2str(freq_range(freq)));
            saveas(fig,fullfile('Outputs','Figures',strcat(filename,'.fig')));
            saveas(fig,fullfile('Outputs','Figures',strcat(filename,'.png')));
            close(fig);
        end
    end    
end