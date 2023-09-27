load("test.mat")
metric_interp = zeros(length(Vertices),48);
metric_list = {'Int'}; 

%% Interpolation
% loop metric
for met = 1:length(metric_list)
    mtname = metric_list{met};
    % test
    metric = extractfield(tstatps,mtname);
    metric = reshape(metric,length(ncoor),48);
    % loop frequencies
    for freq = 1:48
        % test
        metric_tmpe = metric(:,freq);
        interp = scatteredInterpolant(ncoor,metric_tmpe);
        metric_interp(:,freq) = interp(Vertices);
        display(strcat("Interpolating metric ",mtname," frequency ",num2str(freq*0.39),'Hz'))
    end
    mtname = strcat(mtname,'_interp');
    tstatps.(mtname) = metric_interp;
end
save('test.mat','pstatsps','tstatps','thrlev','thresholdsps')