load("test.mat")
metric = zeros(length(ncoor),48);
metric_interp = zeros(length(Vertices),48);
metric_list = {'FI','FD','Int','tcFILev1','tcFILev2','tcFDLev1','tcFDLev2','Age','Severity'}; 
%% Interpolation for tstatps
for met = 1:length(metric_list)
    mtname = metric_list{met};
    metric = extractfield(tstatps,mtname);
    metric = reshape(metric,length(ncoor),48);
    for freq = 1:48
        interp = scatteredInterpolant(ncoor,metric(:,freq));
        metric_interp(:,freq) = interp(Vertices);
        display(strcat("Interpolating metric ",mtname," frequency ",num2str(freq*0.39),'Hz'))
    end
    mtname = strcat(mtname,'_interp');
    tstatps.(mtname) = metric_interp;
end

%% Interpolation for pstatsps
for met = 1:length(metric_list)
    mtname = metric_list{met};
    metric = extractfield(pstatsps,mtname);
    metric = reshape(metric,length(ncoor),48);
    for freq = 1:48
        interp = scatteredInterpolant(ncoor,metric(:,freq));
        metric_interp(:,freq) = interp(Vertices);
        display(strcat("Interpolating metric ",mtname," frequency ",num2str(freq*0.39),'Hz'))
    end
    mtname = strcat(mtname,'_interp');
    pstatsps.(mtname) = metric_interp;
end
save('test.mat',pstatsps,tstatps,thrlev,thresholdsps)






