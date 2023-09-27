close all
clear all
load("surface.mat")
load("test.mat")
ncoor = load("nors_MNI_3244xyz.txt","ascii");
ncoor = ncoor/1000;

%% Figure 
figure
scatter3(ncoor(:,1),ncoor(:,2),ncoor(:,3),'r');
hold on
scatter3(Vertices(:,1),Vertices(:,2),Vertices(:,3),'blue');
xlabel 'x'
ylabel 'y'
zlabel 'z'

%% rotation
zang = pi/2;
zrtate = [cos(zang) -sin(zang)  0; sin(zang) cos(zang) 0; 0 0 1];
Vertices = (zrtate*Vertices')';
xang = pi/30;
xrtate = [1 0  0; 0 cos(xang) -sin(xang); 0 sin(xang) cos(xang)];
Vertices = (xrtate*Vertices')';

%% traslation
Vertices(:,2) = Vertices(:,2) - (min(Vertices(:,2)) - min(ncoor(:,2)));
Vertices(:,2) = Vertices(:,2) - (max(Vertices(:,2)) - max(ncoor(:,2)))/2;
Vertices(:,3) = Vertices(:,3) - (max(Vertices(:,3)) - max(ncoor(:,3)));
Vertices(:,3) = Vertices(:,3) - (min(Vertices(:,3)) - min(ncoor(:,3)))/2;

%% Figure aligment
figure
scatter3(ncoor(:,1),ncoor(:,2),ncoor(:,3),'r');
hold on
scatter3(Vertices(:,1),Vertices(:,2),Vertices(:,3),'blue');
xlabel 'x'
ylabel 'y'
zlabel 'z'

%% Interpolation
metric = zeros(length(ncoor),48);
metric_interp = zeros(length(Vertices),48);
metric_list = {'FI','FD','Int','tcFILev1','tcFILev2','tcFDLev1','tcFDLev2','Age','Severity'}; 
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

mtname = metric_list{1};
mtname = strcat(mtname,'_interp');
metric_interp = tstatps.(mtname);
figure;
patch('Vertices',Vertices, ...
    'Faces',Faces, ...
    'FaceVertexCData',metric_interp(:,3), ...
    'FaceColor','inter', ...
    'EdgeColor','none')





