load("bigbrain_MNI_327684xyz.mat")
ncoor = load("nors_MNI_3244xyz.txt","ascii");
ncoor = ncoor/1000;
%% Figure aligment
figure
scatter3(ncoor(:,1),ncoor(:,2),ncoor(:,3),'r','filled');
hold on
scatter3(Vertices(:,1),Vertices(:,2),Vertices(:,3),'blue','o','LineWidth',0.1);
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
fig = figure
% scatter3(ncoor(:,1),ncoor(:,2),ncoor(:,3),'r');
template                = load('axes.mat');
currentAxes         = template.axes;

patch(currentAxes,...
    'Faces',Faces, ...
    'Vertices',Vertices, ...
    'FaceVertexCData',SulciMap*0.06, ...
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
axis off;
rotate3d on;
set(currentAxes,"Parent",fig);


%% Get bands
freqs = 0.78:0.39:19.14;
delta = find(freqs >= 0 & freqs <= 4);      % delta 0 - 4 Hz
theta = find(freqs >= 4 & freqs <= 8);      % theta 4 - 8 Hz
alpha = find(freqs >= 8 & freqs <= 12.5);   % alpha 8 - 12.5 Hz
beta  = find(freqs >= 12.5 & freqs <= 20);  % beta 12.5 - 20 Hz

