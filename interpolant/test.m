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
figure
scatter3(ncoor(:,1),ncoor(:,2),ncoor(:,3),'r');
hold on
patch( 'Faces',            Faces, ...
    'Vertices',         Vertices,...
    'VertexNormals',    VertNormals, ...
    'VertexNormalsMode', 'auto',...
    'FaceNormals',      [],...
    'FaceNormalsMode',  'auto',...
    'FaceVertexCData',  [], ...
    'FaceColor',        [0.7 0.7 0.7], ...
    'FaceAlpha',        0.2, ...
    'AlphaDataMapping', 'none', ...
    'EdgeColor',        'none', ...
    'EdgeAlpha',        1, ...
    'LineWidth',         0.5,...
    'LineJoin',         'miter',...
    'AmbientStrength',  0.5, ...
    'DiffuseStrength',  0.5, ...
    'SpecularStrength', 0.2, ...
    'SpecularExponent', 1, ...
    'SpecularColorReflectance', 0.5, ...
    'FaceLighting',     'gouraud', ...
    'EdgeLighting',     'gouraud', ...
    'Tag',              'AnatSurface');
axis off;