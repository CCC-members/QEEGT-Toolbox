close all
clear all
load("bigbrain_MNI_327684xyz.mat")
ncoor = load("nors_MNI_3244xyz.txt","ascii");
ncoor = ncoor/1000;
%% Figure aligment
fig = figure;
scatter3(ncoor(:,1),ncoor(:,2),ncoor(:,3),'r','filled');
hold on
scatter3(Vertices(:,1),Vertices(:,2),Vertices(:,3),'blue','o','LineWidth',0.1);
xlabel 'x'
ylabel 'y'
zlabel 'z'
saveas(fig,'registration_before.fig');
close(fig)
%% rotation
zang = pi/2;
zrtate = [cos(zang) -sin(zang)  0; sin(zang) cos(zang) 0; 0 0 1];
Vertices = (zrtate*Vertices')';
VertNormals = (zrtate*VertNormals')';
xang = pi/30;
xrtate = [1 0  0; 0 cos(xang) -sin(xang); 0 sin(xang) cos(xang)];
Vertices = (xrtate*Vertices')';
VertNormals = (zrtate*VertNormals')';

%% traslation
Vertices(:,2) = Vertices(:,2) - (min(Vertices(:,2)) - min(ncoor(:,2)));
Vertices(:,2) = Vertices(:,2) - (max(Vertices(:,2)) - max(ncoor(:,2)))/2;
Vertices(:,3) = Vertices(:,3) - (max(Vertices(:,3)) - max(ncoor(:,3)));
Vertices(:,3) = Vertices(:,3) - (min(Vertices(:,3)) - min(ncoor(:,3)))/2;

%% Figure aligment
fig = figure;
scatter3(ncoor(:,1),ncoor(:,2),ncoor(:,3),'r','filled');
hold on
scatter3(Vertices(:,1),Vertices(:,2),Vertices(:,3),'blue','o','LineWidth',0.1);
xlabel 'x'
ylabel 'y'
zlabel 'z'
saveas(fig,'registration_after.fig');
close(fig)