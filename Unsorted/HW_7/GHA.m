% ghademo.m - generalized Hebbian learning demonstration
% call ghafun.m, randomize.m
% (C) 2001 by Yu Hen Hu
% created: 2/25/2001

clear all

% generate data samples
K=75; M=4; N = 4;

fileID = fopen('iris-train.txt');
C = textscan(fileID,'%n %n %n %n',...
'Delimiter','&', 'headerLines', 8);
fclose(fileID);
iris_train = cell2mat(C);

x = iris_train(1:2:end,:);
x = x - repmat(mean(x,1),75,1);

disp('the randomly generated covariance matrix is:')
C = x'*x;
[V,D]=eig(C);  % eigenvalue/eigenvector of covariance matrix
[Dsort,eidx]=sort(diag(-D)); % sort eigenvalue from largest to smallest
V=V(:,eidx); % sort eigenvector accordingly
yexact=x*V(:,1:N); % desireced output if exact principal components are there

% beginning of the algorithm
winit=randn(N,M)*0.05;   eta=0.0005;
itermax=10000;
w=ghafun(x,winit,itermax,eta,0);

% to compare with the principal components computed using eigenvalue
% decomposition, we first normalize each row of w:
disp(['first ' int2str(N) ' columns: true principal componenet vectors']);
disp(['last  ' int2str(N) ' columns: estimated principal component vectors'])
Trueest=[V(:,1:N) w']