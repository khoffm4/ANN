%This example takes the canoncial iris dataset and then
function Ex1
addpath(genpath('../Data'))
addpath(genpath('../Code/SOM'))
close all
hold on
rng(10)

grid_size = 10;
grid_dim = 4;
n_iterations =10000;
learning_rate = 1;
stdev = 2;
alpha = 6000; % stdev for the nieghborhood function
beta = 1000; %learning rate decreases learning rate
plot_freq = 10000;


%Create the Data
fileID = fopen('iris.txt');
C = textscan(fileID,'%n %n %n %n',...
'Delimiter','&');
fclose(fileID);
iris = cell2mat(C);

Data = iris(1:2:end,:);
Outputs = iris(2:2:end,:);
Outputs = Outputs(:, 2:4);
Outputs_tmp = Outputs(:, 1:3);

%Convert outputs from 1 in k encoding to integers
Outputs = [];
for i = 1: length(Outputs_tmp)
   if Outputs_tmp(i,:) == [1,0,0]
       Outputs = [Outputs; 0];
   elseif Outputs_tmp(i,:) == [0,1,0]
       Outputs = [Outputs; 1];
   else
       Outputs = [Outputs; 2];
   end
end

%Clear temporary variables
clear('Classes_tmp', 'Classes_tmp', 'C')


%%SOM
%Create the Grid
G = Grid_creator(grid_dim, grid_size);


%Initialize weights
weights = rand(grid_dim,grid_size ^2) + 3.5*ones(grid_dim,grid_size ^2);

%Call the SOM function
[weights] = SOM(G,Data, weights, n_iterations,  learning_rate,stdev, alpha, beta, grid_size, grid_dim, plot_freq);

%Create the M-U Matrix
figure
Density(G, weights, grid_size, Data, grid_dim, Outputs);
UMatrix(G, weights, grid_size, Data, grid_dim, Outputs)
title('Density Map for the Iris Data')
colorbar

%Create M-U Matrix with Classes
figure
[BMU, pic]=Density(G, weights, grid_size, Data, grid_dim, Outputs);
Color_Classes_Iris(G, weights, grid_size, Data, grid_dim, Outputs,BMU)
UMatrix(G, weights, grid_size, Data, grid_dim, Outputs)
title('Classification of Chain Link Data')



end

