%The Main function for HW 6
function Main
addpath('IsomapR1')
addpath('SOM')
close all
hold on
rng(10)
%FOR QUESTION 2A UNCOMMENT THE FOLLOWING BLOCK

grid_size = 8;
grid_dim = 2;
n_iterations = 9000;
learning_rate = 0.5;
stdev = 10;
alpha = 2000; % stdev for the nieghborhood function
beta = 2000; %learning rate decreases learning rate
plot_freq = 1000;
beta_bias = 0.00001;
gamma_bias   = 20000;


filename = 'Compound.txt';
delimiterIn = '\t';
A = importdata(filename,delimiterIn);
Data = A(:,[1,2])
Outputs = A(:,3)


scatter(A(:,1), A(:,2),[], A(:,3))
title('Original Data')


%%%KNN;
k = 30;
%plot3(Data(1,:), Data(2,:), Data(3,:), 'c*', 'MarkerSize',2)
[idx,C] = kmeans(Data,k);
for i = 1:k
    plot(C(i,1), C(i,2), 'kx', 'Markersize', 50)
end

%%%%%%%
Data = Data';
%{
%%SOM
Data = Data';
%Create the Grid
G = Grid_creator(grid_dim, grid_size);


%Initialize weights
weights = rand(grid_dim,grid_size ^2) + 3.5*ones(grid_dim,grid_size ^2);

%Call the SOM function
hold on
[weights] = SOM(G,Data, weights, n_iterations,  learning_rate,stdev, alpha, beta, grid_size, grid_dim, plot_freq);
plot(Data(1,:), Data(2,:), 'c*', 'MarkerSize',2)
plot(weights(1,:), weights(2,:), 'r+', 'MarkerSize',10)
lines(G,weights, grid_size, grid_dim)
title('SOM')

%Create the Density Plot
figure
[winner, pic]=Density(G, weights, grid_size, Data, grid_dim, Outputs)
title('Density Plot for Chain Data') 
fig = figure;
ax = axes;
new_handle = copyobj(pic,ax);
set(gca,'YDir','normal')
colormap('gray')

%Create the M-U Matrix
UMatrix(G, weights, grid_size, Data, grid_dim, Outputs)
title('M-U Matrix for the Chain Data')
set(gca,'YDir','normal')
colormap('gray')
colorbar

%Create M-U Matrix with Classes
fig = figure;
ax = axes;
new_handle2 = copyobj(pic,ax);
set(gca,'YDir','normal')
colormap('gray')
UMatrix(G, weights, grid_size, Data, grid_dim, Outputs)
Color_Classes_Zahn(G, weights, grid_size, Data, grid_dim, Outputs,winner)
title('Classification of Chain Link Data')

%}


%%%%%%%%%%%%%%%%%
%Initialize weights
weights = rand(grid_dim,grid_size ^2)-0.5
weights = Grid_creator(grid_dim, grid_size);
weights = weights.* (1/8) - 0.5;
G = Grid_creator(grid_dim, grid_size);
figure
hold on
[weights, chosen,b] = CSOM(G,Data, weights, n_iterations,  learning_rate,stdev, alpha, beta, grid_size, grid_dim, plot_freq, beta_bias, gamma_bias);
plot(Data(1,:), Data(2,:), 'c*', 'MarkerSize',2)
plot(weights(1,:), weights(2,:), 'r+', 'MarkerSize',10)
lines(G,weights, grid_size, grid_dim)
title('CSOM')

%Create the Density Plot
figure
[winner, pic]=Density(G, weights, grid_size, Data, grid_dim, Outputs)
title('Density Plot for Chain Data') 
fig = figure;
ax = axes;
new_handle = copyobj(pic,ax);
set(gca,'YDir','normal')
colormap('gray')

%Create the M-U Matrix
UMatrix(G, weights, grid_size, Data, grid_dim, Outputs)
title('M-U Matrix for the Chain Data')
set(gca,'YDir','normal')
colormap('gray')
colorbar

%Create M-U Matrix with Classes
fig = figure;
ax = axes;
new_handle2 = copyobj(pic,ax);
set(gca,'YDir','normal')
colormap('gray')
UMatrix(G, weights, grid_size, Data, grid_dim, Outputs)
Color_Classes_Zahn(G, weights, grid_size, Data, grid_dim, Outputs,winner)
title('Classification of Chain Link Data')
%}

end

