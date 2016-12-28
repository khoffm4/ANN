%The Main function for HW 6
function Main
close all
rng(10)
%FOR QUESTION 2A UNCOMMENT THE FOLLOWING BLOCK

grid_size = 8;
grid_dim = 2;
n_iterations = 78000;
learning_rate = 0.5;
stdev = 5;
alpha = 20000; % stdev for the nieghborhood function
beta = 5000; %learning rate decreases learning rate

%Initialize weights
weights = rand(grid_dim,grid_size ^2) + 3.5*ones(grid_dim,grid_size ^2);

%Create the Data
Data = sqrt(0.1)*randn(grid_dim,4000);
Data=detrend(Data')';
Data(1,1:1000)=Data(1,1:1000) + 7*ones(1,1000);
Data(2,1:1000)=Data(2,1:1000) + 7*ones(1,1000);  
Data(1,1001:2000)=Data(1,1001:2000) + 7*ones(1,1000);
Data(2,2001:3000)=Data(2,2001:3000) + 7*ones(1,1000); 
Outputs = [];
%}

%FOR QUESTION 2B UNCOMMENT THE FOLLOWING BLOCK
%{
grid_size = 10;
grid_dim = 3;
n_iterations = 198000; 
learning_rate = 0.2;
stdev = 3;
alpha = 10000; % stdev for the nieghborhood function
beta = 40000; % learning rate decreases learning rate

%Initialize weights
weights = rand(grid_dim,grid_size ^2) + 3.5*ones(grid_dim,grid_size ^2);

%Create the Data
Data = sqrt(0.1)*randn(grid_dim,4000);
Data=detrend(Data')';
Data(1,1:1000)=Data(1,1:1000) + 7*ones(1,1000);
Data(2,1:1000)=Data(2,1:1000) + 7*ones(1,1000);  
Data(1,1001:2000)=Data(1,1001:2000) + 7*ones(1,1000);
Data(2,2001:3000)=Data(2,2001:3000) + 7*ones(1,1000); 
Outputs = [];
%}

%FOR IRIS DATA, UNCOMMENT THIS BLOCK
%Parameters
grid_size = 10;
grid_dim = 4;
n_iterations = 60000; 
learning_rate = 0.25;
stdev = 5;
alpha = 5000; % stdev for the nieghborhood function
beta = 5000; % learning rate decreases learning rate


%import the Iris Data
fileID = fopen('iris-train.txt');
C = textscan(fileID,'%n %n %n %n',...
'Delimiter','&', 'headerLines', 8);
fclose(fileID);
iris_train = cell2mat(C);

X = iris_train(1:2:end,:);
D = iris_train(2:2:end,:);
D = D(:, 2:4);

fileID = fopen('iris-test.txt');
C = textscan(fileID,'%n %n %n %n',...
'Delimiter','&', 'headerLines', 8);
fclose(fileID);
iris_test = cell2mat(C);
X_test = iris_test(1:2:end,:);
D_test = iris_test(2:2:end,:);
D_test = D_test(:, 2:4);

Data = [X; X_test]';
Outputs = [D; D_test]';

%Initialize weights
weights = rand(grid_dim,grid_size ^2)% + 3.5*ones(grid_dim,grid_size ^2);
%}

%Create the Grid
G = Grid_creator(grid_dim, grid_size);

%}

%Call thr SOM function
[weights] = SOM(G,Data, weights, n_iterations,  learning_rate,stdev, alpha, beta, grid_size, grid_dim);

%Plots
hold on
if grid_dim == 2
%2D plots 
plot(Data(1,:), Data(2,:), 'c*')
plot(weights(1,:), weights(2,:), 'r+', 'MarkerSize',10)
title('Weight vectors superimposed on Input Vectors')
elseif grid_dim == 3
%3D plots
hold on
plot3(Data(1,:), Data(2,:), Data(3,:), 'c*', 'MarkerSize',2)
plot3(weights(1,:), weights(2,:), weights(3,:), 'r+', 'MarkerSize',10)
title('Weight vectors superimposed on Input Vectors')
end


%Creates the lattis lines for the plot
lines(G,weights, grid_size, grid_dim)

%Create the Density Plot
Density(G, weights, grid_size, Data, grid_dim, Outputs)
title('Density Plot for Question 2a') 


end

%This function runs the SOM algorithmn.
function [weights] = SOM(G, Data, weights, n_iterations,o_learning_rate,o_stdev, alpha, beta , grid_size, grid_dim)
rng(10)

%initialize values
s_1 = size(weights);
s_2 = size(Data);
stdev = o_stdev;
learning_rate = o_learning_rate;
i = 0;



while  i <= n_iterations
    
%choose a datapoint

current_data = Data(:, randi([1,s_2(2)],1,1));
winner_v = Inf;
winner_i = Inf;
distances = [];
winner = zeros(1,length(weights));


    %Find the winner of current_data
    for j = 1: s_1(2)
        current_weight = weights(:,j);
        if sum(sum(sum(((current_weight - current_data).^2) .^0.5))) <= winner_v
            winner_v = sum((current_weight - current_data).^2 .^0.5);
            winner_i = j;
        end
    end

    
    %Update Neighbors of the point
    distance = mandist([G(:,winner_i)]', G); %Computes the manhatten distance from the winner
    distance = exp((-distance .^2)./(stdev^2));
    %distance = abs(distance .* stdev);
    weights = weights + learning_rate .* [ones(s_1(1),1) * distance] .*(current_data * ones(1,s_1(2)) - weights);
   
    %increment time
    i = i + 1;
    
    %Exponential decay for the Neighborhood function and learning rate
    %stdev = o_stdev * exp(-i/alpha);
    %learning_rate = o_learning_rate * exp(-i/beta);
    
    if mod(i,alpha) == 0
        stdev = stdev / 1.5;
    end
    
    if mod(i, beta) == 0 
        learning_rate = learning_rate /1.2;
    end
        
   %{
    if mod(i, 6000) == 0 
            figure
               hold on
        plot3(Data(1,:), Data(2,:), Data(3,:), 'c*', 'MarkerSize',2)
         plot3(weights(1,:), weights(2,:), weights(3,:), 'r+', 'MarkerSize',10)
%plot(Data(1,:), Data(2,:), 'c*')
%plot(weights(1,:), weights(2,:), 'r+', 'MarkerSize',10)
       lines(G,weights, grid_size, 3) 
        title(i) 
    end
        
       %} 
   
    
end
end

%Creates the Grids for the SOM
function [G] = Grid_creator(grid_dim, grid_size)
G = [];
if grid_dim == 2
    for i  = 1: grid_size
        current_x = i.*ones(1, grid_size);
        current_y = 1:grid_size;
        current_row = [current_x; current_y];
        G = [G,current_row]; 
    end
else 
    for i  = 1: grid_size
        current_x = i.*ones(1, grid_size);
        current_y = 1:grid_size;
        current_row = [current_x; current_y];
        G = [G,current_row]; 
    end
    G =  [G; zeros(1, length(G))];

end
end

%This function makes the Density plots, the U Matrix, and the color coding
%for the iris data
function Density(G, weights, grid_size, Data, grid_dim, Outputs)
winner = zeros(1,length(weights));
winner_v = Inf;
winner_i = Inf;
total = zeros(grid_size, grid_size);
dist = [];

%For each datapoint
for i = 1:length(Data)
    current_data = Data(:, i);
    
    %compare again each prototype
    for j = 1: length(weights)
        current_weight = weights(:,j);
        if sum(sum(((current_weight - current_data).^2) .^0.5)) <= winner_v
            winner_v = sum((current_weight - current_data).^2 .^0.5);
            winner(i) = j;
        end
    end
    winner_v = Inf;
end

    
Map = zeros(1,grid_size^2);
%count the number atrributed to each prototype
for j = 1: length(G)
   count = sum( winner ==j ) ;
   Map(j) = count;
end

if grid_dim == 2 
    vec = Map;
    Map = vec2mat(Map,grid_size);
    figure
    imagesc([0.5,7.5], [0.5,7.5],Map);
    colorbar;
    set(gca,'YDir','normal')
    colormap('hot')
elseif grid_dim == 3
    vec = Map;
    Map = vec2mat(Map,grid_size);
    figure
    colormap('hot')
    set(gca,'YDir','normal')
    imagesc([0.5,9.5], [0.5,9.5],Map);
    colorbar;
elseif grid_dim == 4
    vec = Map;
    Map = vec2mat(Map,grid_size);
    
end

%Making the U-Matrix
for i  = 1: length(G)-1
    current_weight = weights(:,i);
    if G(2,i) == grid_size
        top_weight = weights(:, i +grid_size);
        x = [NaN; norm(current_weight - top_weight)];
        dist = [dist, x];     
        continue
    elseif G(1,i) == grid_size
        right_weight = weights(:,i+1);
        x = [norm(current_weight - right_weight); NaN];
        dist = [dist, x]; 
        continue
    end
    
    right_weight = weights(:,i+1);
    top_weight = weights(:, i +grid_size);
    x = [norm(current_weight - right_weight); norm(current_weight - top_weight)];
    dist = [dist, x];     
end
maxi = max(max(dist));

dist = dist./ maxi ;

hold on
for i = 1: length(dist)
    if G(2,i) == grid_size
        color = dist(2,i); 
        x = mod(i-1,grid_size );
        y = floor(i/grid_size) ;
        line([x, x+1], [y,y],[1,1], 'color', [color, color, color], 'LineWidth',8);   
        continue
    elseif G(1,i) == grid_size
        color = dist(1,i); 
        x = mod(i,grid_size );
        y = floor(i/grid_size);
        line([x x],[y y+1],[1 1] , 'color', [color, color, color], 'LineWidth',8);
        continue
    end
    
    color = dist(1,i);
    x = mod(i,grid_size );
    y = floor(i/grid_size);
    line([x x],[y y+1],[1 1], 'color', [color, color, color], 'LineWidth',8);
    
    color = dist(2,i); 
    x = mod(i-1,grid_size );
    y = floor(i/grid_size) + 1;
    line([x, x+1], [y,y], [1,1], 'color', [color, color, color], 'LineWidth',8);
    
    i = i+1;
end
color = [];
%For the Iris Data, color the maximum

if grid_dim == 4
for i  = 1: length(G)
    nodes = find(winner==i);
    Setosa = 0;
    Versacolor = 0;
    Virginica = 0;
    %Find the types of Flower for each datapoint 
    for j  = 1: length(nodes)
        if Outputs(:,nodes(j)) == [1;0;0];
            Setosa = Setosa + 1;
        elseif Outputs(:,nodes(j)) == [0;1;0];
            Versacolor = Versacolor + 1;
        else
            Virginica = Virginica +1;
        end
            
    end
    
    %Color the Datapoints
    if max([Setosa, Versacolor, Virginica]) == 0
        color = [color,0];
    elseif max([Setosa, Versacolor, Virginica]) == Setosa
        color = [color, 1];
    elseif max([Setosa, Versacolor, Virginica]) == Versacolor
        color = [color, 2];
    else
        color = [color, 3];
    end  
end
Mat = vec2mat(color, 10);

fill([0,10,10,0],[0,0,10,10], 'k')
%{
for i = 1: length(Mat)
    for j = 1: length(Mat)
        if Mat(i,j) == 0
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'k')
        elseif Mat(i,j) == 1
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'r')
        elseif Mat(i,j) ==2 
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'g')
        else
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'b')
        end
    end
end
%}
title('U-matrix for the Iris Data');
end
%}
end


%This function makes the lines between the prototypes based on their
%relations in the SOM grid
function lines(G, weights, grid_size, grid_dim)
% If 2 Dimensional
if grid_dim == 2
    for i = 1: length(G)-1
        if mod(i,grid_size) == 0 
            continue
        end
       line([weights(1,i), weights(1,i+1)], [weights(2,i), weights(2,i+1)])
    end
    for i = 1: length(G)-grid_size
       line([weights(1,i), weights(1,i+grid_size)], [weights(2,i), weights(2,i+grid_size)])
    end

%If 3 Dimensional    
elseif grid_dim == 3
    for i = 1: length(G)-1
        if mod(i,grid_size) == 0 
            continue
        end
       line([weights(1,i), weights(1,i+1)], [weights(2,i), weights(2,i+1)], [weights(3,i), weights(3,i+1)])
    end
    
    for i = 1: length(G)-grid_size
        if mod(G(2,i),grid_size) == 0
            continue
        end
       line([weights(1,i), weights(1,i+grid_size)], [weights(2,i), weights(2,i+grid_size)], [weights(3,i), weights(3,i+grid_size)])
    end

    for i = 1:length(G) - grid_size^2
        line([weights(1,i), weights(1,i+grid_size^2)], [weights(2,i), weights(2,i+grid_size^2)], [weights(3,i), weights(3,i+grid_size^2)])
    end

end
end
