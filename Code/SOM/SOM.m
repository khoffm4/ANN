%This function runs the SOM algorithmn.
function [weights] = SOM(G, Data, weights, n_iterations,o_learning_rate,o_stdev, alpha, beta , grid_size, grid_dim, plot_freq)
rng(10)

%initialize values
s_1 = size(weights);
s_2 = size(Data);
stdev = o_stdev;
learning_rate = o_learning_rate;
i = 0;



while  i <= n_iterations
    
%choose a datapoint
current_data = Data(randi([1,s_2(1)],1,1), :);
current_data = current_data';
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
        
   
    if mod(i, plot_freq) == 0 
            figure
            hold on
       if grid_dim == 3
         plot3(Data(:,1), Data(:,2), Data(:,3), 'c*', 'MarkerSize',2)
         plot3(weights(1,:), weights(2,:), weights(3,:), 'r+', 'MarkerSize',10)
         lines(G,weights, grid_size, 3) 
         title(i) 
       else 
         plot(Data(:,1), Data(:,2), 'c*')
         plot(weights(1,:), weights(2,:), 'r+', 'MarkerSize',10)
         lines(G,weights, grid_size, 2) 
         title(i)   
       end
           
           
               
               



    end
        
        
   
    
end
