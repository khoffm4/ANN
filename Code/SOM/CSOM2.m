%This function runs the CSOM algorithmn.
function [weights] = CSOM2(G, Data, weights, n_iterations,o_learning_rate,o_stdev, alpha, beta , grid_size, grid_dim, plot_freq, Beta_bias, Gamma_bias)
rng(10)

%initialize values
s_1 = size(weights);
s_2 = size(Data);
learning_rate = o_learning_rate;
i = 0;

%Initialization for CSOM
if i == 0
    Count = zeros(s_1(2),1); %Number of times won. 
    Freq = zeros(s_1(2),1); %winning frequencey of each node
    b = Gamma_bias .* (1/(grid_size^2) * ones(s_1(2),1)- Freq) ;
    i = 1;
end

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
        if sum(sum(sum(((current_weight - current_data).^2) .^0.5)) ) - b(j) <= winner_v
            winner_v = sum((current_weight - current_data).^2 .^0.5) - b(j);
            winner_i = j;
        end
    end
    
    
    %update the winning Freq
    Count(winner_i) = Count(winner_i) + 1;
    Freq = Count / i;
    b = Gamma_bias .* (1/(grid_size^2) * ones(s_1(2),1)- Freq) ;

    
    %Update Neighbors of the point
    weights(:,winner_i) = weights(:,winner_i) + learning_rate .*(current_data  - weights(:,winner_i));
    %weights(:,winner_i) = weights(:,winner_i) + learning_rate .* ( current_data - weights(:,winner_i));
    
    
    %increment time
    i = i + 1;
 
    
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
         plot3(Data(1,:), Data(2,:), Data(3,:), 'c*', 'MarkerSize',2)
         plot3(weights(1,:), weights(2,:), weights(3,:), 'r+', 'MarkerSize',10)
         lines(G,weights, grid_size, 3) 
         title(i) 
       elseif grid_dim == 2
         plot(Data(1,:), Data(2,:), 'c*')
         plot(weights(1,:), weights(2,:), 'r+', 'MarkerSize',10)
         lines(G,weights, grid_size, 2) 
         title(i)   
       end
    end 
end
