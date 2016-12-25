%Makes a Conscious SOM
function [weights, error_list, learning_rate] = C_SOM(G,Data, weights, n_iterations,  o_learning_rate,o_stdev, alpha, beta, grid_size, grid_dim, check, error_list, stop, Gamma_bias, Beta_bias);
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


if i == 0
Freq = zeros(s_2(2),1);
b = ones(s_1(2),1);
Freq = zeros(s_1(2),1);
b = Gamma_bias .* (1/(grid_size^2) * ones(s_1(2),1)- Freq) ;
end


    %Find the winner of current_data
    for j = 1: s_1(2)
        current_weight = weights(:,j);
        current_b = b(j);
        
        
        if sum(sum(sum(((current_weight - current_data).^2) .^0.5))) <= winner_v
            winner_v = sum(sum(sum(((current_weight - current_data).^2) .^0.5)));
            winner_i = j;
        end
    end
    
    %Update the Frequency
    Freq(winner_i) = Freq(winner_i) + 1;
    
    for m = 1:length(Freq)
        if m == winner_i
            Freq(m) = Freq(m) + Beta_bias* (1 - Freq(m));
        else
            Freq(m) = Freq(m) + Beta_bias* (0 - Freq(m));
        end
        
    end
    
    %update Beta
    b = Gamma_bias .* (1/(grid_size^2) * ones(s_1(2),1)- Freq) ;


    

    
    %Update only the winner
    weights(:,winner_i) = weights(:,winner_i) + learning_rate .* (current_data - weights(:,winner_i));
    
   %{
    %Update Neighbors of the point
    distance = mandist([G(:,winner_i)]', G); %Computes the manhatten distance from the winner
    for j = 1: length(distance)
        if distance(j) == 0
            yes(j) = 1;
        elseif distance(j) == 1
            yes(j) = 1;
        else
            yes(j) = 0;
        end
    end
    distance = exp((-distance .^2)./(stdev^2));
    distance = distance .* yes;
    %distance = abs(distance .* stdev);
    weights = weights + learning_rate .* [ones(s_1(1),1) * distance] .*(current_data * ones(1,s_1(2)) - weights);
   %}
   
   
   %increment time
    i = i + 1
    
    %Exponential decay for the Neighborhood function and learning rate   
    
    if mod(i, beta) == 0 
        learning_rate = learning_rate /1.2;
    end
        
   
    if mod(i, check) == 0 
        figure
        hold on
        plot(Data(1,:), Data(2,:), 'c*')
        plot(weights(1,:), weights(2,:), 'r+', 'MarkerSize',10)
        lines(G,weights, grid_size, grid_dim) 
        title(i) 
        
        %Computes the Error for the SOM
        error_list = Error(weights, Data, error_list);
        
        if error_list(end) <= stop
            break
        end
    end
        
       %} 
   
    
end
end
