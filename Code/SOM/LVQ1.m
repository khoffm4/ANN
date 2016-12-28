%This function runs the LVQ1 algorithmn.
function [weights] = LVQ1(Data,Classes,Classes_PE, weights, n_iterations, learning_rate)
rng(10)

%initialize values
s_1 = size(weights);
s_2 = size(Data);

i = 0;
while  i <= n_iterations
    
%choose a datapoint
seed = randi([1,s_2(1)],1,1);
current_data = Data(seed, :);
current_class = Classes(seed,:);
winner_v = Inf;
winner_i = Inf;


    %Find the winner of current_data
    for j = 1: s_1(2)
        current_weight = weights(j,:);
        if sum(sum(sum(((current_weight - current_data).^2) .^0.5))) <= winner_v
            winner_v = sum((current_weight - current_data).^2 .^0.5);
            winner_i = j;
        end
    end
    
    %update the winner
    if Classes_PE(winner_i) == current_class
        weights(winner_i,:) = weights(winner_i,:) + learning_rate * (current_data - weights(winner_i,:));
    else
        weights(winner_i,:) = weights(winner_i,:) - learning_rate * (current_data - weights(winner_i,:));
    end

   
    %increment time
    i = i + 1;

end
end