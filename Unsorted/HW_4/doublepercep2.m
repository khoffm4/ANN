function doublepercep2
n_iterations = 50000;
%Setting Parameters
time = 0;
n1_percepts = 2
n2_percepts = 2
slope = 1 
learning_rate = 0.005
error_list = []

%Generate XOR rules
Rules = [1,1,0;1,0,1;0,1,1;0,0,0];
n_rules = size(Rules);
n_rules = n_rules(1);
Inputs = Rules(1:4, 1:2);
Outputs = Rules(1:4,3);


%Randomize Weights
weights_1 = rand(2,3);
weights_2 = rand(1,3);

for i = 1: n_iterations
    %Choose pattern
    current = randi(n_rules);
    X = Inputs(current,:);
    D = Outputs(current,:);    
    
    %Layer 1
    bias_1 = weights_1(:,1);
    not_bias_1 = weights_1(:,2:3);
    Net_1 = bias_1 + not_bias_1 * X';
    y_1 = tanh(Net_1);
    
    %Layer 2
    bias_2 = weights_2(:,1);
    not_bias_2 = weights_2(:,2:3);
    Net_2 = bias_2 + not_bias_2 * y_1;
    y_2 = tanh(Net_2);
    
    %Feedback Errors
    delta_2 = (D - y_2) * (1 - y_2 ^2);
    delta_1 = delta_2 * not_bias_2' .* (1- y_1 .^2);
    
    %updating Errors
    d_bias_1 = learning_rate * delta_1 *1 ;
    d_bias_2 = learning_rate * delta_2 *1 ;
    d_not_bias_1 = learning_rate * delta_1 * X;
    d_not_bias_2 = learning_rate * delta_2 * y_1;
    
    weights_1 = [bias_1 + d_bias_1, not_bias_1 + d_not_bias_1];
    weights_2 = [bias_2 + d_bias_2, not_bias_2 + d_not_bias_2'];

    
    %Error
    current_error = abs(y_2 -D);
    error_list = [error_list, current_error];
    
   
    
    
    
    
end

x = 1:n_iterations;
plot(x, error_list)
end
