function [weights_1, weights_2, delta_box, biases, y_2] = BPlearn(Params, X, D, weights_1, weights_2, delta_box)

    %Layer 1
    biases.bias_1 = weights_1(:,1);
    biases.not_bias_1 = weights_1(:,2:end);
    Net_1 = biases.bias_1 + biases.not_bias_1 * X';
    y_1 = tanh(Net_1);
    
    %Layer 2
    biases.bias_2 = weights_2(:,1);
    biases.not_bias_2 = weights_2(:,2:end);
    Net_2 = biases.bias_2 + biases.not_bias_2 * y_1;
    y_2 = tanh(Net_2);
    
    %Feedback Errors
    delta_2 = (D - y_2) * (1 - y_2 ^2);
    delta_1 = delta_2 * biases.not_bias_2' .* (1- y_1 .^2);
    
    %updating Errors
    delta_box.d_bias_1 = delta_box.d_bias_1 + Params.learning_rate * delta_1 *1 ;
    delta_box.d_bias_2 = delta_box.d_bias_2 + Params.learning_rate * delta_2 *1 ;
    delta_box.d_not_bias_1 = delta_box.d_not_bias_1 + Params.learning_rate * delta_1 * X;
    delta_box.d_not_bias_2 = delta_box.d_not_bias_2 + Params.learning_rate * delta_2 * y_1;
    
    %increment time
    Params.time = Params.time +1;

end
