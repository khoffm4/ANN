nfunction Iris_Main
%Main function for NML. Simply run to use.
clear all
close all
rng(20)

%Setting Parameters
Params.time = 0;
Params.n_percepts = 10;
Params.slope = 1 ;
Params.learning_rate = 0.01;
Params.tolerance = 0.005;
Params.epoch = 200;
Params.n_checks = 20;
Params.n_iterations = 600000;
Params.weight_min = -0.1;
Params.weight_max = 0.1;
Params.Momentum = 0;


%Variables
delta_box = struct;
error_list = [];
delta_box.d_bias_1 = 0;
delta_box.d_bias_2 = 0;
delta_box.d_not_bias_1 = 0;
delta_box.d_not_bias_2 = 0;
delta_box.d_bias_1_prev = 0;
delta_box.d_bias_2_prev = 0;
delta_box.d_not_bias_1_prev = 0;
delta_box.d_not_bias_2_prev = 0;
y_list_training = [];
RMSE_error_list_training = [];
RMSE_error_list_testing = [];
RMSE_error_training = inf;
RMSE_error_training = inf;




%Generate Testing and Training Data
Training = [rand(200,1) * 0.9 + 0.1];
Training = [Training, 1./Training ./6 - 0.92]
[n_rules_training, n_inputs_training] = size(Training);
n_inputs_training = n_inputs_training -1;
Inputs = Training(1:n_rules_training, 1:n_inputs_training);
Outputs = Training(1:n_rules_training,n_inputs_training + 1);

Testing = fractional_generator(100);
[n_rules_testing, n_inputs_testing] = size(Testing);
n_inputs_testing = n_inputs_testing -1;
Inputs_testing = Testing(1:100, 1:n_inputs_training);
Outputs_testing = Testing(1:100,n_inputs_training + 1);

%}



%Randomize Weights  
weights_1 = (Params.weight_max -Params.weight_min).*rand(Params.n_percepts,n_inputs_training+1) + Params.weight_min ;
weights_2 = (Params.weight_max -Params.weight_min).*rand(1,Params.n_percepts+1) + Params.weight_min;


%Main Simulation
while Params.time < Params.n_iterations 
    %Choose an epoch
    batch = randperm(n_rules_training);
    
        for j = 1:Params.epoch 
        %Choose pattern
        current = batch(j);
        X = Inputs(current,:);
        D = Outputs(current,:);    

        %Pass through the simulation
        [weights_1, weights_2, delta_box, biases, y_2] = BPlearn(Params, X, D, weights_1, weights_2, delta_box);

        %increment time
        Params.time = Params.time +1;
        end
        

    
    %At the end of the epoch, update the weights.    
    weights_1 = [biases.bias_1 + delta_box.d_bias_1 + Params.Momentum * delta_box.d_bias_1_prev, biases.not_bias_1 + delta_box.d_not_bias_1 + Params.Momentum * delta_box.d_not_bias_1_prev];
    %weights_2 = [biases.bias_2 + delta_box.d_bias_2 + Params.Momentum * delta_box.d_bias_2_prev, biases.not_bias_2 + delta_box.d_not_bias_2'];
    weights_2 = [biases.bias_2 + delta_box.d_bias_2 + Params.Momentum * delta_box.d_bias_2_prev, biases.not_bias_2 + delta_box.d_not_bias_2' + Params.Momentum * delta_box.d_not_bias_2_prev'];
    
    %Set the previous deltas as past ones
    delta_box.d_bias_1_prev = delta_box.d_bias_1;
    delta_box.d_bias_2_prev = delta_box.d_bias_2;
    delta_box.d_not_bias_1_prev = delta_box.d_not_bias_1;
    delta_box.d_not_bias_2_prev = delta_box.d_not_bias_2;
    
    
    %Compute the Error (absolute value of the output minus the desired)
    current_error = abs(y_2 -D);
    error_list = [error_list, current_error];
    
    
    %Performance Check
        if mod(Params.time,Params.n_iterations/Params.n_checks) == 0
            RMSE_error_training = 0;
            RMSE_error_testing = 0;
            [RMSE_error_list_testing,RMSE_error_list_training, x_list_testing, x_list_training, y_list_testing, y_list_training] = BPrecall(Params, X, D, weights_1, weights_2, Training, Testing, RMSE_error_list_testing, RMSE_error_list_training, Inputs, Outputs, Inputs_testing, Outputs_testing);
            
        %If RMSE less than the tolerance, then stop simulation 
        end
end


%Final Performance Test
figure
hold on 

%1/x
x = 0.1 : 0.001: 1;
y = 1./x;
plot(x,y)

%Training Data
y_list_training = y_list_training +0.92 * 6;
points = [x_list_training', y_list_training'];
points = sortrows(points);
x_list_training = points(:,1);
y_list_training = points(:,2);
plot(x_list_training, y_list_training, 'b--o')


%Testing Data
points = [];
y_list_testing = y_list_testing +0.92 * 6;
points = [x_list_testing', y_list_testing'];
points = sortrows(points);
x_list_testing = points(:,1);
y_list_testing = points(:,2);
plot(x_list_testing, y_list_testing, 'c*')
legend('1/x','Outputs of the Training Data', 'Outputs of the Testing Data')
title('Plot of Desired Outputs vs Training Outputs and Testing Outputs')
xlabel('Input Value')
ylabel('Output Value')


end

%%
%This function takes in all that is necessary to do one time step of the
%simulation and outputs the changed weights, the outut (y_2) and all other
%relevent parameters.
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


%%
%This function takes in all that is necessary, and tests the error of the
%function
function [RMSE_error_list_testing,RMSE_error_list_training, x_list_testing, x_list_training, y_list_testing, y_list_training] = BPrecall(Params, X, D, weights_1, weights_2, Training, Testing, RMSE_error_list_testing, RMSE_error_list_training, Inputs, Outputs, Inputs_testing, Outputs_testing)
RMSE_error_training = 0;
RMSE_error_testing = 0;
x_list_testing = [];
y_list_testing = [];
x_list_training = [];
y_list_training = [];

%Training Errors
       for i = 1:length(Training)
            %Choose pattern
            X = Inputs(i,:);
            x_list_training = [x_list_training, X];
            D = Outputs(i,:);    

            %Layer 1
            bias_1 = weights_1(:,1);
            not_bias_1 = weights_1(:,2:end);
            Net_1 = bias_1 + not_bias_1 * X';
            y_1 = tanh(Net_1);

            %Layer 2
            bias_2 = weights_2(:,1);
            not_bias_2 = weights_2(:,2:end);
            Net_2 = bias_2 + not_bias_2 * y_1;
            y_2 = tanh(Net_2);
            y_list_training = [y_list_training, y_2];
            
            %Error Computation
            RMSE_error_training = (D - y_2)^2 + RMSE_error_training;
       end
            
            RMSE_error_training = RMSE_error_training / length(Training);
            RMSE_error_training = sqrt(RMSE_error_training);
            RMSE_error_list_training = [ RMSE_error_list_training, RMSE_error_training];

            
            %Testing Errors
       for i = 1:length(Testing)
                
            %Choose pattern
            X = Inputs_testing(i,:);
            x_list_testing = [x_list_testing, X];
            D = Outputs_testing(i,:);    

            %Layer 1
            bias_1 = weights_1(:,1);
            not_bias_1 = weights_1(:,2:end);
            Net_1 = bias_1 + not_bias_1 * X';
            y_1 = tanh(Net_1);

            %Layer 2
            bias_2 = weights_2(:,1);
            not_bias_2 = weights_2(:,2:end);
            Net_2 = bias_2 + not_bias_2 * y_1;
            y_2 = tanh(Net_2);
            y_list_testing= [y_list_testing, y_2];
            
            %Error Computation
            RMSE_error_testing = (D - y_2)^2 + RMSE_error_testing;
       end
            
            RMSE_error_testing = RMSE_error_testing / length(Testing);
            RMSE_error_testing = sqrt(RMSE_error_testing);
            RMSE_error_list_testing = [ RMSE_error_list_testing, RMSE_error_testing]; 
            

end




