function fractional4
%Main function for NML. Simply run to use.
%Setting Parameters
time = 0;
n_percepts = 10;
slope = 1 ;
learning_rate = 0.01;
tolerance = 0.005;
epoch = 200;
n_checks = 20;
n_iterations = 600000;


%Variables
error_list = [];
d_bias_1 = 0;
d_bias_2 = 0;
d_not_bias_1 = 0;
d_not_bias_2 = 0;
y_list = [];
RMSE_error_list_training = [];
RMSE_error_list_testing = [];
RMSE_error_training = 1000;
RMSE_error_training = 1000;

%Generate Testing and Training Data
Training =  fractional_generator(200);
maximum = max(Training(:,2));
Training = [Training(:,1) , Training(:,2)/maximum];
[n_rules_training, n_inputs_training] = size(Training);
n_inputs_training = n_inputs_training -1;
Inputs = Training(1:n_rules_training, 1:n_inputs_training);
Outputs = Training(1:n_rules_training,n_inputs_training + 1);

Testing = fractional_generator(100);
[n_rules_testing, n_inputs_testing] = size(Testing);
n_inputs_testing = n_inputs_testing -1;
maximum_testing = max(Testing(:,2));
Testing = [Testing(:,1) , Testing(:,2)/maximum_testing];
Inputs_testing = Testing(1:100, 1:n_inputs_training);
Outputs_testing = Testing(1:100,n_inputs_training + 1);


%Randomize Weights
a = -0.1;
b = 0.1;
r = (b-a).*rand(1000,1) + a;
weights_1 = (b-a).*rand(n_percepts,n_inputs_training+1) + a ;
weights_2 = (b-a).*rand(1,n_percepts+1) + a;


while time < n_iterations 
    %Choose an epoch
    batch = datasample(1:n_rules_training, epoch, 'Replace', false);
    for j = 1:epoch 
    
    
    %Choose pattern
    current = batch(j);
    X = Inputs(current,:);
    D = Outputs(current,:);    
    
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
    
    %Feedback Errors
    delta_2 = (D - y_2) * (1 - y_2 ^2);
    delta_1 = delta_2 * not_bias_2' .* (1- y_1 .^2);
    
    %updating Errors
    d_bias_1 = d_bias_1 + learning_rate * delta_1 *1 ;
    d_bias_2 = d_bias_2 + learning_rate * delta_2 *1 ;
    d_not_bias_1 = d_not_bias_1 + learning_rate * delta_1 * X;
    d_not_bias_2 = d_not_bias_2 + learning_rate * delta_2 * y_1;
    
    %increment time
    time = time +1;
    end
    
    weights_1 = [bias_1 + d_bias_1, not_bias_1 + d_not_bias_1];
    weights_2 = [bias_2 + d_bias_2, not_bias_2 + d_not_bias_2'];

    
    %Error
    current_error = abs(y_2 -D);
    error_list = [error_list, current_error];
    
    
    
    %Performance Check
        if mod(time,n_iterations/n_checks) == 0
            RMSE_error_training = 0;
            RMSE_error_testing = 0;
            for i = 1:n_rules_training 
            %Choose pattern
            X = Inputs(i,:);
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
            
            %Error Computation
            RMSE_error_training = (D - y_2)^2 + RMSE_error_training;
            end
            
            RMSE_error_training = RMSE_error_training / n_rules_training;
            RMSE_error_training = sqrt(RMSE_error_training);
            RMSE_error_list_training = [ RMSE_error_list_training, RMSE_error_training];
            
            
            %Testing Errors
            for i = 1:100
                
            %Choose pattern
            X = Inputs_testing(i,:);
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
            
            %Error Computation
            RMSE_error_testing = (D - y_2)^2 + RMSE_error_testing;
            end
            
            RMSE_error_testing = RMSE_error_testing / n_rules_testing;
            RMSE_error_testing = sqrt(RMSE_error_testing);
            RMSE_error_list_testing = [ RMSE_error_list_testing, RMSE_error_testing];
            end
            
           
        
        %If RMSE less than the tolerance, then stop simulation 
        if RMSE_error_training < tolerance
                break
        end
end
%RMSE Error Plots
hold on
%x = 30000:30000:600000;
%scatter(x, RMSE_error_list_training)
%scatter(x, RMSE_error_list_testing)
%lsline
%legend('RMSE of the Training Data', 'RMSE of the Testing Data')
%title('RMSE of Training Outputs and Testing Outputs')
%xlabel('Input Value')
%ylabel('RMSE')
%figure


%Final Performance Review
%Function approximation
figure
hold on 

%1/x
x = 0.1 : 0.001: 1;
y = 1./x;
plot(x,y)

%training Data
y_list = [];
x_list = [];
for i = 1:n_rules_training 
            %Choose pattern
            X = Inputs(i,:);
            x_list = [x_list, X];
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
            y_list= [y_list, y_2];
end
%plot(x_list,y_list)
y_list = y_list * maximum;
points = [x_list', y_list'];
points = sortrows(points);
x_list = points(:,1);
y_list = points(:,2);

plot(x_list, y_list, 'b--o')


%Testing Data

Inputs = Testing(1:100, 1:n_inputs_training);
Outputs = Testing(1:100,n_inputs_training + 1);

x_list_testing = [];
y_list_testing = [];
points = [];

for i = 1:100 
            %Choose pattern
            X = Inputs(i,:);
            x_list_testing = [x_list_testing, X];
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
            y_list_testing= [y_list_testing, y_2];
end
y_list_testing = y_list_testing * maximum;
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

function [z] = fractional_generator(n)
%computes a 2xn list where the first column is x values between 0.1 and 1
%and the y column is 1/x
a = 0.1;
b = 1;
x = (b-a).*rand(n,1) + a;
y = 1./x;
z = [x,y];

end
