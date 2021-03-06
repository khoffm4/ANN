function  MainCopy
rng(39)
close all

%Setting Parameters
n_percepts = 2;
learning_rate = 0.01;
max_weight = 0.1;
min_weight = -0.1;
n_epochs = 3000; 
n_checks = 200;
stopping = 0.01;
K = 100;
Momentum = 0.5;

%Setting Varaibles up
delta_1p = 0;
delta_2p = 0;


%Generate Training Data for 1/x
%{
X = [rand(200,1) * 0.9 + 0.1];
D = 1./X ./6 - 0.92;
n_inputs = 1;
n_outputs = 1;
%}

%Generating Training Data for Iris

fileID = fopen('iris-train.txt');
C = textscan(fileID,'%n %n %n %n',...
'Delimiter','&');
fclose(fileID);
iris_train = cell2mat(C);

X = iris_train(1:2:end,:);
D = iris_train(2:2:end,:);
D = D(:, 2:4);

fileID = fopen('iris-test.txt');
C = textscan(fileID,'%n %n %n %n',...
'Delimiter','&');
fclose(fileID);
iris_test = cell2mat(C);
X_test = iris_test(1:2:end,:);
D_test = iris_test(2:2:end,:);
D_test = D_test(:, 2:4);

n_inputs = 4;
n_outputs = 3;
%}

%Randomize Weights
Weight_1 = (max_weight -min_weight).*rand(n_percepts,n_inputs + 1) + min_weight;
Weight_2 = (max_weight -min_weight).*rand(n_percepts + 1,n_outputs) + min_weight;
Weight_2 = Weight_2';


%random sequence of samples
n = randperm(K);

for epoch = 1:n_epochs
    k = randperm(K);
    xk = X(k,:)'; %1*200
    
    [Weight_1, Weight_2, yk, Dk, delta_1p, delta_2p] = BPlearn(K, xk, Weight_1, Weight_2, X, D, learning_rate, k, Momentum,  delta_1p, delta_2p);
    
    %If 1/x then uncomment:
    %{
    trainerr(epoch) =  mean(abs((Dk+0.92)*6-(yk+0.92)*6));
    trainerr(epoch) =  mean(abs((Dk+0.92)*6-(yk+0.92)*6));
    [~,~,testerr(epoch)] = recip_recall(Weight_1,Weight_2)  ; 
    %}
    
    %If IRIS, then uncomment:
    
    [trainerr(epoch), thresh_train] = hits(Dk', yk');
    [yrecall,thresh_test,testerr(epoch)] = Iris_recall(Weight_1,Weight_2, X_test, D_test)  ; 
    %}
    
    %Stopping Criteria
    if trainerr(epoch) <= stopping
        break
    end
    
end
'Learning steps', epoch*K
trainerr(epoch)
testerr(epoch)

%1/x Plots
%{
hold on
plot(K*(1:n_checks:length(testerr)),trainerr(1:n_checks:length(testerr)),'b.--')
plot(K*(1:n_checks:length(testerr)),testerr(1:n_checks:length(testerr)), 'r+-')
axis([0 K*length(testerr)  0 1])
legend('Mean Absolute Error of Training Data', 'Mean Absolute Error of Testing Data')
title('Mean Absolute Error of Training and Testing Data')
xlabel('Learning steps')
ylabel('Mean Absolute Error')


figure
plot([0.1:0.001:1],1./[0.1:0.001:1])
hold on
plot(xk,(yk+0.92)*6,'r.')
[x,y,e] = recip_recall(Weight_1,Weight_2);
plot(x,y,'g.')
legend('1/x','Outputs of the Training Data', 'Outputs of the Testing Data')
title('Momentum vs Expected Output on Testing')
xlabel('Input Value')
ylabel('Output Value')
%}




%Iris Plots

hold on
plot(K*(1:n_checks:length(testerr)),trainerr(1:n_checks:length(testerr)),'b.--')
plot(K*(1:n_checks:length(testerr)),testerr(1:n_checks:length(testerr)),'r+-')
axis([0 K*length(testerr)  0 0.6])
legend('Classification Error of Training Data', 'Classification Error of Testing Data')
title('Classification Error of Training and Testing Data')
xlabel('Learning steps')
ylabel('Classification Error')

figure
hold on
for j = 1 : 75
    if D_test(j,:) == [1,0,0]
        plot(j,0,'s')
    elseif D_test(j,:) == [0,1,0]
        plot(j,1, 's')
    else
        plot(j,2,'s')
    end 
    
end
axis([0 75  -0.5 2.5])

for j = 1 : 75
    if thresh_test(j,:) == [1,0,0]
        plot(j,0,'*')
    elseif thresh_test(j,:) == [0,1,0]
        plot(j,1, '*')
    else
        plot(j,2,'*')
    end 

end
title('Predicted vs Desired Outputs of the Iris Data set on the Testing Data')
xlabel('Flower Number')
ylabel('Classification Class')

%}





end

%%
%This function takes in all that is necessary, and tests the error of the
%function with the dataset is the IRIS data
function [yrecall,thresh,err] = Iris_recall(W1,W2, X_test, D_test)
rng(49)
fNET1 = tanh(W1*[ones(1,75);X_test']);
fNET2 = tanh(W2*[ones(1,75);fNET1]);
yrecall = fNET2';

%defining the Number of Correct hits
[err, thresh] = hits(D_test, yrecall);
end

%%Computes the Number of Correct Hits
function [err, thresh_out] = hits(D_test, yrecall)
err = 0;
thresh_out = [];
for i = 1: length(D_test)
    current_output = yrecall(i,:);
    Desired_output = D_test(i,:);
    
    %Compute the distance to the three categorization points
    dist1 = pdist([current_output ; 1,0,0]);
    dist2 = pdist([current_output ; 0,1,0]);
    dist3 = pdist([current_output ; 0,0,1]);
    
    minimum = min([dist1, dist2, dist3]);
    
    %Assign it to the closest point
    if minimum == dist1
        current_output = [1,0,0];
        thresh_out = [thresh_out; current_output];
    elseif minimum == dist2
        current_output = [0,1,0];
        thresh_out = [thresh_out; current_output];
    else
        current_output = [0,0,1];
        thresh_out = [thresh_out; current_output];
    end
    
    %compute Error
    if current_output == Desired_output
    else
        err = err +1;
    end
end
err = err / 75;
end



%%
%This function takes in all that is necessary, and tests the error of the
%function for the 1/x function
function [xtest,yrecall,err] = recip_recall(W1,W2)
rng(49)
xtest = rand(100,1)*0.9+0.1;
fNET1 = tanh(W1*[ones(1,100);xtest']);
fNET2 = tanh(W2*[ones(1,100);fNET1]);
yrecall = (fNET2+0.92)*6;
err = mean(abs(1./xtest' - yrecall));
end

%%
%This function takes in all that is necessary to do one time step of the
%simulation and outputs the changed weights, the outut (y_2) and all other
%relevent parameters.
function [Weight_1, Weight_2, yk, Dk, delta_1p, delta_2p] = BPlearn(K, xk, Weight_1, Weight_2, X, D, learning_rate, k, Momentum, delta_1p, delta_2p);
    NET1 = Weight_1*[ones(1,K);xk]; %10*200
    fNET1 = tanh(NET1); %10*200
    NET2 = Weight_2*[ones(1,K);fNET1]; %1*200
    fNET2 = tanh(NET2); %1*200
    yk = fNET2; %1*200
    Dk = D(k,:)'; %1*200


    %Update Deltas
    delta_l2 = (Dk - yk) .* (1-(fNET2).^2) + Momentum * delta_1p; %1*200
    delta_l1 = (1-(fNET1).^2).* (Weight_2(:,2:end)'*delta_l2) + Momentum * delta_2p; %10*200
    
    %Update Previous Deltas
    delta_1p = delta_l2;
    delta_2p = delta_l1;
    
    %Update Weights
    Weight_1 = Weight_1 + learning_rate * (delta_l1 * [ones(1,K);xk]');
    Weight_2 = Weight_2 + learning_rate * (delta_l2 * [ones(1,K);fNET1]'); 
end
