function  Main2
rng(39)
close all

%Setting Parameters
n_percepts = 80;
learning_rate = 0.1;
max_weight = 0.1;
min_weight = -0.1;
n_epochs = 30000; 
n_checks = 10;
stopping = 0.05;
K = 40;
Momentum = 0;
n_folds = 3;
current_fold = 1;

%Setting Varaibles up
delta_1p = 0;
delta_2p = 0;

%Generating Training Data for Iris
%{
fileID = fopen('iris-train.txt');
C = textscan(fileID,'%n %n %n %n',...
'Delimiter','&');
fclose(fileID);
iris_train = cell2mat(C);

X_train_origin = iris_train(1:2:end,:);
D_train_origin = iris_train(2:2:end,:);
D_train_origin = D_train_origin(:, 2:4);

fileID = fopen('iris-test.txt');
C = textscan(fileID,'%n %n %n %n',...
'Delimiter','&');
fclose(fileID);
iris_test = cell2mat(C);
X_test_origin = iris_test(1:2:end,:);
D_test_origin = iris_test(2:2:end,:);
D_test_origin = D_test_origin(:, 2:4);

n_inputs = 4;
n_outputs = 3;

%[X_train, X_test, D_train, D_test] = folding(current_fold, n_folds, X_train, D_train, X_test, D_test);
[X_test, X_train, D_test, D_train] = folding(current_fold, n_folds, X_train_origin, D_train_origin, X_test_origin, D_test_origin);
%}

%Generating Data for the memoryless communication channel.
A = 1;
B = 0.2;
n = 1 : 100;
T = 1;

%Generate the Clean Signal (Desired)
D_train = 2*sin(((pi .* n) ./10));
D_test1 = 0.8 .* sin(pi .* n /5) + 0.25 .* cos(2*pi.*n/25) ;

%Apply the Noise
X_train = A*D_train + B * D_train.^2;
Noisey = X_train;

%Scale down 
D_train = D_train ./2.1;
X_train_max = max(X_train);
X_train_median = median(X_train);
X_train = (X_train - median(X_train)) ./ X_train_max;
X_train = X_train';
D_train = D_train';

%Variables
n_inputs = 1;
n_outputs = 1;



%Randomize Weights
Weight_1 = (max_weight -min_weight).*rand(n_percepts,n_inputs + 1) + min_weight;
Weight_2 = (max_weight -min_weight).*rand(n_percepts + 1,n_outputs) + min_weight;
Weight_2 = Weight_2';


%random sequence of samples
n = randperm(K);

for epoch = 1:n_epochs
    k = randperm(K);
    xk = X_train(k,:)'; %1*200
    
    %Run thought the simulation
    [Weight_1, Weight_2, yk, Dk, delta_1p, delta_2p] = BPlearn(K, xk, Weight_1, Weight_2, X_train, D_train, learning_rate, k, Momentum,  delta_1p, delta_2p);
    
    %If IRIS, then uncomment:
    %[trainerr(epoch), thresh_train] = hits(Dk', yk');
    %{
    [trainerr(epoch), thresh_train] = hits(Dk', yk');
    [~,thresh_test,testerr(epoch)] = Iris_recall(Weight_1,Weight_2, X_test, D_test); 
    %}
    
    %If memoryless communication channel then uncomment:
    [trainerr(epoch)] = (1/length(Dk)).*(sum(Dk * 2.1 - yk * X_train_max + median(X_train)).^2);  
    
    %}
    %Stopping Criteria
    if trainerr(epoch) <= stopping
        break
    end
    
end
%Run one last time
%{
k = randperm(K);
xk = X_train_origin(k,:)';
[Weight_1, Weight_2, yk, Dk, delta_1p, delta_2p] = BPlearn(K, xk, Weight_1, Weight_2, X_train, D_train, learning_rate, k, Momentum,  delta_1p, delta_2p);
[trainerr(epoch), thresh_train] = hits(Dk', yk');
[~,thresh_test,testerr(epoch)] = Comm_Recall(Weight_1,Weight_2, X_test_origin, D_test_origin) ;
%}

'Learning steps', epoch*K
trainerr(epoch)
%Uncomment if IRIS Data
%testerr(epoch)


%Communication Plots
plot(trainerr)
legend('MSE Training Error')
figure
[Y_recall] = Comm_Recall(Weight_1,Weight_2, X_train, D_train);
Y_recall = Y_recall .* X_train_max +  mean(X_train);
plot(1:length(Y_recall), Y_recall, 'blue' , 1:length(Y_recall), D_train* 2.1, 'red', 1:length(Y_recall), Noisey, 'green')
legend('Recalled Signal', 'Original Signal', 'Noisey Signal')
'yes'




%Iris Plots
%{
hold on
plot(K*(1:n_checks:length(testerr)),trainerr(1:n_checks:length(testerr)),'b.--')
plot(K*(1:n_checks:length(testerr)),testerr(1:n_checks:length(testerr)),'r+-')
axis([0 K*length(testerr)  0 0.6])
legend('Classification Error of Training Data', 'Classification Error of Testing Data')
string = strcat('Classification Error of Training and Testing Data for Fold: ', num2str(current_fold)) ; 
title(string,'FontSize',14)
xlabel('Learning steps','FontSize',12)
ylabel('Classification Error', 'FontSize',12)

figure
hold on
for j = 1 : length(D_test_origin)
    if D_test_origin(j,:) == [1,0,0]
        plot(j,0,'s', 'markers',8)
    elseif D_test_origin(j,:) == [0,1,0]
        plot(j,1, 's', 'markers',8)
    else
        plot(j,2,'s', 'markers',8)
    end 
    
end
axis([0 length(D_test_origin)  -0.5 2.5])

for j = 1 : length(D_test_origin)
    if thresh_test(j,:) == [1,0,0]
        plot(j,0,'*', 'markers',8)
    elseif thresh_test(j,:) == [0,1,0]
        plot(j,1, '*', 'markers',8)
    else
        plot(j,2,'*', 'markers',8)
    end 

end
string =  strcat('Predicted vs Desired Outputs for the Testing Data for Fold Number', num2str(current_fold)) ; 
title(string, 'FontSize',14)
xlabel('Flower Number', 'FontSize',12)
ylabel('Classification Class', 'FontSize',12)

%}
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
    %Dk = D(k,:)'; %1*200
    %D = D';
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


%%Takes in inputs and outputs and outputs a fold for the training and
%testing data
function [TrainFoldIn, TestFoldIn, TrainFoldOut, TestFoldOut] = folding(current_fold, FoldCount, X_train, D_train, X_test, D_test);
rng(10)
%% generate a sample dataset
TrainIn  = X_train;
TrainOut = D_train;

TestIn   = X_test;
TestOut  = D_test;


%% Generate the folds
% collect the total size of the dataset
InSize = length(TrainIn)+length(TestIn);

% combine the test and training data and permute them
Perm    = randsample(1:InSize,InSize);
InPerm  = [TrainIn ;TestIn ];
OutPerm = [TrainOut;TestOut];
InPerm = InPerm(Perm,:);
OutPerm = OutPerm(Perm,:);

% print warning if this will not fold evenly
if mod(InSize,FoldCount) ~= 0
    error('This will not fold evenly');
end

% save memory for the following loop to save computation time
FoldIndex = [repmat([1], 1,50), repmat([2], 1,50), repmat([3], 1,50)];
Perm    = randsample(1:150,150);
FoldIndex = FoldIndex(Perm);
FoldIndex(FoldIndex~=current_fold) = 0;
FoldIndex(FoldIndex==current_fold) = 1;


FoldIndex = logical(FoldIndex');

% 0 - training data
% 1 - test data

%% extract the data from the specified fold
TrainFoldIn = InPerm(FoldIndex, :);
TestFoldIn  = InPerm(~FoldIndex, :);

TrainFoldOut = OutPerm(FoldIndex, :);
TestFoldOut  = OutPerm(~FoldIndex, :);
end


function [Y_recall] = Comm_Recall(W1,W2, X_train, D_train) ;
rng(49)
fNET1 = tanh(W1*[ones(1,length(D_train));X_train']);
fNET2 = tanh(W2*[ones(1,length(D_train));fNET1]);
Y_recall = fNET2';
end

%%
%This function takes in all that is necessary, and tests the error of the
%function with the dataset is the IRIS data
function [yrecall,thresh,err] = Iris_recall(W1,W2, X_test, D_test)
rng(49)
fNET1 = tanh(W1*[ones(1,length(D_test));X_test']);
fNET2 = tanh(W2*[ones(1,length(D_test));fNET1]);
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


