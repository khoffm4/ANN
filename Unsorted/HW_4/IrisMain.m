function IrisMain
rng(39)

%Setting Parameters
n_percepts = 10;
learning_rate = 0.01;
max_weight = 0.1;
min_weight = -0.1;
n_epochs = 3000; 
n_checks = 20;
stopping = 0.1;
K = 1;
Momentum = 0.5;

%Setting Varaibles up
delta_1p = 0;
delta_2p = 0;


%Generate Training Data 
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
X_test = iris_train(1:2:end,:);
D_test = iris_train(2:2:end,:);
D_test = D_test(:, 2:4);

%Randomize Weights
Weight_1 = (max_weight -min_weight).*rand(n_percepts,5) + min_weight;
Weight_2 = (max_weight -min_weight).*rand(n_percepts + 1,3) + min_weight;
Weight_2 = Weight_2';


%random sequence of samples
n = randperm(K);

for epoch = 1:n_epochs
    k = randperm(K);
    xk = X(k,:)'; %1*200
    
    [Weight_1, Weight_2, yk, Dk, delta_1p, delta_2p] = BPlearn(K, xk, Weight_1, Weight_2, X, D, learning_rate, k, Momentum,  delta_1p, delta_2p);
    
    trainerr(epoch) =  mean(abs(Dk-yk));
    [~,testerr(epoch)] = recall(Weight_1,Weight_2, X_test, D_test)  ; 
 
    %Stopping Criteria
    if trainerr(epoch) < stopping
        break
    end
    
end
'Learning steps', epoch*200
trainerr(epoch)
testerr(epoch)

end

%%
%This function takes in all that is necessary, and tests the error of the
%function
function [yrecall,err] = recall(W1,W2, X_test, D_test)
rng(49)
fNET1 = tanh(W1*[ones(1,75);X_test']);
fNET2 = tanh(W2*[ones(1,75);fNET1]);
yrecall = fNET2';
err = 2;
%defining the Number of Correct hits




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
