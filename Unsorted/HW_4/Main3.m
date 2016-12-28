function Main3() %main function for HW04 problem 2
%Runtime before optimization: 6.703630 seconds.
clear all
close all
clc
rng(20)
%Setting Parameters

learning_rate = 0.01;
n_epochs = 3000;

%initialize weights; first column is bias
Weight_1 = (rand(10,2)-0.5)/5;
Weight_2 = (rand(1,11)-0.5)/5;
%set learning parameters

%input/output matrix where rows are observations and columns are variables
%first column is bias
X = rand(200,1)*0.9+0.1;
D = 1./X./6-0.92;


%K samples in each epoch
K = 200;

%random sequence of samples
n = randperm(K);

for epoch = 1:n_epochs
    
    k = randperm(K);
    xk = X(k,:)'; %1*200
    NET1 = Weight_1*[ones(1,K);xk]; %10*200
    fNET1 = tansig(NET1); %10*200
    NET2 = Weight_2*[ones(1,K);fNET1]; %1*200
    fNET2 = tansig(NET2); %1*200
    yk = fNET2; %1*200
    Dk = D(k,1)'; %1*200
    
    delta_l2 = (Dk - yk) .* (1-(fNET2).^2); %1*200
    delta_l1 = (1-(fNET1).^2).* (Weight_2(2:end)'*delta_l2); %10*200
    
    
    Weight_1 = Weight_1 + learning_rate * (delta_l1 * [ones(1,K);xk]');
    Weight_2 = Weight_2 + learning_rate * (delta_l2 * [ones(1,K);fNET1]');   
    
    trainerr(epoch) =  mean(abs((Dk+0.92)*6-(yk+0.92)*6));
    [~,~,testerr(epoch)] = recall(Weight_1,Weight_2);    
    
end

plot(200*(1:20:3000),trainerr(1:20:3000),'b.')
hold on
plot(200*(1:20:3000),testerr(1:20:3000),'r+')
axis([0 600000 0 1])

figure
plot([0.1:0.001:1],1./[0.1:0.001:1])
hold on
plot(xk,(yk+0.92)*6,'r.')
[x,y,e] = recall(Weight_1,Weight_2);
plot(x,y,'g.')

function [xtest,yrecall,err] = recall(W1,W2)
rng(100)
xtest = rand(100,1)*0.9+0.1;
fNET1 = tanh(W1*[ones(1,100);xtest']);
fNET2 = tanh(W2*[ones(1,100);fNET1]);
yrecall = (fNET2+0.92)*6;
err = mean(abs(1./xtest' - yrecall));