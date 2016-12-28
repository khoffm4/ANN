function  Main
rng(39)
close all

%Setting Parameters
n_percepts = 10;
learning_rate = 0.02;
max_weight = 0.1;
min_weight = -0.1;
n_epochs = 10000; 
n_checks = 5;
stopping = 0.01;
K = 1;
Momentum = 0.3;

%Setting Varaibles up
delta_1p = 0;
delta_2p = 0;


%Generate Training Data for 1/x

X = [rand(100,1) * 0.9 + 0.1];
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

%Importing the data
[full_data, label, tr_tst_index, tr_data, tst_data] = eight_class_data;
tr_D = label(tr_tst_index == 0);
tr_D = tr_D(1:200,:)
tr_data = tr_data(1:200,:)
tst_D = label(tr_tst_index == 1);
n_inputs = 6;
n_outputs = 1;


%Scale the Training Input
len = size(tr_data,1);
wid = size(tr_data,2);
max_Input = max(max(tr_data));
O_Input = tr_data;
tr_data = tr_data ./ max_Input;
Input_R = max(max(tr_data)) - min(min(tr_data));
X = tr_data - (Input_R /2) ;

%Scaling the Training Outputs
max_D = max(max(tr_D));
O_D = tr_D;
tr_D = tr_D ./ max_D;
D_R = range(tr_D);
D = tr_D - (D_R /2) ;

%Scaling the Testing Inputs
O_Input_test = tst_data;
tst_data = tst_data ./ max_Input;
X_test = tst_data - (Input_R /2) ;

%Scaling the Testing outputs
O_D_tst = tst_D;
tst_D = tst_D ./ max_D;
D_R = range(tst_D);
D_test = tst_D - (D_R /2) ;













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
    [~,~,testerr(epoch)] = recip_recall(Weight_1,Weight_2)  ; 
    %}
    
    %If IRIS, then uncomment:
    
    %[trainerr(epoch), thresh_train] = hits(Dk', yk');
    [yrecall,thresh_test,testerr(epoch)] = Iris_recall(Weight_1,Weight_2, X_test, D_test)  ; 
    %}
    
    %Stopping Criteria
    %if trainerr(epoch) <= stopping
    %    break
    %end
    
end
'Learning steps', epoch*K
trainerr(epoch)
testerr(epoch)

%1/x Plots

hold on
plot(K*(1:n_checks:length(testerr)),trainerr(1:n_checks:length(testerr)),'b.--')
plot(K*(1:n_checks:length(testerr)),testerr(1:n_checks:length(testerr)), 'r+-')
%axis([0 K*length(testerr)  0 1])
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
%{
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
fNET1 = tanh(W1*[ones(1,length(X_test));X_test']);
fNET2 = tanh(W2*[ones(1,length(X_test));fNET1]);
yrecall = fNET2';

%defining the Number of Correct hits
%[err, thresh] = hits(D_test, yrecall);
thresh= 0
err = 0
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

function [full_data, label, tr_tst_index, tr_data, tst_data] = eight_class_data
% Import the data
nrow = 128; % 128 rows in each image band
ncol = 128; % 128 columns in each image band
dim = 6; % 6 image bands
datalen = nrow*ncol; % number of entries in the file

%read data file
fid  = fopen('8class-caseI-n_m0v5.viff','r','l'); 
% NOTE: 
%       You may have to change the path to this file. For example if this
%       were placed in a data directory this line would be:
%       fid  = fopen('/path_to_data_directory/8class-caseI-n_m0v5.viff','r','l'); 
% NOTE:
%       'l' is needed if you work on a big endian machine otherwise
%       it can be omitted
[a,count] = fread(fid,1024); % file has a standard 1024-byte header
x = zeros(datalen,dim); 
%       Reading into a datalen x 8 array for easy handling
%       (making each image band into one continuous string of
%       pixel values)
%       This format is just like the 'iris' 
for i = 1:dim
   [x(:,i),count] = fread(fid,datalen,'float32'); % 32 bit float
end

% extract by band for use
for     i = 1:nrow
    for k = 1:ncol
        band1(i,k) = x((i-1) * ncol + k, 1);
        band2(i,k) = x((i-1) * ncol + k, 2);
        band3(i,k) = x((i-1) * ncol + k, 3);
        band4(i,k) = x((i-1) * ncol + k, 4);
        band5(i,k) = x((i-1) * ncol + k, 5);
        band6(i,k) = x((i-1) * ncol + k, 6);
    end
end

% combine the layers
full_data        = zeros(nrow,ncol,6);
full_data(:,:,1) = band1;
full_data(:,:,2) = band2;
full_data(:,:,3) = band3;
full_data(:,:,4) = band4;
full_data(:,:,5) = band5;
full_data(:,:,6) = band6;
% example of how to plot the image one band at a time
%imagesc(full_data(:,:,i)); caxis([0,255]); colormap(gray); colorbar;

% generate the labels
label                  = zeros(nrow,ncol);
label(  1:64 ,  1:64 ) = 1;    %red
label( 33:64 , 97:128) = 2;    %orange
label(  1:64 , 65:96 ) = 3;    %green
label(  1:32 , 97:128) = 4;    %yellow
label( 65:128, 65:128) = 5;    %white
label( 65:96 ,  1:32 ) = 6;    %blue
label( 97:128,  1:32 ) = 7;    %purple
label( 65:128, 33:64 ) = 8;    %gray

% generate the color labels
                  %r    or  g   y   w   b    p gray
label_color_map_r = [255;254;  0;252;255; 47;164;180];
label_color_map_g = [  0;146;255;228;255;146;  0;180];
label_color_map_b = [  0; 43;  0; 98;255;255;175;180];
label_color_map   = [label_color_map_r label_color_map_g label_color_map_b]/255;
% example of how to plot the labels for the whole image:
%imagesc(label); colormap(label_color_map);
 
% Training and test data creation

% seperate into train and test datas using a simple mask
% 0 = train; 1 = test; -1 = do not include;
tr_tst_index            = -1 * ones(ncol,nrow);
tr_tst_index(  1:64 ,:) = 0;
tr_tst_index( 63:64 ,:) = 1;
tr_tst_index(  1:2  ,:) = 1;
tr_tst_index( 65:128,:) = 0;
tr_tst_index( 65:66 ,:) = 1;
tr_tst_index(127:128,:) = 1;
% example of how to plot the labels of the region included:
% to visualize the tr_tst_index you can use.
% imagesc(tr_tst_index); colorbar;

% convert from the data from matrix to vector form 
% This produces train and test subsets as well as all the data
% use 1-in-C encoding
data_pt    = 1;
tst_data_pt= 1;
tr_data_pt = 1;
for     ii = 1:nrow
    for jj = 1:ncol
        c_data(data_pt,1:6) = ...
            [full_data(ii,jj,1) full_data(ii,jj,2) full_data(ii,jj,3) ...
             full_data(ii,jj,4) full_data(ii,jj,5) full_data(ii,jj,6)];
        data_pt = data_pt + 1;
        % if in train set
        if     tr_tst_index(ii,jj) == 0
            tr_data(tr_data_pt,1:6) = ...
                [full_data(ii,jj,1) full_data(ii,jj,2) full_data(ii,jj,3) ...
                 full_data(ii,jj,4) full_data(ii,jj,5) full_data(ii,jj,6)];
            tr_des(tr_data_pt,label(ii,jj)) = 1;
            tr_data_pt = tr_data_pt + 1;
        % if in test set
        elseif tr_tst_index(ii,jj) == 1
            tst_data(tst_data_pt,1:6) = ...
                [full_data(ii,jj,1) full_data(ii,jj,2) full_data(ii,jj,3) ...
                 full_data(ii,jj,4) full_data(ii,jj,5) full_data(ii,jj,6)];
            tst_des(tst_data_pt,label(ii,jj)) = 1;
            tst_data_pt = tst_data_pt + 1;
        end
    end
end
data_pt    = data_pt    - 1;
tst_data_pt= tst_data_pt- 1;
tr_data_pt = tr_data_pt - 1;
end
