function Main

clear all;
close all;
rng(10)
nrows = 9; ncols = 9; ndim = 2; % Set the dimensions of the input image
N = nrows*ncols;  % We will unfold the input image into a vector of length N
nP = 64;          % Number of LVQ prototypes (weight vectors)
nC = 3;           % Number of classes
maxsteps = 150000;% Max. number of steps allowed. Increase this as needed.
mfr = 10000;      % Monitoring frequency
LRsched = [0.02 0.01 0.005 0.001
           30000 50000 120000 inf ] % Decay schedule for learning rate mu
[lr1,lr2] = size(LRsched); % Get the size of the decay step function
n_points = 30;
check = maxsteps;

disp 'N, np, nC, mu, maxsteps'
N
nP
nC
maxsteps

% Specify the matrix of class labels for the spatial locations
	A = [ 3 1 1 1 1 2 3 3 3
   	3 3 3 3 3 2 3 3 3
   	3 3 3 3 3 1 2 2 2
   	3 1 1 1 1 1 1 1 1
   	3 1 1 1 2 1 1 1 1
   	3 1 1 2 2 2 1 1 1 
   	3 1 2 2 1 3 2 2 2
   	3 3 3 3 3 3 2 2 2
   	3 3 3 3 3 3 3 3 2];

disp 'Target classification, A'
figure, imagesc(A), colormap('gray'); % Check the image of target class labels
title('original')

% Assign class labels to prototypes
%    Cw = randsample(1:nC,nP,true); % To assign labels randomly
% Here I assign labels by hand, divide them evenly among prototypes
nLperClass = ceil(nP/nC); % This may not devide evenly into nP, take care
% The label assignment below is hard wired for the HW06 problem (i.e., for 3
% classes. Generalize / extend for more classes.
Cw = zeros(1,nP);
Cw(1:nLperClass)=1; 
Cw(nLperClass+1:2*nLperClass)=2;
Cw(2*nLperClass+1:nP)=3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% No change should be needed below this line for learning
%%%%%%% the classification of a different image, with different
%%%%%%% number of prototyopes.

% Set the input vectors and associated categories
% We will assign the spatial coordinates of the labels in A
% to the ndim elements of each of the N, in a row-wise fashion
% and center the coordinates
%  
	X = zeros(ndim,N); 
	Cx = zeros(1,N);
	k = 0;
	for i = 1:nrows
   	for j = 1:ncols
      	k = k+1;
%       X(1,k) = i-ceil(nrows/2); % to center the y coordinates if desired
%      	X(2,k) = j-ceil(ncols/2); % to center the x coordinates if desired
        X(1,k) = i;
      	X(2,k) = j;
%      	Cx(k) = A(ncols-j+1, i); % this seems to flip
                                 % then requires a flipud at the end
        Cx(k) = A(j,i);
   	end
	end



% Implement the LVQ1 learning here to see how all of this hangs together
% Initialize prototypes 
%	W = 8*rand(2,nP)-4; % For centered coordinates. Notice the scaling!
%  
data_min=min(min(X));
data_max=max(max(X));
% Initialize the weights in the range of the input data
scale=1; % Make this a parameter if you like
W=scale*(rand(ndim,nP)*(data_max - data_min)+data_min);
%

 mu = LRsched(1,1); % initial value of the learning rate
for lstep = 1:maxsteps
    i = randsample(1:N,1); % Select sample index randomly
%		
   		d = zeros(1,nP);
   		for j = 1:nP
      		d(j) = norm(W(:,j)-X(:,i));
   		end
   		[mindist,I] = min(d);
   		if Cw(I) == Cx(i)
      		W(:,I) = W(:,I) + mu * (X(:,i)-W(:,I));
   		else
      		W(:,I) = W(:,I) - mu * (X(:,i)-W(:,I));
        end
%  
        if mod(lstep,mfr) == 0 
            lstep, mu
        end
        for lr = 2:lr2
        if lstep > LRsched(2,lr-1)
            mu = LRsched(1,lr);
        end
        end  
       


if mod(lstep, check) == 0 
% Test the classification of the training data using the prototypes 
% (weights, W) learned
	Cxhat = zeros(1,N);
	for i = 1:N
   	d = zeros(1,nP);
   	for j = 1:nP
      	d(j) = norm(W(:,j)-X(:,i));
   	end
   	[mindist,I] = min(d);
   	Cxhat(i) = Cw(I);
    end
    
% Reshape to original spatial image format, and display
    Cxhat_reshaped = reshape(Cxhat,nrows,ncols);
% predicted_classes = flipud(Cxhat_reshaped);
    predicted_classes = Cxhat_reshaped;
    disp 'predicted classes';
%    imagesc(Cxhat_reshaped); colormap('gray');
   figure, imagesc(predicted_classes), colormap('gray');
   title('recalled image')
end
end
   

   figure
  imagesc(abs(predicted_classes - A))
  colormap('gray');
  title('Difference image')
  colorbar()
    
  
   
   %% Testing the LVQ1

   	B = linspace(1,9,n_points);
    X_test = [];
    for i = 1: n_points
        X_test = [X_test, [ones(1,n_points) * B(i);B]];
    end
    
    

winner_list = [];
for i = 1:length(X_test)
    current_data = X_test(:,i);
    
    %Find the closest weight vector   
    D = [current_data(1) .* ones(length(W),1)'; current_data(2) .* ones(length(W),1)'];
    dist = sqrt(sum((D - W).^2));
    [~,index]=min(dist);
    winner = W(:,index);
    winner_list = [winner_list, Cw(index)];
end
  figure
hold on
  
  imagesc([1.5,8.5], [1.5,8.5], A)
    colormap('jet');
  title('Test Input Points and Prototypes superimposed on the Original Classes')
  colorbar()
  
  
for i = 1: length(winner_list)
    if winner_list(i) == 1
        scatter(X_test(1,i),X_test(2,i), 'b')
    elseif winner_list(i) == 2
        scatter(X_test(1,i),X_test(2,i), 'g')
    else
        scatter(X_test(1,i),X_test(2,i), 'r')
    end
end

plot(W(1,:), W(2,:), 'k*')
axis([1,9, 1,9])

%Categorization

C = [];
for i = 1: 9
    for j = 1: 9
        x = X_test(1,:);
        y = X_test(2,:);
        
        lx_1 = x<=i+1;
        lx_2 = i<=x;
        lx = and(lx_1, lx_2);
        
        ly_1 = y<=j+1;
        ly_2 = j<=y;
        ly = and(ly_1, ly_2);
        
        log = and(lx,ly);
        
        x = x(log);
        y = y(log);
        
        current_color = mode(winner_list(log));
        log_2 = repmat(log,2,1);
        
        C(j, i) = current_color;
     
    end
end
figure 

imagesc(flipud(C));
title('testing Data Classification')

%}


%% Generalized Hebbian Algorithmn

close all
rng(10);
%import the Iris Data
fileID = fopen('iris-train.txt');
C = textscan(fileID,'%n %n %n %n',...
'Delimiter','&', 'headerLines', 8);
fclose(fileID);
iris_train = cell2mat(C);

X = iris_train(1:2:end,:);
D = iris_train(2:2:end,:);
D = D(:, 2:4);


%Zero out the means of the cols
avg = mean(X,1);
X = X - repmat(mean(X,1),75,1) ;
var = corr(X);
[pc,score,eigvalue] = princomp(X);
pc
pc * pc';


%The GHA algorithmn
%Setting Parameters
n_percepts = 4;
learning_rate = 0.001;
max_weight = 0.1;
min_weight = -0.1;
n_epochs = 200000; 
check = 200000/20;
stopping = 0.0001;
K=75; M=4; N = 4;

%Weights
n_inputs = 4;
n_output = 4;
Weight_1 = (max_weight -min_weight).*rand(n_percepts,n_inputs) + min_weight;
%Weight_1 = randn(N,M)*0.05;
error_list = [];
epoch = 0

[w_1, error_list] = GHablearn(X,K, Weight_1, learning_rate, n_epochs, n_percepts, check, error_list, pc)
w_1 = w_1' 
w_1 = [w_1(:,1:2) .* -1, w_1(:,3:4)]
pc
plot(error_list)
title('Learning History')
xlabel('Learning Steps')
ylabel('Mean Absolute Error')
'hoi'




%%
%This function takes in all that is necessary to do one time step of the
%simulation and outputs the changed weights, the outut (y_2) and all other
%relevent parameters.
function [weight, error_list] = GHablearn(X,K,weight,learning_rate,n_epochs, n_percepts, check, error_list, pc)

for z = 1: n_epochs 
   k = randperm(length(X),K);
   X = X(k,:); 
   for i=1:n_percepts
      y=weight*X(i,:)'; 
      e_vector=[];
      for j=1:4
         e_vector=[e_vector; y(j)*(X(i,:)-y(1:j)'*weight(1:j,:))];
      end
      weight=weight+learning_rate*e_vector;
   end
   
   
   if mod(n_epochs,check) == 0
       error_list = [error_list, 1/16*sum(sum(abs(weight * weight' - eye(4,4))))];
       m = weight * weight';
   end
   z
end

end
end