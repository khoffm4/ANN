%% HEADER
% Author: Patrick O'Driscoll
% Created: 2014/02/03
% Updated: 2015/02/18 - provided better comments and augmented header
% Description:
%   Example of how to generate cross validation, and extract a single fold.

%% generate a sample dataset
%TrainIn  = rand(1,100);
TrainIn = rand(1, 100) .* 0;
TrainOut = 2*TrainIn;

TestIn   = rand(1,100);
TestOut  = 2*TestIn;

%% permute the data
% set the number of folds (this is a parameter)
FoldCount = 10;

%% Generate the folds
% collect the total size of the dataset
InSize = length(TrainIn)+length(TestIn);

% combine the test and training data and permute them
Perm    = randsample(1:InSize,InSize);
InPerm  = [TrainIn ,TestIn ];
OutPerm = [TrainOut,TestOut];
InPerm = InPerm(Perm);
OutPerm = OutPerm(Perm);

% print warning if this will not fold evenly
if mod(InSize,FoldCount) ~= 0
    error('This will not fold evenly');
end

% save memory for the following loop to save computation time
FoldIndex = zeros(FoldCount,InSize);

% set the index of the test and training data
for ii = 0:FoldCount-1
    temp = zeros(1,InSize);
    temp(1,ii*(1/FoldCount*InSize)+1:(ii+1)*(1/FoldCount*InSize)) = 1;
    FoldIndex(ii+1,:) = temp;
end
% 0 - training data
% 1 - test data

%% extract the data from the specified fold
CurrentFold = 1;

TrainFoldIn = InPerm(1,find(FoldIndex(CurrentFold,:)==0));
TestFoldIn  = InPerm(1,find(FoldIndex(CurrentFold,:)==1));

TrainFoldOut = OutPerm(1,find(FoldIndex(CurrentFold,:)==0));
TestFoldOut  = OutPerm(1,find(FoldIndex(CurrentFold,:)==1));

%% Verify that the permutation did not fail
figure
plot(TestIn,TestOut,'.');
figure
plot(TrainIn,TrainOut,'.');
figure
plot(TrainFoldIn,TrainFoldOut,'.');
figure
plot(TestFoldIn,TestFoldOut,'.');