%This runs LVQ 1 on the Iris Dataset, and then computes the Meansquared
%Distance bewteen the protypes and the datapoints
function Ex2
addpath(genpath('../Data'))
addpath(genpath('../Code/SOM'))
close all
hold on
rng(10)

n_prototypes = 10;
n_iterations =100;
learning_rate = 0.001;

%Create the Data
fileID = fopen('iris.txt');
C = textscan(fileID,'%n %n %n %n',...
'Delimiter','&');
fclose(fileID);
iris = cell2mat(C);
Data = iris(1:2:end,:);

Classes = iris(2:2:end,:);
Classes = Classes(:, 2:4);
Outputs_tmp = Classes(:, 1:3);

%Convert outputs from 1 in k encoding to integers
Classes = [];
for i = 1: length(Outputs_tmp)
   if Outputs_tmp(i,:) == [1,0,0]
       Classes = [Classes; 0];
   elseif Outputs_tmp(i,:) == [0,1,0]
       Classes = [Classes; 1];
   else
       Classes = [Classes; 2];
   end
end




%Randomly assign classes for the prototypes
Classes_PE = randi([1,3],n_prototypes,1);

center0= mean(Data(find(Classes ==0),:));
center1= mean(Data(find(Classes ==1),:));
center2= mean(Data(find(Classes ==2),:));

%Initialize weights in a box of 0.1x 0.1x around the center of mass of each
%class
%weights = -0.05 + 0.1* rand(grid_dim,grid_size ^2) + center0 .* ones(grid_dim,grid_size ^2);
    for j = 1:length(Classes_PE)
        if Classes(j) == 0
            weights(j,:) = [center0 + rand*0.001]';
        elseif Classes(j) == 1
            weights(j,:) = [center1 + rand*0.001]';
        else
            weights(j,:) = [center2 + rand*0.001]';
        end
    end

%Call the LVQ1 function
for i = 1:n_iterations
    [weights] = LVQ1(Data,Classes,Classes_PE, weights, n_iterations, learning_rate);
    i/n_iterations
end

hold on
plot(Data(:,1), Data(:,2), 'k.')

for i = 1: length(Classes_PE)
   if Classes_PE(i) == 0
       plot(weights(:,1), weights(:,2), 'r.')
   elseif Classes_PE(i) == 0
       plot(weights(:,1), weights(:,2), 'g.')
   else
       plot(weights(:,1), weights(:,2), 'b.')
   end
end

end