
%This function makes the Density plots, the U Matrix, and the color coding
%for the iris data
function [winner, pic]=Density(G, weights, grid_size, Data, grid_dim, Outputs)
winner = zeros(1,length(weights));
winner_v = Inf;

%For each datapoint
for i = 1:size(Data,1)
    current_data = Data(i,:)';
    
    %compare again each prototype
    for j = 1: length(weights)
        current_weight = weights(:,j);
        if sum(sum(((current_weight - current_data).^2) .^0.5)) <= winner_v
            winner_v = sum((current_weight - current_data).^2 .^0.5);
            winner(i) = j;
        end
    end
    winner_v = Inf;
end

    
Map = zeros(1,grid_size^2);
%count the number atrributed to each prototype
for j = 1: length(G)
   count = sum( winner ==j ) ;
   Map(j) = count;
end

%Density Plot
    vec = Map;
    Map = vec2mat(Map,grid_size);
    figure
    %pic = imagesc([0.5,7.5], [0.5,7.5],Map);
    pic = imagesc([0.5,grid_size  - 0.5], [0.5,grid_size  - 0.5],Map);
    set(gca,'YDir','normal')
    colormap('hot')
    colorbar
    
    
end