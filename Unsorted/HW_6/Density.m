function Density(G, weights, grid_size, Data, grid_dim, Outputs)
winner = zeros(1,length(weights));
winner_v = Inf;
winner_i = Inf;
total = zeros(grid_size, grid_size);
dist = [];

%For each datapoint
for i = 1:length(Data)
    current_data = Data(:, i);
    
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

    
Map = ones(1,grid_size^grid_dim);
%count the number atrributed to each prototype
for j = 1: grid_size^grid_dim
   count = sum( winner ==j ) ;
   Map(j) = count;
end

if grid_dim == 2 
    vec = Map;
    Map = vec2mat(Map,grid_size);
    figure
    imagesc([0.5,7.5], [0.5,7.5],Map);
    colorbar;
    set(gca,'YDir','normal')
    colormap(jet(5))
elseif grid_dim == 3
    vec = Map;
    Map = vec2mat(Map,grid_size);
    for i = 0: grid_size-1
        current_block = Map(1 + (grid_size * i):(grid_size * (i+1)) , :) ; 
        total = [total + current_block];     
    end
    Map = vec2mat(total,grid_size);
    figure
    colormap(jet(5))
    set(gca,'YDir','normal')
    imagesc([0.5,9.5], [0.5,9.5],Map);
    colorbar;
elseif grid_dim == 4
    vec = Map;
    Map = vec2mat(Map,grid_size);
    
end

%Making the U-Matrix
for i  = 1: length(G)-1
    current_weight = weights(:,i);
    if G(2,i) == grid_size
        top_weight = weights(:, i +grid_size);
        x = [NaN; norm(current_weight - top_weight)];
        dist = [dist, x];     
        continue
    elseif G(1,i) == grid_size
        right_weight = weights(:,i+1);
        x = [norm(current_weight - right_weight); NaN];
        dist = [dist, x]; 
        continue
    end
    
    right_weight = weights(:,i+1);
    top_weight = weights(:, i +grid_size);
    x = [norm(current_weight - right_weight); norm(current_weight - top_weight)];
    dist = [dist, x];     
end
maxi = max(max(dist));

dist = dist./ maxi ;

hold on
for i = 1: length(dist)
    if G(2,i) == grid_size
        color = dist(2,i); 
        x = mod(i-1,grid_size );
        y = floor(i/grid_size) ;
        line([x, x+1], [y,y], 'color', [color, color, color], 'LineWidth',8);   
        continue
    elseif G(1,i) == grid_size
        color = dist(1,i); 
        x = mod(i,grid_size );
        y = floor(i/grid_size);
        line([x x],[y y+1],[1 1] , 'color', [color, color, color], 'LineWidth',8);
        continue
    end
    
    color = dist(1,i);
    x = mod(i,grid_size );
    y = floor(i/grid_size);
    line([x x],[y y+1],[1 1], 'color', [color, color, color], 'LineWidth',8);
    
    color = dist(2,i); 
    x = mod(i-1,grid_size );
    y = floor(i/grid_size) + 1;
    line([x, x+1], [y,y], 'color', [color, color, color], 'LineWidth',8);
    
    i = i+1;
end
color = [];
%For the Iris Data, color the maximum

if grid_dim == 4
for i  = 1: length(G)
    nodes = find(winner==i);
    Setosa = 0;
    Versacolor = 0;
    Virginica = 0;
    %Find the types of Flower for each datapoint 
    for j  = 1: length(nodes)
        if Outputs(:,i) == [1;0;0];
            Setosa = Setosa + 1;
        elseif Outputs(:,i) == [0;1;0];
            Versacolor = Versacolor + 1;
        else
            Virginica = Virginica +1;
        end
            
    end
    
    %Color the Datapoints
    if max([Setosa, Versacolor, Virginica]) == Setosa
        color = [color, 1];
    elseif max([Setosa, Versacolor, Virginica]) == Versacolor
        color = [color, 2];
    else
        color = [color, 3];
    end  
end
Mat = vec2mat(color, 10);

for i = 1: length(Mat)
    for j = 1: length(Mat)
        if Mat(i,j) == 1
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'r')
        elseif Mat(i,j) ==2 
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'g')
        else
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'b')
        end
    end
end
end
%}
end
