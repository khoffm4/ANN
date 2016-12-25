function UMatrix(G, weights, grid_size, Data, grid_dim, Outputs)
dist = [];
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
        line([x, x+1], [y,y],[1,1], 'color', [color, color, color], 'LineWidth',8);   
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
    line([x, x+1], [y,y], [1,1], 'color', [color, color, color], 'LineWidth',8);
    
    i = i+1;
end

end





