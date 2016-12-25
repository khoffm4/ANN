
function Color_Classes_Iris(G, weights, grid_size, Data, grid_dim, Outputs,BMU)
color = []; 
for i  = 1: length(G) 
    nodes = find(BMU==i);
    C_1 = 0;
    C_2 = 0;
    C_3 = 0;
    %Find the types of Flower for each datapoint 
    for j  = 1: length(nodes)
        if Outputs(nodes(j)) == 0;
            C_1 = C_1 + 1;
        elseif Outputs(nodes(j)) == 1;
            C_2 = C_2 + 1;
        elseif Outputs(nodes(j)) == 2;
            C_3 = C_3 + 1;
        end
            
    end
    
    %Color the Datapoints
    if max([C_1, C_2, C_3]) == 0
        color = [color,0];
    elseif max([C_1, C_2, C_3]) == C_1
        color = [color, 1];
    elseif max([C_1, C_2, C_3]) == C_2
        color = [color, 2];
    elseif max([C_1, C_2, C_3]) == C_3
        color = [color, 3];
    end
end

Mat = vec2mat(color, grid_size);

%Color the classes
hold on
for i = 1: length(Mat)
    for j = 1: length(Mat)
        if Mat(i,j) == 0
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'k')
        elseif Mat(i,j) == 1
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'r')
        elseif Mat(i,j) ==2 
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'b')
        elseif Mat(i,j) ==3
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'g')
        end
    end
end

end

