
function Color_Classes_Zahn(G, weights, grid_size, Data, grid_dim, Outputs,winner)
color = []; 
for i  = 1: length(G) 
    nodes = find(winner==i);
    C_1 = 0;
    C_2 = 0;
    C_3 = 0;
    C_4 = 0;
    C_5 = 0;
    C_6 = 0;
    %Find the types of Flower for each datapoint 
    for j  = 1: length(nodes)
        if Outputs(nodes(j)) == 1;
            C_1 = C_1 + 1;
        elseif Outputs(nodes(j)) == 2;
            C_2 = C_2 + 1;
        elseif Outputs(nodes(j)) == 3;
            C_3 = C_3 + 1;
        elseif Outputs(nodes(j)) == 4;
            C_4 = C_4 + 1;
        elseif Outputs(nodes(j)) == 5;
            C_5 = C_5 + 1;    
        elseif Outputs(nodes(j)) == 6;
            C_6 = C_6 + 1;    
        end
            
    end
    
    %Color the Datapoints
    if max([C_1, C_2, C_3, C_4, C_5, C_6]) == 0
        color = [color,0];
    elseif max([C_1, C_2, C_3, C_4, C_5, C_6]) == C_1
        color = [color, 1];
    elseif max([C_1, C_2, C_3, C_4, C_5, C_6]) == C_2
        color = [color, 2];
    elseif max([C_1, C_2, C_3, C_4, C_5, C_6]) == C_3
        color = [color, 3];
    elseif max([C_1, C_2, C_3, C_4, C_5, C_6]) == C_4
        color = [color, 4];
    elseif max([C_1, C_2, C_3, C_4, C_5, C_6]) == C_5
        color = [color, 5];
    elseif max([C_1, C_2, C_3, C_4, C_5, C_6]) == C_6
        color = [color, 6];    
    end
end

Mat = vec2mat(color, grid_size);

fill([0,8,8,0],[0,0,8,8], 'k')

%Color the classes
for i = 1: length(Mat)
    for j = 1: length(Mat)
        if Mat(i,j) == 0
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'k')
        elseif Mat(i,j) == 1
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'r')
        elseif Mat(i,j) ==2 
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'g')
        elseif Mat(i,j) ==3
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'b')
        elseif Mat(i,j) ==4 
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'c')
        elseif Mat(i,j) ==5 
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'm')
        elseif Mat(i,j) ==6 
            fill([j-1,j,j,j-1],[i-1,i-1,i,i], 'y')    
        end
    end
end

end

