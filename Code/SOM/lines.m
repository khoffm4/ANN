
%This function makes the lines between the prototypes based on their
%relations in the SOM grid
function lines(G, weights, grid_size, grid_dim)
% If 2 Dimensional
if grid_dim == 2
    for i = 1: length(G)-1
        if mod(i,grid_size) == 0 
            continue
        end
       line([weights(1,i), weights(1,i+1)], [weights(2,i), weights(2,i+1)])
    end
    for i = 1: length(G)-grid_size
       line([weights(1,i), weights(1,i+grid_size)], [weights(2,i), weights(2,i+grid_size)])
    end

%If 3 Dimensional    
elseif grid_dim == 3
    for i = 1: length(G)-1
        if mod(i,grid_size) == 0 
            continue
        end
       line([weights(1,i), weights(1,i+1)], [weights(2,i), weights(2,i+1)], [weights(3,i), weights(3,i+1)])
    end
    
    for i = 1: length(G)-grid_size
       line([weights(1,i), weights(1,i+grid_size)], [weights(2,i), weights(2,i+grid_size)], [weights(3,i), weights(3,i+grid_size)])
    end

    for i = 1:length(G) - grid_size^2
        line([weights(1,i), weights(1,i+grid_size^2)], [weights(2,i), weights(2,i+grid_size^2)], [weights(3,i), weights(3,i+grid_size^2)])
    end

end
end