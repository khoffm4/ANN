%Creates the Grids for the SOM
function [G] = Grid_creator(grid_dim, grid_size)
G = [];
if grid_dim == 2
    for i  = 1: grid_size
        current_x = i.*ones(1, grid_size);
        current_y = 1:grid_size;
        current_row = [current_x; current_y];
        G = [G,current_row]; 
    end
else 
    for i  = 1: grid_size
        current_x = i.*ones(1, grid_size);
        current_y = 1:grid_size;
        current_row = [current_x; current_y];
        G = [G,current_row]; 
    end
    G =  [G; zeros(1, length(G))];

end
end