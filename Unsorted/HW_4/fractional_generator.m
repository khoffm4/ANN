function [z] = fractional_generator(n)
a = 0.1;
b = 1;
x = (b-a).*rand(n,1) + a;
%y = 1./x;
y = 1./x./6-0.92;
z = [x,y];

end