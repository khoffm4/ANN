function w=ghafun(x,winit,itermax,eta,dp)
% Usage: w=ghafun(x,winit,itermax)
% generalized Hebbian Learning Algorithm for PCA
% Sanger, T. D., "Optimal unsupervised learing 
%   in a signle layer linear feed-forward neural
%   network", Neural Networks, vol. 12, pp. 459-473, 1989.
% (C) 2001 copyright by Yu Hen Hu
% created: 2/25/2001
% x: K by M data matrix. Each row is a data sample
% w, winit: N by M weight matrix.
% eta: learning rate. default = 0.005
% itermax: maximum number of iterations. default = 500. Each iteration
%   entire set of data x will be used once.
% dp = 0 (default, do not display intermediate result)
%    >0  will display new w matrix after each dp iteration
rng(10)
k = 75;
if nargin<5, 
    dp=0;  %default do not display intermediate result
end

if nargin<4,
   eta=0.005; % default value
end
if nargin<3, 
   itermax=500; 
end
[K,M]=size(x);
[N,M]=size(winit);
w=winit;

iter=0;  done=0;   

while done==0,
   iter=iter+1;
   %x=randomize(x);  % randomize order of presenting x
   k = randperm(length(x),K);
   x = x(k,:); 
   for i=1:K,
      y=w*x(i,:)'; % y is N by 1
      u=[];
      for j=1:N,
         u=[u; y(j)*(x(i,:)-y(1:j)'*w(1:j,:))];
      end
      wold=w;
      w=wold+eta*u; % updated weight matrix
   end
   % display intermediate result if desire
   if dp>0,
       if mod(iter,dp)==0, % for every dp iteration
           %disp(['iteration count = ' int2str(iter) ', and W is: ']);
           disp(w)
       end
   end
   if iter==itermax,
      done=1;
   end
end