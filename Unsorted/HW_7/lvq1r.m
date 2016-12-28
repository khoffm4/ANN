function W = lvq1r(X,CX,m,mu,maxiter)
%
% W =      LVQ1(X,CX,mu,maxiter) computes the weight
%          matrix for learning vector quantization 1 (LVQ1)
% 
% X:       is the matrix of inputs, i.e., each column
%          vector is an input
% CX:      is a row vector of scalar "classes" associated
%          with the column vectors of X
% m:       number of different classes
% mu:      initial learning rate parameter
% maxiter: maximum number of training epochs (iterations)     
%         
[r,N]=size(X);
% Initialize the weight vectors randomly
W=0.1*randn(r,m);
CW=[1:m]; % classes for weight vectors
snorm=zeros(1,m);
niter=1;
while niter <= maxiter
   if niter == 1
      for i=m+1:N
         for j=1:m
          snorm(1,j)=norm(X(:,i)-W(:,j))^2;
    end
    [mind,index]=min(snorm);
    if CX(i)==CW(index)
       W(:,index)=W(:,index)+mu*(X(:,i)-W(:,index));
    else
       W(:,index)=W(:,index)-mu*(X(:,i)-W(:,index));
       end
       end 
 else
    for i=1:N
       for j=1:m
          snorm(1,j)=norm(X(:,i)-W(:,j))^2;
       end
[mind,index]=min(snorm);
if CX(i)==CW(index)
       W(:,index)=W(:,index)+(mu/niter)*(X(:,i)-W(:,index));
    else
       W(:,index)=W(:,index)-(mu/niter)*(X(:,i)-W(:,index));
    end
    end
    end
 niter=niter+1;
end