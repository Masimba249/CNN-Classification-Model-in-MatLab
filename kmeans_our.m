function [A,M,E] = kmeans_jakob(X,K,iters,chunks)
% kmeans clustering
%
% [A,M,E] = kmeans(X,k,iters)
%
% X     - (d x n) d-dimensional input data
% K     -  number of means
%
% returns
% E  - sum of squared distances to nearest mean
% M  - (k x d) matrix of cluster centers
% A  - (n x 1) index of nearest center for each data point
%
% Jakob Verbeek, 2009-2010

[D N]   = size(X);
done    = 0;
iter    = 0;
E       = zeros(iters,1);

if (nargin == 3)
    chunks = 1;
end

if numel(K) > 1;     % initialize with given means, compute assignment only
    M = K;
    K = size(M,2);
    iters = 1;
else
    M = X(:,randi(N,K,1));       
end    

SumNormsX = sum(X(:).^2);

T = zeros(N,1);
A = zeros(N,1);

while ~done
    iter = iter +1;
        
    NormsM = sum(M.^2,1)/2;
       
    first = 1;
    for ch = 1:chunks
        last = first + floor(N/chunks) - 1;
        [T(first:last) A(first:last)]   = max(X(:,first:last)'*M - repmat(NormsM,last-first+1,1),[],2);
        first = last + 1;
    end
    if (first <= N)
        [T(first:N) A(first:N)] = max(X(:,first:N)'*M - repmat(NormsM,N-first+1,1),[],2);
    end
            
    E(iter) =-2*sum(T) + SumNormsX;     
    
    if iter < iters;        
                     
        C = accumarray(A,1,[K 1])'; % counts of the centers
        for d=1:D
            M(d,:) = accumarray(A,X(d,:)',[K 1])';
        end
        M = M .* repmat(C.^-1,D,1);
        
        C0 = find(C==0);        
        M(:,C0) = X(:,randi(N,numel(C0),1));        
    end

    if iter >= iters; done=1;end
    fprintf('Iter %d, error %f\n',iter, E(iter));

end
E = E(1:iter);

% X = rand(1000,2);[A M E] = kmeans(X', 25, 100);clf;plot(X(:,1),X(:,2),'.g',M(1,:),M(2,:),'kx','LineWidth',15);hold on;h = voronoi(M(1,:),M(2,:));for i=1:size(h,1);set(h(i),'LineWidth',3);end;hold off
