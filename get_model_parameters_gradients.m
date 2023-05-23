function fv = get_model_parameters_gradients(data, gm_parms, softassign, get_normalizer)

D = size(data,1);
N = size(data,2);
K = size(gm_parms.M,2);
assert(size(gm_parms.M,1) == D);
assert(size(gm_parms.Psi,1) == D);
assert(size(gm_parms.Psi,2) == K);
assert(length(gm_parms.mix) == K);

Variance  = gm_parms.Psi;
InvVar    = Variance.^-1;
InvVar2   = InvVar.^2;
Variance2 = Variance.^2;
MyMeans   = gm_parms.M;
MyMeans2  = MyMeans.^2;
MyMeans3  = MyMeans.*MyMeans2;
MyMeans4  = MyMeans2.^2;
        
SumSoftAs = ones(1,N)*softassign;

if strcmp(get_normalizer,'multiplicative')
    softassign  = softassign.^2;    % N x K
    SumSoftAs2  = ones(1,N)*softassign;
        
    W    =   data;     % D x N
    WP2  = W*softassign;     % D x K
    
    W    = W.*data;    % D x N
    W2P2 = W *softassign;    % D x K
    
    W    = W.*data;    % D x N
    W3P2 = W *softassign;    % D x K
    
    W    = W.*data;    % D x N
    W4P2 = W *softassign;    % D x K
    
    SSA2 = ones(D,1)*SumSoftAs2;  % D x K        
    
    d_pi =  N*(gm_parms.mix(:).^2) + SumSoftAs2' -2*gm_parms.mix(:).*SumSoftAs';
    
    d_m  = InvVar2.*W2P2 + SSA2.*InvVar2.*MyMeans2 - 2*InvVar2.*MyMeans.*WP2;
    
    d_S  = SSA2             .* (Variance2 - 2*Variance .* MyMeans2 + MyMeans4) + ...
        WP2              .* (4 * Variance .* MyMeans - 4 * MyMeans3) + ...
        W2P2             .* (6 * MyMeans2 - 2 * Variance) - ...
        W3P2             .* (4 * MyMeans) + ...
        W4P2;
    
    fv =  [ d_pi d_m' d_S'/4 ] / N;
else
    SSA  = ones(D,1)*SumSoftAs; % D x K
    WP   = data * softassign;      % D x K
    data = data.^2;
    W2P  = data*softassign ;  % D x K
    
    d_pi = SumSoftAs' - N*gm_parms.mix(:);
    d_m  = InvVar.*WP - SSA.*InvVar.*MyMeans; % D x K
    d_S  = 2*MyMeans.*WP - W2P + SSA.*(Variance-MyMeans2); % D x K
    fv   = [d_pi d_m' d_S'/2 ] / N;
end    

fv = fv(:);