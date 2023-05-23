function image_vector = get_sfv(fPos, fApp, agm_model, sgm_model, normalizer)

if ~isfield(normalizer,'additive')
    get_normalizer='additive';
elseif ~isfield(normalizer,'multiplicative')
    get_normalizer='multiplicative';
else
    get_normalizer='';
end

N = size(fPos,2);
D = size(agm_model.M,1);
assert(size(fApp,1)==D);
assert(size(sgm_model.M,1) == 2);
assert(size(fPos,1)==2);

K = size(agm_model.M,2);
C = numel(sgm_model.mix);
d = 2;
dims_per_word = 1+C*(1+2*d);

spa_assign = (mfa_E_step(fPos, sgm_model))'; % N x C (C = # of qunatization cells in image space)
app_assign = appearance_assign(fApp, agm_model); % N x K

image_vector = zeros(K*dims_per_word,1);

if strcmp(get_normalizer,'multiplicative')
  if (N > 20000)
    N_chunks = 10;
    chunk_size = floor(N / N_chunks);
    chunk_begin = 1;
    for chunk_id = 1:N_chunks 
            chunk_end = chunk_begin + chunk_size - 1;
            image_vector(1:K) = image_vector(1:K) + sum(bsxfun(@power,bsxfun(@minus,app_assign(chunk_begin:chunk_end,:), agm_model.mix'),2) ,1)';
            chunk_begin = chunk_end + 1;
    end
    chunk_end = N;
    image_vector(1:K) = image_vector(1:K) + sum(bsxfun(@power,bsxfun(@minus,app_assign(chunk_begin:chunk_end,:), agm_model.mix'),2) ,1)';
    image_vector(1:K) = image_vector(1:K) / N;
  else
    image_vector(1:K) = mean( bsxfun(@power,bsxfun(@minus,app_assign, agm_model.mix'),2) ,1);
  end
else
    image_vector(1:K) = mean(app_assign,   1)' - agm_model.mix;
end

CellIndx = K+(1:(dims_per_word-1));

for k = 1:K    
    image_vector(CellIndx) = get_model_parameters_gradients(fPos, sgm_model, bsxfun(@times,spa_assign,app_assign(:,k)), get_normalizer);
    CellIndx = CellIndx + dims_per_word - 1;
end

if strcmp(get_normalizer,'additive')
    normalizer.additive = image_vector; % store the computed normalizer
    image_vector = normalizer;          % return normalizer
elseif strcmp(get_normalizer,'multiplicative')
    normalizer.multiplicative =  image_vector - normalizer.additive.^2 ; % compute variance
    ii = normalizer.multiplicative < 1e-10;                             % check for (almost) zero variances, possibly negatives when numerical error yields negative variance estimates    
    normalizer.multiplicative = 1./sqrt(normalizer.multiplicative);     % store inverse standard deviations    
    normalizer.multiplicative(ii) = 0;                                  % set to zero any components with near zero variances, effectively deleted from the representation 
    image_vector = normalizer;                                          % return normalizer
end
