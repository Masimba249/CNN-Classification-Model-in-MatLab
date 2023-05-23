function parms = get_image_vectors(parms)

if (isempty(parms.image_vectors))
    assert(~isempty(parms.image_vectors_filename));
    if (exist(parms.image_vectors_filename,'file'))
        tmp = load(parms.image_vectors_filename);
        parms.image_vectors = tmp.image_vectors;
        clear tmp;
    else        
        parms.image_vectors = zeros(parms.imagevec_dim, parms.n_images);        
        image_sizes_filename = sprintf('%s/labels/image_sizes.list', parms.root_dir);
        image_sizes = load(image_sizes_filename);
        for image_id = 1:parms.n_images
            image_size = image_sizes(image_id,:);
            data = load(sprintf('%s/data/%06d.mat', parms.root_dir, image_id));
            n_features = size(data.d,2);
            data.d = double(data.d);            
            data_sq = (data.d).^2;
            zerovec_indices = find(sum(data_sq)==0);
            nonzerovec_indices = find(sum(data_sq)~=0);
            data.d(:,nonzerovec_indices) = bsxfun(@rdivide, data.d(:,nonzerovec_indices), sqrt(sum(data_sq(:,nonzerovec_indices))));
            data.d(:,zerovec_indices) = zeros(parms.feat_dim_orig, length(zerovec_indices));                        
            data.f = data.f(1:2,:) ./ (image_size' * ones(1,n_features));
            parms.image_vectors(:,image_id) = get_image_representation(data.f, data.d, parms);
            fprintf('Got representation for %d/%d images\r', image_id, parms.n_images);
        end
        fprintf('\n');
        save(parms.image_vectors_filename,'-struct','parms','image_vectors','-v7.3');
    end
end
