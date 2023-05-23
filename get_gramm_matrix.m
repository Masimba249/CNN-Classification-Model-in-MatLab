function [gramm_matrix_train, gramm_matrix_test] = get_gramm_matrix(parms)

normalizations = {{'ADD',@additive_normalization}, ...
                  {'MULT', @multiplicative_normalization}, ...
                  {'POWER', @power_normalization}, ...
                  {'METRIC', @metric_normalization}};

positions_begin = [1 strfind(parms.transform_imagevecs,'+')+1];
positions_end   = [strfind(parms.transform_imagevecs,'+')-1 length(parms.transform_imagevecs)];

for n = 1:length(positions_begin)
    chunk = parms.transform_imagevecs(positions_begin(n):positions_end(n));
    for m = 1:length(normalizations)
        normalization_type = normalizations{m}{1};
        pos_begin = strfind(chunk,normalization_type);
        if (isempty(pos_begin))
            continue;
        else
            pos_end = pos_begin + length(normalization_type) - 1;
            if (strcmp(chunk,normalization_type))
                normalization_parm_val = '';
            else
                normalization_parm_val = str2double(chunk(pos_end+1:length(chunk)));
            end
            % do selected normalization with param value
            parms.image_vectors = normalizations{m}{2}(parms, normalization_parm_val);            
        end
    end
end

switch parms.kernel_type
    case 'LINEAR'
        % TODO: check this
        gramm_matrix_train = parms.image_vectors(:,parms.train_indices)' * parms.image_vectors(:,parms.train_indices);    
        gramm_matrix_test = parms.image_vectors(:,parms.test_indices)' * parms.image_vectors(:,parms.train_indices);            
    case 'CHISQ'
        error 'TODO';
        % assert(no white, l1) 
        % gramm_matrix_train = 1 - distance(parms.image_vectors(parms.train_indices,:), parms.image_vectors(parms.train_indices,:), 'CHISQ');
        % gramm_matrix_test = 1 - distance(parms.image_vectors(parms.test_indices,:), parms.image_vectors(parms.train_indices,:), 'CHISQ');
    case 'INTERSECT'
        error 'TODO';
        % assert(no white, l1)
        % gramm_matrix_train = intersect(parms.image_vectors(parms.train_indices,:), parms.image_vectors(parms.train_indices,:));
        % gramm_matrix_test = intersect(parms.image_vectors(parms.test_indices,:), parms.image_vectors(parms.train_indices,:));
    case 'RBF'
        error 'TODO';
        % TODO: parse distance
        % dist_matrix_train = distance(parms.image_vectors(parms.train_indices,:), parms.image_vectors(parms.train_indices,:), distance_type);
        % rbf_scale = mean(dist_matrix_train(:));
        % gramm_matrix_train = exp(-dist_matrix_train / (2*rbf_scale));
        % dist_matrix_test = distance(parms.image_vectors(parms.test_indices,:), parms.image_vectors(parms.train_indices,:), distance_type);
        % gramm_matrix_test = exp(-dist_matrix_test / (2*rbf_scale));
    otherwise
        error 'Unknown kernel type';
end