function parms = get_gramm_matrices(parms)

if (isempty(parms.gramm_matrix_train) || isempty(parms.gramm_matrix_test))
    assert(~isempty(parms.gramm_matrices_filename));
    if (exist(parms.gramm_matrices_filename,'file'))
        tmp = load(parms.gramm_matrices_filename);
        parms.gramm_matrix_train = tmp.gramm_matrix_train;        
        parms.gramm_matrix_test = tmp.gramm_matrix_test;
        clear tmp;
    else
        parms = get_image_vectors(parms);
        [parms.gramm_matrix_train parms.gramm_matrix_test] = get_gramm_matrix(parms);        
        save(parms.gramm_matrices_filename,'-struct','parms','gramm_matrix_train');
        save(parms.gramm_matrices_filename,'-struct','parms','gramm_matrix_test','-APPEND');
    end
end
    

