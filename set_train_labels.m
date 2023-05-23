function parms = set_train_abels(parms)

train_labels_filename = sprintf('%s/labels/train_labels.list', parms.root_dir);
train_labels = load(train_labels_filename); 
parms.n_images = length(train_labels); 
% we assume that images are either train (1) or test (0)
parms.train_indices = find(train_labels==1);
parms.test_indices = find(train_labels==0);        
parms.n_train_images = length(parms.train_indices);
parms.n_test_images = length(parms.test_indices);
