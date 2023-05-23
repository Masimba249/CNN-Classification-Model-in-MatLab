function parms = set_class_labels(parms)

class_labels_filename = sprintf('%s/labels/class_labels.list', parms.root_dir);
parms.class_labels = load(class_labels_filename);     
parms.n_class = size(parms.class_labels,2);