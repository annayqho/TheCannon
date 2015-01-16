import numpy as np

def get_training_labels(filename):
    """Extracts training labels from file.

    Assumes that the file has # then label names in first row, that first
    column is the ID (string), that the remaining values are floats
    and that you want all of the labels. User picks specific labels later. 

    Input: name(string) of the data file containing the labels
    Returns: label_names list, and np ndarray (size=numtrainingstars, nlabels)
    consisting of all of the label values
    """

    with open(filename, 'r') as f:
        all_labels = f.readline().split() # ignore the hash
    label_names = all_labels[1:]
    IDs = np.loadtxt(filename, usecols = (0,), dtype='string')
    print "Loaded stellar IDs, format: %s" %IDs[0]
    nlabels = len(label_names)
    cols = tuple(xrange(1,nlabels+1))
    label_values = np.loadtxt(filename, usecols=cols)
    print "Loaded %s labels:" %nlabels
    print label_names
    return IDs, label_names, label_values


