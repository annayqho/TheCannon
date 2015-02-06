from __future__ import (absolute_import, division, print_function, unicode_literals)

import numpy as np

def get_reference_labels(filename):
    """Extracts training labels from file.

    Assumes that the file has # then label names in first row, that first
    column is the filename, that the remaining values are floats
    and that you want all of the labels. User picks specific labels later. 

    Input: name(string) of the data file containing the labels
    Returns: label_names list, and np ndarray (size=numtrainingstars, nlabels)
    consisting of all of the label values
    """

    with open(filename, 'r') as f:
        all_labels = f.readline().split('  ')
    all_labels = filter(None, all_labels)
    label_names = all_labels[1:] # ignore the hash
    IDs = np.loadtxt(filename, usecols = (0,), dtype='string')
    nlabels = len(label_names)
    cols = tuple(xrange(1,nlabels+1))
    label_values = np.loadtxt(filename, usecols=cols)
    sorted_vals = [val for (ID, val) in sorted(zip(IDs,label_values))]
    sorted_vals = np.array(sorted_vals)
    IDs = np.sort(IDs) 
    print("Loaded stellar IDs, format: %s" %IDs[0])
    print("Loaded %s labels:" %nlabels)
    print(label_names)
    return IDs, label_names, sorted_vals
