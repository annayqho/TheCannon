from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import sys

PY3 = sys.version_info[0] > 2

if not PY3:
    range = xrange


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
        # first char is #
        label_names = f.readline()[1:].split('    ')

    IDs = np.loadtxt(filename, usecols=(0,), dtype='str')
    nlabels = len(label_names)
    cols = tuple(range(1,nlabels+1))
    label_values = np.loadtxt(filename, usecols=cols)
    sorted_vals = [val for (ID, val) in sorted(zip(IDs,label_values))]
    sorted_vals = np.array(sorted_vals)
    IDs = np.sort(IDs)
    print("Loaded stellar IDs, format: %s" % IDs[0])
    print("Loaded %s labels:" % nlabels)
    print(label_names)
    return IDs, label_names, sorted_vals
