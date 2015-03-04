import os
import pyfits
import numpy as np
from lamost import LamostDataset

def make_training_set(num):
    """ Make a training set with the top num stars, sorted by S/N. """
    dataset = LamostDataset("example_LAMOST/Data",
                            "example_LAMOST/Data",
                            "example_LAMOST/reference_labels.csv")
    SNRs = dataset.
