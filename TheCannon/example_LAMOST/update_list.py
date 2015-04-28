# update the lamost_sorted_by_ra and apogee_sorted_by_ra files to suit
# which files are actually in the Data/ directory

import numpy as np

lamost = np.loadtxt("example_LAMOST/lamost_sorted_by_ra.txt", dtype=str)
apogee = np.loadtxt("example_DR12/apogee_sorted_by_ra.txt", dtype=str)

# find the indices of the files that are not in the directory 

realstars = os.listdir("example_LAMOST/Data")
remove = np.ones(len(lamost), dtype=bool)
for i in range(len(lamost)):
    if lamost[i] in realstars:
        remove[i] = 0

# lamost[remove] gives you all the filenames you need to remove


