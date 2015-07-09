import os
import numpy as np
import glob

print("Loading dates")
dates = os.listdir("DR2_release")
dates = np.array(dates)
dates = np.delete(dates, np.where(dates=='.directory')[0][0])

print("Loading labels")
all_labels = np.genfromtxt(
    "lamost_labels_all_dates.csv", delimiter=',', dtype=str, skip_header=1)
print("Done loading labels")

filenames = all_labels[:,0]
all_dates = np.array([it.split('/')[0] for it in filenames])

for date in dates:
    print("Working on date %s" %date)
    if glob.glob("lamost_labels_%s.npz" %date):
        print("done already")
    else:
        inds = np.where(all_dates==date)[0]
        take = all_labels[inds,:]
        np.savez_compressed("lamost_labels_%s" %date, take)
