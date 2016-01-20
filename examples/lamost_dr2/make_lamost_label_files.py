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

all_filenames = all_labels[:,0]
all_dates = np.array([it.split('/')[0] for it in all_filenames])

for date in dates:
    date = "20111203"
    print("Working on date %s" %date)
    if glob.glob("lamost_labels_%s.npz" %date):
        print("done already")
    else:
        files_full = glob.glob("DR2_release/%s/*/*.fits" %date)
        files = np.array([f[12:] for f in files_full])
        inds = np.array([np.where(all_filenames==f)[0][0] for f in files])
        take = all_labels[inds,:]
        np.savez_compressed("lamost_labels_%s" %date, take)
