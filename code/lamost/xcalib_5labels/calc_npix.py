""" Calculate the number of pixels with ivar > 0 in each spectrum """

import os
import numpy as np

dates = os.listdir("/home/share/LAMOST/DR2/DR2_release")
dates = np.array(dates)
dates = np.delete(dates, np.where(dates=='.directory')[0][0])
dates = np.delete(dates, np.where(dates=='all_folders.list')[0][0])
dates = np.delete(dates, np.where(dates=='dr2.lis')[0][0])

#dates = ["20120201"]

direc = "../xcalib_4labels/output"

for date in dates:
    print(date)
    test_ivar = np.load("%s/%s_norm.npz" %(direc,date))['arr_1']
    ngood = np.sum(test_ivar > 0, axis=1)
    np.savez("%s/%s_npix.npz" %(direc,date), ngood)
