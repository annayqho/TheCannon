""" Select test objects by their distance from the average of the 10 nearest-
neighbor APOGEE labels in the training set. """

import numpy as np
import os
import glob

def calc_dist(lamost_point, training_points, coeffs):
    """ dists from one lamost point to all training points """
    diff2 = (training_points - lamost_point)**2
    dist = np.sqrt(np.sum(diff2*coeffs, axis=1))
    return np.mean(dist[dist.argsort()][0:10])


def find_test_obj(date):
    print("loading data...")
    with np.load("../test_training_overlap/tr_label.npz") as a:
        training_points = a['arr_0'][:,0:3]
    labels = np.load("lamost_labels_%s.npz" %date)['arr_0']
    lamost_label_id = labels[:,0]
    lamost_teff = labels[:,1].astype(np.float)
    lamost_logg = labels[:,2].astype(np.float)
    lamost_feh = labels[:,3].astype(np.float)

    lamost_points = np.vstack((lamost_teff, lamost_logg, lamost_feh)).T
    coeffs = 1./(np.array([100,0.2,0.1])**2)

    print("calculating training distances")
    training_dist = np.array(
            [calc_dist(p, training_points, coeffs) for p in lamost_points])

    print("finding the test objects")
    test_obj = lamost_label_id[training_dist < 2.5]

    outputf = open('%s_test_obj.txt' %date, "w")
    for obj in test_obj: 
        outputf.write(obj + '\n')
    outputf.close()


if __name__ == "__main__":
    dates = os.listdir("DR2_release")
    dates = np.array(dates)
    dates = np.delete(dates, np.where(dates=='.directory')[0][0])
    for date in dates: 
        print(date)
        if glob.glob("%s_test_obj.txt" %date):
            print("done already")
        else:
            find_test_obj(date)
