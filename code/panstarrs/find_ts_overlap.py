import numpy as np

# load ALL the Pan-STARRS IDs and colors
ids = np.loadtxt("ps_colors.txt", usecols=(0,), dtype='str', delimiter=',')
colors_all = np.loadtxt("ps_colors.txt", usecols=(1,2,3,4,5,6,7,8), dtype='float', delimiter=',')

# now, decide which APOGEE IDs you want Pan-STARRS colors for
# in other words, find all the APOGEE IDs of the training set

training_ids_lamost = np.loadtxt("../tr_files.txt", dtype='str', delimiter=',')
apogee_ids = np.loadtxt("../apogee_dr12_labels.csv", dtype='str', usecols=(1,), delimiter=',')
lamost_ids = np.loadtxt("../apogee_dr12_labels.csv", dtype='str', usecols=(0,), delimiter=',')
inds = np.array([np.where(lamost_ids==training_id_lamost)[0][0] for training_id_lamost in training_ids_lamost])
training_ids_apogee = apogee_ids[inds]

# which training set stars have PanSTARRS colors?

apogee_ids_short = np.array([val[19:37] for val in training_ids_apogee])
intersect = np.intersect1d(ids, apogee_ids_short)
inds = np.array([np.where(ids==intersect_id)[0][0] for intersect_id in intersect])

# pick the apogee ID, lamost ID, and colors of these overlap stars

apogee = ids[inds]
colors = colors_all[inds]
inds = [np.where(apogee_ids_short==val)[0][0] for val in apogee]
lamost = training_ids_lamost[inds]

# the IDs are intersect and the colors are colors
colors = np.vstack((apogee, lamost, colors[:,0], colors[:,1], colors[:,2], colors[:,3], colors[:,4], colors[:,5], colors[:,6], colors[:,7])).T
np.savetxt("ps_colors_ts_overlap.txt", colors, delimiter=',', header='apogee,lamost,gi,gi_err,ri,ri_err,zi,zi_err,yi,yi_err', fmt="%s")
