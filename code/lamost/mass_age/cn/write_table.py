import numpy as np
import glob

print("writing file")
outfile = "lamost_catalog.csv"
fout = open(outfile, "w")

files = np.array(glob.glob("*all.npz"))
names = ['_'.join(item.split('_')[0:-1]) for item in files]
header = ','.join(names)
fout.write(header)

id_test = np.load("id_all.npz")['arr_0']

for i,id_val in enumerate(id_test):
    vals =  np.array([np.load(f)['arr_0'][i] for f in files])
    id_val = vals[files == 'id_all.npz'][0].decode("utf-8")
    id_short = id_val.split("/")[-1]
    vals[files == 'id_all.npz'] = id_short
    vals = [val.decode("utf-8") for val in vals]
    line = ','.join(vals) + '\n'
    fout.write(line)

fout.flush()
fout.close()

# print("loading test data")
# teff_test = np.load("teff_all.npz")['arr_0']
# logg_test = np.load("logg_all.npz")['arr_0']
# feh_test = np.load("feh_all.npz")['arr_0']
# cm_test = np.load("cm_all.npz")['arr_0']
# nm_all = np.load("nm_all.npz")['arr_0']
# alpha_test = np.load("alpha_all.npz")['arr_0']
# 
# # when a fit fails, I set the error to -9999
# print("loading test errs")
# teff_err_test = np.sqrt(np.load("teff_err_all.npz")['arr_0'])
# logg_err_test = np.sqrt(np.load("logg_err_all.npz")['arr_0'])
# feh_err_test = np.sqrt(np.load("feh_err_all.npz")['arr_0'])
# cm_err_test = np.sqrt(np.load("cm_err_all.npz")['arr_0'])
# nm_err_test = np.sqrt(np.load("nm_err_all.npz")['arr_0'])
# alpha_err_test = np.sqrt(np.load("alpha_err_all.npz")['arr_0'])
# chisq_test = np.load("chisq_all.npz")['arr_0']
# 
# npix_test = np.load("npix_all.npz")['arr_0']

# add in the training set
# print("loading training data")
# direc = "../xcalib_5labels" 
# direc1 = "/Users/annaho/TheCannon/data/lamost_paper"
# id_training = np.load("%s/ref_id.npz" %direc1)['arr_0']
# ngoodpix_training = np.sum(
#         np.load("%s/ref_ivar.npz" %direc1)['arr_0'] > 0, axis=1)
# label_training = np.load("%s/all_cannon_label_vals.npz" %direc)['arr_0']
# err_training = np.load("%s/all_cannon_label_errs.npz" %direc)['arr_0']
# chisq_training = np.load("%s/all_cannon_label_chisq.npz" %direc)['arr_0']
# id_total = np.append(id_test, id_training)
# teff_total = np.append(teff_test, label_training[:,0])
# teff_err_total = np.append(teff_err_test, err_training[:,0])
# logg_total = np.append(logg_test,label_training[:,1])
# logg_err_total = np.append(logg_err_test,err_training[:,1])
# feh_total = np.append(feh_test,label_training[:,2])
# feh_err_total = np.append(feh_err_test,err_training[:,2])
# alpha_total = np.append(alpha_test, label_training[:,3])
# alpha_err_total = np.append(alpha_err_test, err_training[:,3])
# ak_total = np.append(ak_test, label_training[:,4])
# ak_err_total = np.append(ak_err_test,err_training[:,4])
# npix_total = np.append(npix_test, ngoodpix_training)
# chisq_total = np.append(chisq_test,chisq_training)
# 
# print("finding unique values")
# id_all, inds_unique = np.unique(id_total, return_index=True)
# teff = teff_total[inds_unique]
# teff_err = teff_err_total[inds_unique]
# logg = logg_total[inds_unique]
# logg_err = logg_err_total[inds_unique]
# feh = feh_total[inds_unique]
# feh_err = feh_err_total[inds_unique]
# alpha = alpha_total[inds_unique]
# alpha_err = alpha_err_total[inds_unique]
# ak = ak_total[inds_unique]
# ak_err = ak_err_total[inds_unique]
# npix = npix_total[inds_unique]
# chisq_all = chisq_total[inds_unique]
 
