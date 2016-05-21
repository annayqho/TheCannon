import numpy as np

print("loading test data")
id_test = np.load("id_all.npz")['arr_0']
label_test = np.load("label_all.npz")['arr_0'].T
err_test = np.load("errs_all.npz")['arr_0'].T
npix_test = np.load("npix_all.npz")['arr_0']

# when a fit fails, I set the error to -9999
print("loading test errs")
teff_err_test = np.sqrt(err_test[:,0])
logg_err_test = np.sqrt(err_test[:,1])
feh_err_test = np.sqrt(err_test[:,2])
alpha_err_test = np.sqrt(err_test[:,3])
ak_err_test = np.sqrt(err_test[:,4])
chisq_test = np.load("chisq_all.npz")['arr_0']

teff_test = label_test[:,0]
logg_test = label_test[:,1]
feh_test = label_test[:,2]
alpha_test = label_test[:,3]
ak_test = label_test[:,4]

# add in the training set
print("loading training data")
direc = "../run_9b_reddening" 
#direc = "../run_14_all_abundances_fe_xcalib/high_snr"
id_training = np.load("%s/tr_id.npz" %direc)['arr_0']
ngoodpix_training = np.sum(np.load("%s/tr_ivar.npz" %direc)['arr_0'] > 0, axis=1)
label_training = np.load("%s/all_cannon_labels.npz" %direc)['arr_0']
err_training = np.load("%s/cannon_label_errs.npz" %direc)['arr_0']
chisq_training = np.load("%s/cannon_label_chisq.npz" %direc)['arr_0']
id_total = np.append(id_test, id_training)
teff_total = np.append(teff_test, label_training[:,0])
teff_err_total = np.append(teff_err_test, err_training[:,0])
logg_total = np.append(logg_test,label_training[:,1])
logg_err_total = np.append(logg_err_test,err_training[:,1])
feh_total = np.append(feh_test,label_training[:,2])
feh_err_total = np.append(feh_err_test,err_training[:,2])
alpha_total = np.append(alpha_test, label_training[:,3])
alpha_err_total = np.append(alpha_err_test, err_training[:,3])
ak_total = np.append(ak_test, label_training[:,4])
ak_err_total = np.append(ak_err_test,err_training[:,4])
npix_total = np.append(npix_test, ngoodpix_training)
chisq_total = np.append(chisq_test,chisq_training)

print("finding unique values")
id_all, inds_unique = np.unique(id_total, return_index=True)
teff = teff_total[inds_unique]
teff_err = teff_err_total[inds_unique]
logg = logg_total[inds_unique]
logg_err = logg_err_total[inds_unique]
feh = feh_total[inds_unique]
feh_err = feh_err_total[inds_unique]
alpha = alpha_total[inds_unique]
alpha_err = alpha_err_total[inds_unique]
ak = ak_total[inds_unique]
ak_err = ak_err_total[inds_unique]
npix = npix_total[inds_unique]
chisq_all = chisq_total[inds_unique]

print("writing file")
outfile = "lamost_catalog.csv"
#outfile = "lamost_catalog_abundances.csv"
fout = open(outfile, "w")
header = "id,teff,logg,m_h,alpha_m,a_k,teff_err,logg_err,mh_err,alpha_err,ak_err,ngoodpix,chisq\n"
#header = "id," + label_names
fout.write(header)
for i,id_val in enumerate(id_all):
    id_short = id_val.split("/")[-1]
    line = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" %(
            id_short, teff[i], logg[i], feh[i], alpha[i], ak[i], 
            teff_err[i], logg_err[i], feh_err[i], alpha_err[i], ak_err[i],
            npix[i], chisq_all[i])
    fout.write(line)


fout.flush()
fout.close()
