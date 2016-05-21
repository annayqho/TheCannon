import numpy as np

direc = 
id_test = np.load("id_all.npz")['arr_0']
label_test = np.load("label_all.npz")['arr_0'].T
teff_test = label_test[:,0]
logg_test = label_test[:,1]
feh_test = label_test[:,2]
alpha_test = label_test[:,3]
ak_test = label_test[:,4]
chisq_test = np.load("chisq_all.npz")['arr_0']

# add in the training set
direc = "../run_9b_reddening" 
direc = "../run_14_all_abundances_fe_xcalib/high_snr"
id_training = np.load("%s/tr_id.npz" %direc)['arr_0']
label_training = np.load("%s/all_cannon_labels.npz" %direc)['arr_0']
chisq_training = np.load("%s/cannon_label_chisq.npz" %direc)['arr_0']
id_total = np.append(id_test, id_training)
teff_total = np.append(teff_test, label_training[:,0])
logg_total = np.append(logg_test,label_training[:,1])
feh_total = np.append(feh_test,label_training[:,2])
alpha_total = np.append(alpha_test, label_training[:,3])
ak_total = np.append(ak_test,label_training[:,4])
chisq_total = np.append(chisq_test,chisq_training)

id_all, inds_unique = np.unique(id_total, return_index=True)
teff = teff_total[inds_unique]
logg = logg_total[inds_unique]
feh = feh_total[inds_unique]
alpha = alpha_total[inds_unique]
ak = ak_total[inds_unique]
chisq_all = chisq_total[inds_unique]


outfile = "lamost_catalog.csv"
outfile = "lamost_catalog_abundances.csv"
fout = open(outfile, "w")
#header = "id,teff,logg,m_h,alpha_m,a_k,chisq\n"
header = "id," + label_names
fout.write(header)
for i,id_val in enumerate(id_all):
    id_short = id_val.split("/")[-1]
    line = "%s,%s,%s,%s,%s,%s,%s\n" %(id_short, teff[i], logg[i], feh[i], alpha[i], ak[i], chisq_all[i])
    fout.write(line)


fout.flush()
fout.close()
