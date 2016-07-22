""" Make a table out of the test labels """

import numpy as np

DATA_DIR = '/Users/annaho/Data/AAOmega/Run_13_July'

test_id = np.load("%s/test_id.npz" %DATA_DIR)['arr_0']
lab = np.load("%s/test_cannon_labels.npz" %DATA_DIR)['arr_0']
errs = np.load("%s/test_errs.npz" %DATA_DIR)['arr_0']

nobj = len(test_id)

outputf = open("science_parameters.txt", "w")
outputf.write("filename,Teff,logg,FeH,aFe,Teff_err,logg_err,FeH_err,aFe_err \n")

for ii,filename in enumerate(test_id):
    f = filename.split("/")[1]
    teff = lab[ii,0]
    logg = lab[ii,1]
    feh = lab[ii,2]
    afe = lab[ii,3]
    teff_err = errs[ii,0]
    logg_err = errs[ii,1]
    feh_err = errs[ii,2]
    afe_err = errs[ii,3]
    line = "%s,%s,%s,%s,%s,%s,%s,%s,%s \n" %(
            f,teff,logg,feh,afe,teff_err,logg_err,feh_err,afe_err)
    outputf.write(line)

outputf.close()
