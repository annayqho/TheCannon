""" Make a table out of the test labels """

import numpy as np

test_id = np.load("test_id.npz")['arr_0']
lab = np.load("test_cannon_labels.npz")['arr_0']
errs = np.load("test_errs.npz")['arr_0']

nobj = len(test_id)

outputf = open("science_parameters.txt", "w")
outputf.write("filename,Teff,logg,FeH,Teff_err,logg_err,FeH_err \n")

for ii,filename in enumerate(test_id):
    f = filename.split("/")[1]
    teff = lab[ii,0]
    logg = lab[ii,1]
    feh = lab[ii,2]
    teff_err = errs[ii,0]
    logg_err = errs[ii,1]
    feh_err = errs[ii,2]
    line = "%s,%s,%s,%s,%s,%s,%s \n" %(
            f,teff,logg,feh,teff_err,logg_err,feh_err)
    outputf.write(line)

outputf.close()
