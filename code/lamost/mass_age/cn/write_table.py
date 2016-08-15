import numpy as np
import sys
sys.path.append("..")
from astropy.table import Table, Column
from astropy.io import ascii
from mass_age_functions import *
import glob

DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age"

print("writing file")
t = Table()

# Training Values
ref_id = np.load("%s/ref_id.npz" %DATA_DIR)['arr_0']
cannon_ref_err = np.load("%s/xval_cannon_label_errs.npz" %DATA_DIR)['arr_0']
cannon_ref_label = np.load("%s/xval_cannon_label_vals.npz" %DATA_DIR)['arr_0']
ref_chisq = np.load("%s/xval_cannon_label_chisq.npz" %DATA_DIR)['arr_0']
ref_label = np.load("%s/ref_label.npz" %DATA_DIR)['arr_0']

# Calculate mass
ref_teff = cannon_ref_label[:,0]
ref_logg = cannon_ref_label[:,1]
ref_feh = cannon_ref_label[:,2]
ref_cm = cannon_ref_label[:,3]
ref_nm = cannon_ref_label[:,4]
ref_afe = cannon_ref_label[:,5]
ref_ak = cannon_ref_label[:,6]
ref_mass = calc_mass_2(ref_feh, ref_cm, ref_nm, ref_teff, ref_logg)
ref_age = 10.0**calc_logAge(ref_feh, ref_cm, ref_nm, ref_teff, ref_logg)

# Test Values
test_id = np.load("%s/test_id_all.npz" %DATA_DIR)['arr_0']
test_id = np.array([(val.decode('utf-8')).split('/')[-1] for val in test_id])
test_label = np.load("%s/test_label_all.npz" %DATA_DIR)['arr_0']
test_err = np.load("%s/test_err_all.npz" %DATA_DIR)['arr_0']
test_chisq = np.load("%s/test_chisq_all.npz" %DATA_DIR)['arr_0']

nobj = len(ref_id) + len(test_id)
print(str(nobj) + " objects in total")

ref_flag = np.zeros(nobj)
ref_flag[0:len(ref_id)] = 1


err_names = np.array(
        ['cannon_teff_err', 'cannon_logg_err', 'cannon_feh_err', 
        'cannon_cm_err', 'cannon_nm_err', 'cannon_afe_err'])

t['LAMOST_ID'] = np.hstack((ref_id, test_id))
t['is_ref_obj'] = ref_flag 
t['chisq'] = np.hstack((ref_chisq, test_chisq))

label_names = np.array(
        ['cannon_teff', 'cannon_logg', 'cannon_mh', 
        'cannon_cm', 'cannon_nm', 'cannon_afe', 'cannon_ak'])

for ii,name in enumerate(label_names):
    t[name] = np.hstack((cannon_ref_label[:,ii], test_label[ii,:]))

label_names = np.array(
        ['apogee_teff', 'apogee_logg', 'apogee_mh', 
        'apogee_cm', 'apogee_nm', 'apogee_afe', 'apogee_ak'])

for ii,name in enumerate(label_names):
    filler = np.zeros(len(test_id))
    t[name] = np.hstack((ref_label[:,ii], filler))

t['cannon_mass'] = np.hstack((ref_mass, test_label[6,:]))
t['cannon_age'] = np.hstack((ref_age, test_label[7,:]))

# Calculate (C+N)/M

t['cannon_c_plus_n'] = calc_sum(t['cannon_mh'], t['cannon_cm'], t['cannon_nm'])

for ii,name in enumerate(err_names):
    t[name] = np.hstack((cannon_ref_err[:,ii], test_err[ii,:]))

t.write('lamost_catalog.csv', format='ascii.fast_csv')
