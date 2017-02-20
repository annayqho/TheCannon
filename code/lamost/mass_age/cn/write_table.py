import numpy as np
import sys
sys.path.append("..")
from astropy.table import Table, Column
from astropy.io import ascii
from mass_age_functions import *
from estimate_mass_age import estimate_age
from marie_cuts import get_mask
import glob


def get_err(snr):
    """ Get approximate scatters from SNR
    as determined in the code, snr_test.py
    Order: Teff, logg, MH, CM, NM, alpha """

    quad_terms = np.array(
            [3.11e-3, 1.10e-5, 6.95e-6, 5.05e-6, 4.65e-6, 4.10e-6])
    lin_terms = np.array(
            [-0.869e-1, -2.07e-3, -1.40e-3, -1.03e-3, -1.13e-3, -7.29e-4])
    consts = np.array([104, 0.200, 0.117, 0.114, 0.156, 0.0624])
    err = quad_terms[:,None] * snr**2 + lin_terms[:,None] * snr + consts[:,None]
    return err


REF_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age/with_col_mask/xval_with_cuts"
DATA_DIR = "/Users/annaho/Data/LAMOST/Mass_And_Age/test_step"

ref_id = np.load("%s/ref_id.npz" %REF_DIR)['arr_0']
cannon_ref_err = np.load("%s/xval_cannon_label_errs.npz" %REF_DIR)['arr_0']
cannon_ref_label = np.load("%s/xval_cannon_label_vals.npz" %REF_DIR)['arr_0']
ref_chisq = np.load("%s/xval_cannon_label_chisq.npz" %REF_DIR)['arr_0']
ref_label = np.load("%s/ref_label.npz" %REF_DIR)['arr_0']
ref_snr = np.load("%s/ref_snr.npz" %REF_DIR)['arr_0']

# Excised Values
print("Loading excised data")
data_direc = REF_DIR + "/../excised_obj"
add_id = np.load("%s/excised_ids.npz" %data_direc)['arr_0']
add_err = np.load("%s/excised_cannon_label_errs.npz" %data_direc)['arr_0']
add_cannon = np.load("%s/excised_all_cannon_labels.npz" %data_direc)['arr_0']
add_chisq = np.load("%s/excised_cannon_label_chisq.npz" %data_direc)['arr_0']
add_label = np.load("%s/excised_label.npz" %data_direc)['arr_0']
add_snr = np.load("%s/excised_snr.npz" %data_direc)['arr_0']

# Add
ref_id = np.hstack((ref_id, add_id))
cannon_ref_err = np.vstack((cannon_ref_err, add_err))
cannon_ref_label = np.vstack((cannon_ref_label, add_cannon))
ref_chisq = np.hstack((ref_chisq, add_chisq))
ref_label = np.vstack((ref_label, add_label))
ref_snr = np.hstack((ref_snr, add_snr))

# Calculate mass & age
ref_teff = cannon_ref_label[:,0]
ref_logg = cannon_ref_label[:,1]
ref_feh = cannon_ref_label[:,2]
ref_cm = cannon_ref_label[:,3]
ref_nm = cannon_ref_label[:,4]
ref_afe = cannon_ref_label[:,5]
ref_ak = cannon_ref_label[:,6]
ref_mask = get_mask(ref_teff, ref_logg, ref_feh, ref_cm, ref_nm, ref_afe)
# some of these are actually outside the mask
ref_age, ref_age_err, ref_mass, ref_mass_err = estimate_age(
        ref_label, cannon_ref_label, ref_snr, ref_label, ref_snr)

# Test Values
test_id = np.load("%s/test_id_all.npz" %DATA_DIR)['arr_0']
test_id = np.array([(val.decode('utf-8')).split('/')[-1] for val in test_id])
test_label = np.load("%s/test_label_all.npz" %DATA_DIR)['arr_0'].T
test_err = np.load("%s/test_err_all.npz" %DATA_DIR)['arr_0'].T
test_chisq = np.load("%s/test_chisq_all.npz" %DATA_DIR)['arr_0']
test_snr = np.load("%s/test_snr_all.npz" %DATA_DIR)['arr_0']
teff = test_label[:,0]
logg = test_label[:,1]
feh = test_label[:,2]
cm = test_label[:,3]
nm = test_label[:,4]
afe = test_label[:,5]
test_mask = get_mask(teff, logg, feh, cm, nm, afe) 
age = np.zeros(len(test_id))
age_err = np.ones(age.shape)*100
mass = np.zeros(age.shape)
mass_err = np.ones(mass.shape)*100

age[test_mask], age_err[test_mask], mass[test_mask], mass_err[test_mask] = estimate_age(
        ref_label, cannon_ref_label, ref_snr,
        test_label[test_mask], test_snr[test_mask])

nobj = len(ref_id) + len(test_id)
print(str(nobj) + " objects in total")

ref_flag = np.zeros(nobj, dtype=bool)
ref_flag[0:len(ref_id)] = True

print("writing file")
t = Table()

t['LAMOST_ID'] = np.hstack((ref_id, test_id))
t['is_ref_obj'] = ref_flag 
t['in_martig_range'] = np.hstack((ref_mask, test_mask))

label_names = np.array(
        ['Teff', 'logg', 'MH', 'CM', 'NM', 'AM', 'Ak'])

for ii,name in enumerate(label_names):
    t[name] = np.hstack((cannon_ref_label[:,ii], test_label[:,ii]))

t['Mass'] = np.hstack((ref_mass, mass))
t['logAge'] = np.hstack((ref_age, age))

err_names = np.array(
        ['Teff_err', 'logg_err', 'MH_err', 
        'CM_err', 'NM_err', 'AM_err', 'Ak_err'])

for ii,name in enumerate(err_names):
    t[name] = np.hstack((cannon_ref_err[:,ii], test_err[:,ii]))

t['Mass_err'] = np.hstack((ref_mass_err, mass_err))
t['logAge_err'] = np.hstack((ref_age_err, age_err))

t['SNR'] = np.hstack((ref_snr, test_snr))
scatters = get_err(t['SNR'])
print(scatters.shape)
t['Teff_scatter'] = scatters[0]
t['logg_scatter'] = scatters[1]
t['MH_scatter'] = scatters[2]
t['CM_scatter'] = scatters[3]
t['NM_scatter'] = scatters[4]
t['AM_scatter'] = scatters[5]
t['chisq'] = np.hstack((ref_chisq, test_chisq)) / 1800.0

t.write('lamost_catalog.csv', format='ascii.fast_csv')
