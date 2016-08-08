import numpy as np
import pyfits
from mass_age_functions import asteroseismic_mass
from mass_age_functions import calc_mass_2


# load data
ref_id = np.load("ref_id.npz")['arr_0']
ref_label = np.load("ref_label.npz")['arr_0']
teff = ref_label[:,0]
mh = ref_label[:,2]
logg =ref_label[:,1]
cm = ref_label[:,3]
nm = ref_label[:,4]
a = pyfits.open("apokasc_lamost_overlap.fits")
data = a[1].data
a.close()
apokasc_ids = data['2MASS_ID']
apokasc_ids = np.array([val.strip() for val in apokasc_ids])
nu_max = data['OCT_NU_MAX']
delta_nu = data['OCT_DELTA_NU']
marie_vals = np.load("marie_vals.npz")
marie_ids = marie_vals['arr_0']
marie_masses = marie_vals['arr_1']


keep = np.logical_and(nu_max > -900, delta_nu > -900)
apokasc_ids = apokasc_ids[keep]
nu_max = nu_max[keep]
delta_nu = delta_nu[keep]

# find corresponding 2mass IDs to the LAMOST IDs
direc = "/home/annaho/aida41040/annaho/TheCannon/examples"
apogee_key = np.loadtxt("%s/apogee_sorted_by_ra.txt" %direc, dtype=str)
lamost_key = np.loadtxt("%s/lamost_sorted_by_ra.txt" %direc, dtype=str)
ref_id_2mass = []
for ii,val in enumerate(ref_id):
    ind = np.where(lamost_key==val)[0][0]
    twomass = apogee_key[ind][19:37]
    ref_id_2mass.append(twomass)
ref_id_2mass = np.array(ref_id_2mass)

# find the overlap between LAMOST and APOKASC
#overlap_id = np.intersect1d(ref_id_2mass, apokasc_ids) #1426 objects
overlap_id = np.intersect1d(
        ref_id_2mass, np.intersect1d(marie_ids, apokasc_ids))

# for each ID in ref ID, calculate mass using the astroseismic scaling relation
inds_astr = np.array([np.where(apokasc_ids==val)[0][0] for val in overlap_id])
inds_ref = np.array([np.where(ref_id_2mass==val)[0][0] for val in overlap_id])

m_astr = asteroseismic_mass(
    nu_max[inds_astr], delta_nu[inds_astr], teff[inds_ref])

# for each ID in ref ID, calculate mass using Marie's formula
m_marie = calc_mass_2(
    mh[inds_ref], cm[inds_ref], nm[inds_ref], teff[inds_ref], logg[inds_ref])

