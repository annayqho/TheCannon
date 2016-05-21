import pyfits
import numpy as np

def get_snr_g(name):
    a = pyfits.open("/home/share/LAMOST/DR2/DR2_release/%s" %name)
    val = a[0].header['sn_g']
    return val 

def get_snr_g_all(ids):
    snr_g = np.array([get_snr_g(name) for name in ids])
    return snr_g
