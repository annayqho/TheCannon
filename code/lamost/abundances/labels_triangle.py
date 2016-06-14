import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
sys.path.insert(0, '/home/annaho/aida41040/annaho/TheCannon/TheCannon')
sys.path.insert(0, '/home/annaho/aida41040/annaho/TheCannon')

rc('text', usetex=True)
rc('font', family='serif')
from helpers.triangle import corner

#label_names = ['$T_{eff}$', '$\log g$', '[M/Fe]', r'$[\alpha/Fe]$', '[Al/Fe]', '[Ca/Fe]',
#               'C', 'Fe/H', 'K', 'Mg', 'Mn','Na', 'Ni', 'N', 'O', 'Si', 'S',
#               'Ti', 'V']
label_names = ['$T_{eff}$', '$\log g$', '[Al/Fe]', '[Ca/Fe]',
               'C', 'Fe/H', 'K', 'Mg', 'Mn','Na', 'Ni', 'N', 'O', 'Si', 'S',
               'Ti', 'V']
label_vals = np.load("run_14_all_abundances_fparams/all_cannon_labels.npz")['arr_0']
#label_vals_rotated = np.zeros(label_vals.shape)
#FeH = label_vals[:,7]
#teff = label_vals[:,0]
#logg = label_vals[:,1]
#MH = label_vals[:,2]
#alphaM = label_vals[:,3]
#alphaFe = alphaM-FeH+MH
#MFe = MH - FeH

#label_vals_rotated[:,0] = teff
#label_vals_rotated[:,1] = logg
#label_vals_rotated[:,2] = MFe
#label_vals_rotated[:,3] = alphaFe
#label_vals_rotated[:,4:] = label_vals[:,4:] - FeH[:,None]

# some of the labels are bad...
good = np.min(label_vals, axis=1) > -500
abundances = label_vals[good,2:]
element_names = np.array(label_names[2:])
# plot Fe/H, not Fe/Fe

#abundances[:,4] = FeH[good]
fig = corner(abundances, labels=element_names, show_titles=True,
        title_args={"fontsize":12})
#plt.show()
plt.savefig("abundances_triangle.png")
