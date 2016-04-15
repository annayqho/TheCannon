import numpy as np
import matplotlib.pyplot as plt

lines = [4066, 4180, 4428, 4502, 4726, 4760, 4763, 4780, 4882, 4964, 5404, 5488, 5494, 5508, 5545, 5705, 5778, 5780, 5797, 5844, 5850, 6010, 6177, 6196, 6203, 6234, 6270, 6284,
6376, 6379, 6445, 6533, 6614, 6661, 6699,
6887, 6919, 6993, 7224, 7367, 7562, 8621]

def get_lvec(label_vals, pivots):
    nlabels = label_vals.shape[1]
    nstars = label_vals.shape[0]
    # specialized to second-order model
    linear_offsets = label_vals - pivots
    quadratic_offsets = np.array([np.outer(m, m)[np.triu_indices(nlabels)]
                                  for m in (linear_offsets)])
    ones = np.ones((nstars, 1))
    lvec = np.hstack((ones, linear_offsets, quadratic_offsets))
    return lvec

direc ="run_9b_reddening"
cannon = np.load("%s/all_cannon_labels.npz" %direc)['arr_0'] # (9956, 4)
ak = cannon[:,4]
flux = np.load("%s/tr_flux.npz" %direc)['arr_0'] # (9956, 3626)
ivar = np.load("%s/tr_ivar.npz" %direc)['arr_0'] # (9956, 3626)
coeffs = np.load("%s/coeffs.npz" %direc)['arr_0'] # (3626, 15)
s = np.load("%s/scatters.npz" %direc)['arr_0'] # (3626)
lams = np.load("run_2_train_on_good/wl.npz")['arr_0'] # (3626)
npixels = len(lams)

pivots = np.mean(cannon, axis=0) # 4
lvec = get_lvec(cannon, pivots) # 9956, 15
#lvec_full = np.array([lvec,] * npixels) # 3626, 9956, 15

model = np.dot(lvec, coeffs.T)
#np.savez("%s/model_spec_with_Ak.npz" %direc, model)

# now do it again, except with Ak = 0, to get the "unextincted" spectrum
cannon_temp = np.hstack((cannon[:,0:4], np.zeros(len(cannon[:,4]))[:,None]))
pivots_temp = np.append(pivots[0:4], 0)
lvec = get_lvec(cannon_temp, pivots_temp)
model_noAk = np.dot(lvec, coeffs.T)

# now, select all the stars with Ak > 0.1
tr_label = np.load("%s/tr_label.npz" %direc)['arr_0']
choose = np.logical_and(tr_label[:,4] > 0.3, ak > 0.3) # 2887 stars
model_noAk_choose = model_noAk[choose]
model_choose = model[choose]
unextincted_spec = np.median((model_choose - model_noAk_choose), axis=0)


plt.plot(lams, unextincted_spec, c='k')
for line in lines: plt.axvline(x=line, c='r')
plt.show()
#chisq = np.sum((model-flux)**2 * ivar / (1 + ivar * s**2), axis=1)
#ivar = 1/np.abs((flux-model)**2 - s[None,:]**2) # (9956,3626)


#snr_all = flux * ivar**0.5
#snr_obj = np.median(snr_all, axis=1)
#np.savez("snr_all.npz", snr_all)
