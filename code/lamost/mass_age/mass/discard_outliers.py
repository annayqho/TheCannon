import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ids = np.load("tr_id.npz")['arr_0']
chisq = np.load("cannon_label_chisq.npz")['arr_0']
ivar =np.load("tr_ivar.npz")['arr_0']
npix = 3626-np.sum(ivar==0, axis=1)
redchisq = 3*chisq / npix
np.savez("redchisq.npz", redchisq)
good = redchisq <= 10
plt.hist(redchisq, bins=100, range=(0,20))
new_ts = ids[good]
np.savez("tr_ids_good.npz", new_ts)
plt.savefig('redchisq.png')
