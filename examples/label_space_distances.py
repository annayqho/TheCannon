import numpy as np
a = np.load("test_results.npz")
test_ID, test_labels = a['arr_0'], a['arr_1']

label_file = "lamost_dr2/lamost_labels_20121125.csv"
lamost_label_id = np.loadtxt(label_file, usecols=(0,), dtype=str, delimiter=',', skiprows=1)
lamost_teff = np.loadtxt(label_file, usecols=(1,), dtype=float, delimiter=',', skiprows=1)
lamost_logg = np.loadtxt(label_file, usecols=(2,), dtype=float, delimiter=',', skiprows=1)
lamost_feh = np.loadtxt(label_file, usecols=(3,), dtype=float, delimiter=',', skiprows=1)

cannon_teff = test_labels[:,0]
cannon_logg = test_labels[:,1]
cannon_feh = test_labels[:,2]
cannon_alpha = test_labels[:,3]

union = np.intersect1d(lamost_label_id, test_ID)

cannon_inds = [np.where(test_ID==item)[0][0] for item in union]
lamost_inds = [np.where(lamost_label_id==item)[0][0] for item in union]

diff_feh = cannon_feh[cannon_inds] - lamost_feh[lamost_inds]
diff_logg = cannon_logg[cannon_inds] - lamost_logg[lamost_inds]
diff_teff = cannon_teff[cannon_inds] - lamost_teff[lamost_inds]

scatter(lamost_teff[lamost_inds], lamost_logg[lamost_inds], marker='x', c=diff_teff)
colorbar(label=r"$\Delta T_{eff}$")
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
draw()
xlabel(r"LAMOST $T_{eff}$")
ylabel(r"LAMOST $\log g$")
title(r"$\Delta T_{eff}$ in LAMOST Label Space")


scatter(lamost_feh[lamost_inds], cannon_feh[cannon_inds], marker='x', c=lamost_teff[lamost_inds])
colorbar(label=r"LAMOST $T_{eff}$")
plot([-2,1],[-2,1], c='k')
xlim(-2,1)
ylim(-2,1)
xlabel(r"LAMOST $[Fe/H]$")
ylabel(r"Cannon $[Fe/H]$")
title("Lamost-Cannon Comparison for $[Fe/H]$")
savefig("cannon_lamost_comparison_feh.png")

scatter(lamost_logg[lamost_inds], cannon_logg[cannon_inds], marker='x', c='k')
xlabel(r"LAMOST $\log g$")
ylabel(r"Cannon $\log g$")
scatter(lamost_logg[lamost_inds], cannon_logg[cannon_inds], marker='x', c=lamost_teff[lamost_inds])
colorbar(label=r"LAMOST $T_{eff}$")
xlabel(r"LAMOST $\log g$")
ylabel(r"Cannon $\log g$")
title("Lamost-Cannon Comparison for $\log g$")
xlim(0.5,5.5)
ylim(0.5,5)
plot([0,10], [0,10], c='k')
savefig("cannon_lamost_comparison_logg.png")

scatter(lamost_teff[lamost_inds], cannon_teff[cannon_inds], marker='x', c=lamost_logg[lamost_inds])
colorbar(label=r"LAMOST $\log g$")
xlabel(r"LAMOST $T_{eff}$")
ylabel(r"Cannon $T_{eff}$")
title("Lamost-Cannon Comparison for $T_{eff}$")
plot([0,10000],[0,10000], c='k')
xlim(3500,9000)
ylim(3500,6000)
savefig("cannon_lamost_comparison_teff.png")

cannon_alpha = test_labels[:,3]
scatter(cannon_alpha[cannon_inds], cannon_feh[cannon_inds], c=lamost_teff[lamost_inds])
scatter(cannon_alpha[cannon_inds], cannon_feh[cannon_inds], c=lamost_teff[lamost_inds], marker='x')
scatter(cannon_feh[cannon_inds], cannon_alpha[cannon_inds], c=lamost_teff[lamost_inds], marker='x')
ylim(-0.5,0.8)
xlabel(r"Cannon $[Fe/H]$")
ylabel(r"Cannon $[\alpha/Fe]$")
title(r"$[Fe/H]$-$[\alpha/Fe]$ Plane for Sample Test Objects")
colorbar(label=r"LAMOST $T_{eff}$")
savefig("cannon_lamost_alpha_feh_plane.png")

