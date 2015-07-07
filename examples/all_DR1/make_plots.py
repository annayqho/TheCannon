import numpy as np
SNR = np.load("test_SNR.npz")['arr_0']
tr_label = np.load("tr_label.npz")['arr_0']
cannon_label = np.load("cannon_label.npz")['arr_0']

scatter(tr_label[1,:], cannon_label[1,:], marker='x', c=SNR, vmin=0, vmax=100, cmap="Greys")
colorbar(label="Formal SNR")
xlabel("LAMOST logg")
ylabel("Cannon logg")
ylim(0.5,4.5)
xlim(0.5,4.5)
title("LAMOST vs. Cannon logg")
savefig("logg.png")

xlim(3500,5800)
ylim(3500,5800)
plot([3500,5800],[3500,5800], c='k')

xlim(-2.0,1.0)
ylim(-2.0,1.0)
plot([-2,1],[-2,1.0], c='k')

scatter(tr_label[1,:][SNR<50], cannon_label[1,:][SNR<50], marker='x', c=SNR[SNR<50], vmin=0, vmax=50, cmap="Greys")
colorbar(label="Formal SNR")
ylim(0.5,4.5)
xlim(0.5,4.5)
title("LAMOST vs. Cannon logg for SNR < 50")
xlabel("LAMOST logg")
ylabel("Cannon logg")
plot([0,5],[0,5],c='k')
savefig("logg_lowSNR.png")

scatter(tr_label[1,:][SNR>50], cannon_label[1,:][SNR>50], marker='x', c=SNR[SNR>50], vmin=50, vmax=150, cmap="Greys")
colorbar(label="Formal SNR")
ylim(0.5,4.5)
xlim(0.5,4.5)
title("LAMOST vs. Cannon logg for SNR > 50")
xlabel("LAMOST logg")
ylabel("Cannon logg")
plot([0,5],[0,5],c='k')
savefig("logg_bigSNR.png")

dteff = abs(tr_label[0,:]-cannon_label[0,:])
dlogg = abs(tr_label[1,:]-cannon_label[1,:])
dfeh = abs(tr_label[2,:]-cannon_label[2,:])

scatter(tr_label[0,:], tr_label[1,:], marker='x', alpha=0.5, c=dteff, cmap="Greys")
gca().invert_yaxis()
gca().invert_xaxis()
draw()
colorbar(label="dTeff")
xlabel("LAMOST Teff")
ylabel("LAMOST logg")
title("dTeff in LAMOST Label Space")
savefig("delta_teff.png")


def calc_dist(lamost_point, training_points, coeffs):
    """ avg dist from one lamost point to nearest 10 training points """
    diff2 = (training_points - lamost_point)**2
    dist = np.sqrt(np.sum(diff2*coeffs, axis=1))
    return np.mean(dist[dist.argsort()][0:10])


def dist(lamost_point, cannon_point, coeffs):
    diff2 = (lamost_point - cannon_point)**2
    dist = np.sqrt(np.sum(diff2*coeffs,axis=1))
    return dist


coeffs = 1./(np.array([100,0.2,0.1])**2)
with np.load("../test_training_overlap/tr_label.npz") as a:
    training_points = a['arr_0'][:,0:3]

lamost_points = tr_label[0:3,:].T
training_dist = np.array([calc_dist(p, training_points, coeffs) for p in lamost_points])
cannon_points = cannon_label[0:3,:].T
diff2 = (lamost_points - cannon_points)**2
dist = np.sqrt(np.sum(diff2*coeffs,axis=1))
scatter(training_dist, dist, marker='x', c=SNR, vmin=0, vmax=100, cmap="Greys")
colorbar(label="Formal SNR")
xlabel("Distance from LAMOST point to avg(10 nearest training points)")
ylabel("Dist from LAMOST point to corresponding Cannon Point")
title("Comparing distances between points in label space")
savefig("distance_comparison.png")


