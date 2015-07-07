import numpy as np
import matplotlib.pyplot as plt

SNR = np.load("../all_DR1/test_SNR.npz")['arr_0']
tr_label = np.load("../all_DR1/tr_label.npz")['arr_0']
cannon_label = np.load("../all_DR1/cannon_label.npz")['arr_0']

cut1 = 50
cut2 = 100

# For Teff

fig,axarr=plt.subplots(1,3, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0)
axarr[1].set_xlabel("LAMOST Teff")
axarr[0].set_ylabel("Cannon Teff")

cond = SNR < cut1
axarr[0].scatter(tr_label[0,:][cond], cannon_label[0,:][cond], marker='x', c='k', alpha=0.5)
count = sum(cond)
axarr[0].set_title("SNR < %s \n (%s Objects)" %(cut1, count))
axarr[0].set_xlim(3500,6000)
axarr[0].set_ylim(3500,6000)
axarr[0].plot([3500,6000],[3500,6000], c='k')

cond = np.logical_and(SNR>cut1,SNR<cut2)
axarr[1].scatter(tr_label[0,:][cond], cannon_label[0,:][cond], marker='x', c='k', alpha=0.5)
count = sum(cond)
axarr[1].set_title("%s < SNR < %s \n (%s Objects)" %(cut1,cut2,count))
axarr[1].plot([3500,6000],[3500,6000], c='k')

cond = SNR > cut2
axarr[2].scatter(tr_label[0,:][cond], cannon_label[0,:][cond], marker='x', c='k', alpha=0.5)
count = sum(cond)
axarr[2].set_title("SNR > %s \n (%s Objects)" %(cut2, count))
axarr[2].plot([3500,6000],[3500,6000], c='k')

# For logg

i = 1
fig,axarr=plt.subplots(1,3, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0)
axarr[1].set_xlabel("LAMOST logg")
axarr[0].set_ylabel("Cannon logg")

cond = SNR < cut1
axarr[0].scatter(tr_label[i,:][cond], cannon_label[i,:][cond], marker='x', c='k', alpha=0.5)
count = sum(cond)
axarr[0].set_title("SNR < %s \n (%s Objects)" %(cut1, count))
axarr[0].set_xlim(0,5)
axarr[0].set_ylim(0,5)
axarr[0].plot([0,5.0],[0,5.0], c='k')

cond = np.logical_and(SNR>cut1,SNR<cut2)
axarr[1].scatter(tr_label[i,:][cond], cannon_label[i,:][cond], marker='x', c='k', alpha=0.5)
count = sum(cond)
axarr[1].set_title("%s < SNR < %s \n (%s Objects)" %(cut1,cut2,count))
axarr[1].plot([0,5.0],[0,5.0], c='k')

cond = SNR > cut2
axarr[2].scatter(tr_label[i,:][cond], cannon_label[i,:][cond], marker='x', c='k', alpha=0.5)
count = sum(cond)
axarr[2].set_title("SNR > %s \n (%s Objects)" %(cut2, count))
axarr[2].plot([0,5],[0,5], c='k')

# For FeH

i = 2
xmin = -1.5
xmax = 1

fig,axarr=plt.subplots(1,3, sharex=True, sharey=True)
fig.subplots_adjust(wspace=0)
axarr[1].set_xlabel("LAMOST FeH")
axarr[0].set_ylabel("Cannon FeH")

cond = SNR < cut1
axarr[0].scatter(tr_label[i,:][cond], cannon_label[i,:][cond], marker='x', c='k', alpha=0.5)
count = sum(cond)
axarr[0].set_title("SNR < %s \n (%s Objects)" %(cut1, count))
axarr[0].set_xlim(xmin,xmax)
axarr[0].set_ylim(xmin,xmax)
axarr[0].plot([xmin, xmax],[xmin, xmax], c='k')

cond = np.logical_and(SNR>cut1,SNR<cut2)
axarr[1].scatter(tr_label[i,:][cond], cannon_label[i,:][cond], marker='x', c='k', alpha=0.5)
count = sum(cond)
axarr[1].set_title("%s < SNR < %s \n (%s Objects)" %(cut1,cut2,count))
axarr[1].plot([xmin,xmax],[xmin,xmax], c='k')

cond = SNR > cut2
axarr[2].scatter(tr_label[i,:][cond], cannon_label[i,:][cond], marker='x', c='k', alpha=0.5)
count = sum(cond)
axarr[2].set_title("SNR > %s \n (%s Objects)" %(cut2, count))
axarr[2].plot([xmin,xmax],[xmin,xmax], c='k')


