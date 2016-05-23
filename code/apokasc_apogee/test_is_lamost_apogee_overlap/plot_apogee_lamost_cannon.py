import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from math import log10, floor
rc('font', family='serif', size=20)
rc('text', usetex=True)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

def round_2(x):
    if x < 0:
        x = -x
        return -round(x,2-int(floor(log10(x)))-1)
    else:
        return round(x, 2-int(floor(log10(x)))-1)


def load_data():
    print("loading data...")

    tr_IDs_lamost = np.load(
            "../../test_training_overlap/tr_IDs.npz")['arr_0']
    SNR = np.load(
            "../../test_training_overlap/tr_SNRs.npz")['arr_0'][1,:].astype(float)

    # for all tr_IDs, get their LAMOST parameters
    IDs_lamost = np.loadtxt(
        "../../test_training_overlap/lamost_sorted_by_ra_with_dr2_params.txt", usecols=(0,), dtype=(str))
    labels_all_lamost = np.loadtxt(
        "../../test_training_overlap/lamost_sorted_by_ra_with_dr2_params.txt", usecols=(3,4,5), dtype=(float))
    inds = np.array(
            [np.where(IDs_lamost==a)[0][0] for a in tr_IDs_lamost])
    labels_lamost = labels_all_lamost[inds,:]

    # get apogee equivalent of lamost training IDs
    IDs_lamost_all = np.loadtxt(
            "../../apogee_dr12_labels.csv", usecols=(0,),
            delimiter=',', dtype=(str))
    IDs_apogee_all = np.loadtxt(
            "../../apogee_dr12_labels.csv", usecols=(1,),
            delimiter=',', dtype=(str))
    inds = np.array(
            [np.where(IDs_lamost_all==a)[0][0] for a in tr_IDs_lamost])
    tr_IDs_apogee = IDs_apogee_all[inds]

    # get Cannon-APOGEE labels for all APOGEE tr IDs
    IDs_capogee_all = np.load("test_ids.npz")['arr_0']
    IDs_capogee_all = np.array(
            [a.split('/')[-1] for a in IDs_capogee_all])
    labels_capogee_all = np.load("test_labels.npz")['arr_0']
    inds = np.array(
            [np.where(IDs_capogee_all==a)[0][0] for a in tr_IDs_apogee])
    labels_capogee = labels_capogee_all[inds,:]

    return tr_IDs_apogee, SNR, labels_lamost, labels_capogee


def plot_one(title, ax, x, y, lim):
    ax.scatter(x, y-x, marker='x', c='k', alpha=0.5)
    # ax.set_title(r"%s" %title)
    #axarr[0].plot([-100,10000],[-100,10000], c='r')
    ax.axhline(y=0, c='r')
    scat = np.std(y-x)
    scat = round_2(scat)
    bias = np.mean(y-x)
    bias = round_2(bias)
    textstr = "RMS: %s \nBias: %s" %(scat, bias)
    ax.text(0.05,0.95, textstr, ha='left', va='top', transform=ax.transAxes)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    #ymin = -10*scat
    #ymax = 10*scat
    ax.set_ylim(-1*lim, lim)
    #print(ymin, ymax)
    num_up = sum((y-x)>lim)
    num_down = sum((y-x)<-1*lim)
    print("%s above, %s below" %(num_up, num_down))


def plot_row(name, SNR, ax, x_all, y_all, xlabel, ylabel, cut1, cut2, lim):
    #axarr[0].set_ylabel("LAMOST %s" %name)
    #ax[0].set_ylabel(ylabel)
    #ax[1].set_xlabel(xlabel)

    cond = SNR < cut1
    x = x_all[cond]
    y = y_all[cond]
    plot_one(title, ax[0], x, y, lim)

    cond = np.logical_and(SNR>cut1,SNR<cut2)
    x = x_all[cond]
    y = y_all[cond]
    plot_one(title, ax[1], x, y, lim)

    cond = SNR > cut2
    x = x_all[cond]
    y = y_all[cond]
    plot_one(title, ax[2], x, y, lim)


def create_grid():
    fig = plt.figure(figsize=(15,20))
    #plt.locator_params(nbins=5)
    #ax = fig.add_subplot(111)
    #plt.setp(ax.get_yticklabels(), visible=False)
    #plt.setp(ax.get_xticklabels(), visible=False)
    ax00 = fig.add_subplot(331)
    ax01 = fig.add_subplot(332, sharex=ax00, sharey=ax00)
    plt.setp(ax01.get_yticklabels(), visible=False)
    xticks = ax01.xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    ax02 = fig.add_subplot(333, sharex=ax00, sharey=ax00)
    plt.setp(ax02.get_yticklabels(), visible=False)
    xticks = ax02.xaxis.get_major_ticks()
    xticks[0].set_visible(False) 
    ax10 = fig.add_subplot(334)
    ax11 = fig.add_subplot(335, sharex=ax10, sharey=ax10)
    plt.setp(ax11.get_yticklabels(), visible=False)
    xticks = ax11.xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    ax12 = fig.add_subplot(336, sharex=ax10, sharey=ax10)
    plt.setp(ax12.get_yticklabels(), visible=False)
    xticks = ax12.xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    ax20 = fig.add_subplot(337)
    ax21 = fig.add_subplot(338, sharex=ax20, sharey=ax20)
    plt.setp(ax21.get_yticklabels(), visible=False)
    xticks = ax21.xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    ax22 = fig.add_subplot(339, sharex=ax20, sharey=ax20)
    plt.setp(ax22.get_yticklabels(), visible=False)
    xticks = ax22.xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    fig.subplots_adjust(wspace=0)
    fig.subplots_adjust(hspace=0.2)
    axarr = ((ax00,ax01,ax02), (ax10,ax11,ax12), (ax20,ax21,ax22))
    return fig, axarr


if __name__ == "__main__":
    fig, axarr = create_grid()
    axarr[1][0].set_ylabel(r"$\Delta$(LAMOST, Cannon-APOGEE)",
                           labelpad=30)
    #ax.set_title("Big Title", y=1.1)
    axarr[2][1].set_xlabel("Cannon-APOGEE", labelpad=30)
    cut1, cut2 = 50, 100 # SNR cuts
    tr_IDs, SNR, labels_lamost, labels_capogee = load_data()
    cond = SNR < 50
    count = sum(cond)
    title = r"SNR $< %s$ \\ (%s Objects)" %(cut1, count)
    axarr[0][0].set_title(title)
    cond = np.logical_and(SNR>cut1,SNR<cut2)
    count = sum(cond)
    title = r"$%s <$ SNR $< %s$ \\ (%s Objects)" %(cut1, cut2, count)
    axarr[0][1].set_title(title)
    names = [r"$T_{eff}$", r"$\log g$", r"$[Fe/H]$"]
    lims = [1000, 2.6, 0.88]
    cond = SNR > cut2
    count = sum(cond)
    title = r"SNR $> %s$ \\ (%s Objects)" %(cut2, count)
    axarr[0][2].set_title(title)
    for i in range(0,3):
        axarr[i][2].text(1.1, 0.5, names[i], rotation=270, 
                         transform=axarr[i][2].transAxes, va='center')
    for i in range(0,3):
        plot_row(names[i], SNR, axarr[i], 
                labels_capogee[:,i], labels_lamost[:,i], 
                 names[i], names[i], cut1, cut2, lims[i])
    plt.savefig("capogee_lamost.png")
