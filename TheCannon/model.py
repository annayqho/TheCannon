from .dataset import Dataset
from .train_model import _train_model 
from .train_model import _train_model_new
from .train_model import _get_lvec
from .infer_labels import _infer_labels
from .helpers.corner import corner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from copy import deepcopy

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

class CannonModel(object):
    def __init__(self, order, useErrors):
        self.coeffs = None
        self.scatters = None
        self.chisqs = None
        self.pivots = None
        self.scales = None
        self.new_tr_labels = None
        self.order = order
        self.model_spectra = None
        self.useErrors = useErrors


    def model(self):
        """ Return the model definition or raise an error if not trained """
        if self.coeffs is None:
            raise RuntimeError('Model not trained')
        else:
            return self.coeffs


    def train(self, ds):
        """ Run training step: solve for best-fit spectral model """
        if self.useErrors:
            self.coeffs, self.scatters, self.new_tr_labels, self.chisqs, self.pivots, self.scales = _train_model_new(ds)
        else:
            self.coeffs, self.scatters, self.chisqs, self.pivots, self.scales = _train_model(ds)

    def diagnostics(self):
        """ Produce a set of diagnostics plots about the model. """
        _model_diagnostics(self.dataset, self.model)


    def infer_labels(self, ds, starting_guess = None):
        """
        Uses the model to solve for labels of the test set, updates Dataset
        Then use those inferred labels to set the model.test_spectra attribute

        Parameters
        ----------
        ds: Dataset
            Dataset that needs label inference

        Returns
        -------
        errs_all: ndarray
            Covariance matrix of the fit
        """
        return _infer_labels(self, ds, starting_guess)


    def infer_spectra(self, ds):
        """ 
        After inferring labels for the test spectra,
        infer the model spectra and update the dataset
        model_spectra attribute.
        
        Parameters
        ----------
        ds: Dataset object
        """
        lvec_all = _get_lvec(ds.test_label_vals, self.pivots, self.scales, derivs=False)
        self.model_spectra = np.dot(lvec_all, self.coeffs.T)


    def plot_contpix(self, x, y, contpix_x, contpix_y, figname):
        """ Plot baseline spec with continuum pix overlaid 

        Parameters
        ----------
        """
        fig, axarr = plt.subplots(2, sharex=True)
        plt.xlabel(r"Wavelength $\lambda (\AA)$")
        plt.xlim(min(x), max(x))
        ax = axarr[0]
        ax.step(x, y, where='mid', c='k', linewidth=0.3,
                label=r'$\theta_0$' + "= the leading fit coefficient")
        ax.scatter(contpix_x, contpix_y, s=1, color='r',
                label="continuum pixels")
        ax.legend(loc='lower right', 
                prop={'family':'serif', 'size':'small'})
        ax.set_title("Baseline Spectrum with Continuum Pixels")
        ax.set_ylabel(r'$\theta_0$')
        ax = axarr[1]
        ax.step(x, y, where='mid', c='k', linewidth=0.3,
             label=r'$\theta_0$' + "= the leading fit coefficient")
        ax.scatter(contpix_x, contpix_y, s=1, color='r',
                label="continuum pixels")
        ax.set_title("Baseline Spectrum with Continuum Pixels, Zoomed")
        ax.legend(loc='upper right', prop={'family':'serif', 
            'size':'small'})
        ax.set_ylabel(r'$\theta_0$')
        ax.set_ylim(0.95, 1.05)
        print("Diagnostic plot: fitted 0th order spec w/ cont pix")
        print("Saved as %s.png" % (figname))
        plt.savefig(figname)
        plt.close()


    def diagnostics_contpix(self, data, nchunks=10, fig = "baseline_spec_with_cont_pix"):
        """ Call plot_contpix once for each nth of the spectrum """
        if data.contmask is None:
            print("No contmask set")
        else:
            coeffs_all = self.coeffs
            wl = data.wl
            baseline_spec = coeffs_all[:,0]
            contmask = data.contmask
            contpix_x = wl[contmask]
            contpix_y = baseline_spec[contmask]
            rem = len(wl)%nchunks
            wl_split = np.array(np.split(wl[0:len(wl)-rem],nchunks))
            baseline_spec_split = np.array(
                    np.split(baseline_spec[0:len(wl)-rem],nchunks))
            nchunks = wl_split.shape[0]
            for i in range(nchunks):
                fig_chunk = fig + "_%s" %str(i)
                wl_chunk = wl_split[i,:]
                baseline_spec_chunk = baseline_spec_split[i,:]
                take = np.logical_and(
                        contpix_x>wl_chunk[0], contpix_x<wl_chunk[-1])
                self.plot_contpix(
                        wl_chunk, baseline_spec_chunk, 
                        contpix_x[take], contpix_y[take], fig_chunk)


    def diagnostics_leading_coeffs(self, ds):
        label_names = ds.get_plotting_labels()
        lams = ds.wl
        npixels = len(lams)
        pivots = self.pivots
        nlabels = len(pivots)
        chisqs = self.chisqs
        coeffs = self.coeffs
        scatters = self.scatters

        # Leading coefficients for each label & scatter
        fig, axarr = plt.subplots(nlabels+1, figsize=(8,8), sharex=True)
        ax1 = axarr[0]
        plt.subplots_adjust(hspace=0.001)
        nbins = len(ax1.get_xticklabels())
        for i in range(1,nlabels+1):
            axarr[i].yaxis.set_major_locator(
                    MaxNLocator(nbins=nbins, prune='upper'))
        plt.xlabel(r"Wavelength $\lambda (\AA)$", fontsize=14)
        plt.xlim(np.min(lams), np.max(lams))
        plt.tick_params(axis='x', labelsize=14)
        axarr[0].set_title(
                "First-Order Fit Coeffs and Scatter from the Spectral Model",
                fontsize=14)
        axarr[0].locator_params(axis='x', nbins=10)
        first_order = np.zeros((len(coeffs[:,0]), nlabels))
        for i in range(0, nlabels):
            ax = axarr[i]
            lbl = r'$%s$'%label_names[i]
            ax.set_ylabel(lbl, fontsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.xaxis.grid(True)
            y = coeffs[:,i+1]
            first_order[:, i] = y
            ax.step(lams, y, where='mid', linewidth=0.5, c='k')
            ax.locator_params(axis='y', nbins=4)
        ax = axarr[nlabels]
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylabel("scatter", fontsize=14)
        top = np.max(scatters[scatters < 0.8])
        stretch = np.std(scatters[scatters < 0.8])
        ax.set_ylim(0, top + stretch)
        ax.step(lams, scatters, where='mid', c='k', linewidth=0.7)
        ax.xaxis.grid(True)
        ax.locator_params(axis='y', nbins=4)
        print("Diagnostic plot: leading coeffs and scatters across wavelength.")
        return fig


    def diagnostics_leading_coeffs_triangle(self, ds, 
            figname = "leading_coeffs_triangle.png"):
        label_names = ds.get_plotting_labels()
        lams = ds.wl
        pivots = self.pivots
        npixels = len(lams)
        nlabels = len(pivots)
        chisqs = self.chisqs
        coeffs = self.coeffs
        first_order = coeffs[:,1:1+nlabels]
        scatters = self.scatters

        # triangle plot of the higher-order coefficients
        labels = [r"$%s$" % l for l in label_names]
        fig = corner(first_order, labels=labels, show_titles=True,
                     title_args = {"fontsize":12})
        filename = "leading_coeffs_triangle.png"
        print("Diagnostic plot: triangle plot of leading coefficients")
        fig.savefig(figname)
        print("Saved as %s" %figname)
        plt.close(fig)


    def diagnostics_plot_chisq(self, ds, figname = "modelfit_chisqs.png"):
        """ Produce a set of diagnostic plots for the model 

        Parameters
        ----------
        (optional) chisq_dist_plot_name: str
            Filename of output saved plot
        """
        label_names = ds.get_plotting_labels()
        lams = ds.wl
        pivots = self.pivots
        npixels = len(lams)
        nlabels = len(pivots)
        chisqs = self.chisqs
        coeffs = self.coeffs
        scatters = self.scatters

        # Histogram of the chi squareds of ind. stars
        plt.hist(np.sum(chisqs, axis=0), color='lightblue', alpha=0.7,
                bins=int(np.sqrt(len(chisqs))))
        dof = len(lams) - coeffs.shape[1]   # for one star
        plt.axvline(x=dof, c='k', linewidth=2, label="DOF")
        plt.legend()
        plt.title("Distribution of " + r"$\chi^2$" + " of the Model Fit")
        plt.ylabel("Count")
        plt.xlabel(r"$\chi^2$" + " of Individual Star")
        print("Diagnostic plot: histogram of the red chi squareds of the fit")
        print("Saved as %s" %figname)
        plt.savefig(figname)
        plt.close()

    # convenient namings to match existing packages
    predict = _infer_labels
    fit = train
