""" Make a single model class to rule them all """
from .dataset import Dataset
from .train_model import train_model as _train_model
from .infer_labels import infer_labels
from .spectral_model import diagnostics as _diagnostics
from .helpers.triangle import corner
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

LARGE = 200.
SMALL = 1. / LARGE

class CannonModel(object):
    def __init__(self, dataset, order):
        if not isinstance(dataset, Dataset):
            txt = 'Expecting a Dataset instance, got {0}'
            raise TypeError(txt.format(type(dataset)))
        self.dataset = dataset
        self._model = None
        self.order = order

    @property
    def model(self):
        """ return the model definition or raise an error if not trained """
        if self._model is None:
            raise RuntimeError('Model not trained')
        else:
            return self._model

    def train(self, *args, **kwargs):
        """ Train the model """
        self._model = _train_model(self.dataset)

    def diagnostics(self):
        """Run a set of diagnostics on the model.

        Plot the 0th order coefficients as the baseline spectrum.
        Overplot the continuum pixels.

        Plot each label's leading coefficient as a function of wavelength.
        Color-code by label.

        Histogram of the chi squareds of the fits.
        Dotted line corresponding to DOF = npixels - nlabels

        Parameters
        ----------
        contpix: str
            continuum pixel definition file
        """
        _model_diagnostics(self.dataset, self.model)

    def infer_labels(self, dataset):
        """
        Uses the model to solve for labels of the test set.

        Parameters
        ----------
        test_set: Dataset
            dataset that needs label inference

        Returns
        -------
        test_set: Dataset
            same dataset as the input value with updated label_vals attribute

        covs_all:
            covariance matrix of the fit
        """
        return infer_labels(self.model, dataset)

    def draw_spectra(self, dataset):
        """
        Parameters
        ----------
        test_set: Dataset
            dataset that needs label inference

        Returns
        -------
        cannon_set: Dataset
            same dataset as the input value with updated fluxes and variances

        """
        coeffs_all, covs, scatters, red_chisqs, pivots, label_vector = self.model
        nstars = len(dataset.test_SNRs)
        cannon_fluxes = np.zeros(dataset.test_fluxes.shape)
        cannon_ivars = np.zeros(dataset.test_ivars.shape)
        for i in range(nstars):
            x = label_vector[:,i,:]
            spec_fit = np.einsum('ij, ij->i', x, coeffs_all)
            cannon_fluxes[i,:] = spec_fit
            bad = dataset.test_ivars[i,:] == SMALL
            cannon_ivars[i,:][~bad] = 1. / scatters[~bad] ** 2
        cannon_set = deepcopy(dataset)
        cannon_set.test_fluxes = cannon_fluxes
        cannon_set.test_ivars = cannon_ivars

        return cannon_set

    def split_array(self, array, num):
        avg = len(array) / float(num)
        out = []
        last = 0.0
        while last < len(array):
            out.append(array[int(last):int(last+avg)])
            last += avg
        return out

    def diagnostics(
            self, baseline_spec_plot_name = "baseline_spec_with_cont_pix",
            leading_coeffs_plot_name = "leading_coeffs.png",
            chisq_dist_plot_name = "modelfit_chisqs.png"):
        """Run a set of diagnostics on the model.

        Plot the 0th order coefficients as the baseline spectrum.
        Overplot the continuum pixels.

        Plot each label's leading coefficient as a function of wavelength.
        Color-code by label.

        Histogram of the chi squareds of the fits.
        Dotted line corresponding to DOF = npixels - nlabels

        Parameters
        ----------
        """
        dataset = self.dataset
        model = self.model

        contmask = dataset.contmask
        lams = dataset.wl
        label_names = dataset.get_plotting_labels()
        coeffs_all, covs, scatters, chisqs, pivots, label_vector = model
        npixels = len(lams)
        nlabels = len(pivots)

        if contmask != None:
            # Baseline spectrum with continuum
            baseline_spec = coeffs_all[:,0]
            bad = np.round(baseline_spec,5) == 0
            baseline_spec = np.ma.array(baseline_spec, mask=bad)
            lams = np.ma.array(lams, mask=bad)

            # Continuum pixels
            contpix_lambda = lams[contmask]
            y = baseline_spec[contmask]

            # Split into ten segments
            nseg = 10
            lams_seg = self.split_array(lams.compressed(), nseg)
            xmins = [] 
            xmaxs = [] 
            for seg in lams_seg:
                xmins.append(seg[0])
                xmaxs.append(seg[-1])

            for i in range(nseg):
                fig, axarr = plt.subplots(2, sharex=True)
                plt.xlabel(r"Wavelength $\lambda (\AA)$")
                plt.xlim(xmins[i], xmaxs[i])
                ax = axarr[0]
                ax.step(lams, baseline_spec, where='mid', c='k', linewidth=0.3,
                        label=r'$\theta_0$' + "= the leading fit coefficient")
                ax.scatter(contpix_lambda, y, s=1, color='r',label="continuum pixels")
                ax.legend(loc='lower right', prop={'family':'serif', 'size':'small'})
                ax.set_title("Baseline Spectrum with Continuum Pixels")
                ax.set_ylabel(r'$\theta_0$')
                ax = axarr[1]
                ax.step(lams, baseline_spec, where='mid', c='k', linewidth=0.3,
                     label=r'$\theta_0$' + "= the leading fit coefficient")
                ax.scatter(contpix_lambda, y, s=1, color='r',label="continuum pixels")
                ax.set_title("Baseline Spectrum with Continuum Pixels, Zoomed")
                ax.legend(loc='upper right', prop={'family':'serif', 'size':'small'})
                ax.set_ylabel(r'$\theta_0$')
                ax.set_ylim(0.95, 1.05)
                print("Diagnostic plot: fitted 0th order spec, cont pix overlaid.")
                print("Saved as %s_%s.png" % (baseline_spec_plot_name, i))
                plt.savefig(baseline_spec_plot_name + "_%s" %i)
                plt.close()

        # Leading coefficients for each label & scatter
        # Scale coefficients so that they can be overlaid on the same plot
        c = ['r', 'g', 'b', 'k']
        stds = np.array([np.std(coeffs_all[:, i + 1]) for i in range(nlabels)])
        pivot_std = max(stds)
        ratios = np.round(pivot_std / stds, -1)  # round to the nearest 10
        ratios[ratios == 0] = 1
        fig, axarr = plt.subplots(2, sharex=True)
        plt.xlabel(r"Wavelength $\lambda (\AA)$")
        plt.xlim(np.ma.min(lams), np.ma.max(lams))
        ax = axarr[0]
        ax.set_ylabel("Leading coefficient " + r"$\theta_i$")
        ax.set_title("First-Order Fit Coefficients for Labels")

        first_order = np.zeros((npixels, nlabels))
        lbl = r'$\theta_{0:d}$=coeff for ${1:s}$ * ${2:d}$'
        for i in range(nlabels):
            coeffs = coeffs_all[:,i+1] * ratios[i]
            first_order[:,i] = coeffs
            ax.step(lams, coeffs, where='mid', linewidth=0.5, 
                    label=lbl.format(i+1, label_names[i], int(ratios[i])), c=c[i])
        box = ax.get_position()
        ax.set_position(
                [box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])
        ax.legend(
                bbox_to_anchor=(0., -.2, 1., .102), loc=3, ncol=3, 
                mode="expand", prop={'family':'serif', 'size':'small'})
        ax = axarr[1]
        ax.set_ylabel("scatter")
        ax.set_title("Scatter of Fit")
        ax.step(lams, scatters, where='mid', c='k', linewidth=0.7)
        fig.tight_layout(pad=2.0, h_pad=4.0)
        print("Diagnostic plot: leading coeffs and scatters across wavelength.")
        print("Saved as %s" %leading_coeffs_plot_name)
        fig.savefig(leading_coeffs_plot_name)
        plt.close(fig)

        # triangle plot of the higher-order coefficients
        labels = [r"$%s$" % l for l in label_names]
        fig = corner(first_order, labels=labels, show_titles=True,
                     title_args = {"fontsize":12})
        filename = "leading_coeffs_triangle.png"
        fig.savefig(filename)
        plt.close(fig)

        # Histogram of the chi squareds of ind. stars
        plt.hist(np.sum(chisqs, axis=0), color='lightblue', alpha=0.7)
        dof = len(lams) - coeffs_all.shape[1]   # for one star
        plt.axvline(x=dof, c='k', linewidth=2, label="DOF")
        plt.legend()
        plt.title("Distribution of " + r"$\chi^2$" + " of the Model Fit")
        plt.ylabel("Count")
        plt.xlabel(r"$\chi^2$" + " of Individual Star")
        print("Diagnostic plot: histogram of the red chi squareds of the fit")
        print("Saved as %s" %chisq_dist_plot_name)
        plt.savefig(chisq_dist_plot_name)
        plt.close()

    # convenient namings to match existing packages
    predict = infer_labels
    fit = train
