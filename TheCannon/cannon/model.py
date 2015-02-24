""" Make a single model class to rule them all """
from .dataset import Dataset
from .train_model import train_model as _train_model
from .train_model import model_diagnostics as _model_diagnostics
from .infer_labels import infer_labels
from .spectral_model import diagnostics as _diagnostics
import numpy as np
from copy import deepcopy


class CannonModel(object):
    def __init__(self, dataset, order):
        if not isinstance(dataset, Dataset):
            txt = 'Expecting a Dataset instance, got {0}'
            raise TypeError(txt.format(type(training_set)))
        self.dataset = dataset
        self._model = None
        self.order = order # right now this does nothing,
                           # because we only have one fitting func...

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
            bad = dataset.test_ivars[i,:] == 0
            cannon_ivars[i,:][~bad] = 1. / scatters[~bad] ** 2
        cannon_set = deepcopy(dataset)
        cannon_set.test_fluxes = cannon_fluxes
        cannon_set.test_ivars = cannon_ivars

        return cannon_set

    def spectral_diagnostics(self, dataset):
        _diagnostics(self.draw_spectra(dataset), dataset, self.model)

    # convenient namings to match existing packages
    predict = infer_labels
    fit = train
