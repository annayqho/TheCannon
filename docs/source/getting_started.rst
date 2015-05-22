***************
Getting Started
***************

``TheCannon`` can be installed using ``pip``:

    $ pip install TheCannon

Here is an overview of the basic workflow using a simple illustration 
with APOGEE DR10 data in which the test set is identical to the training set.
To run this example, download the file ``example_DR10.tar.gz`` by clicking 
:download:`here <example_DR10.tar.gz>`
and unzip it using the command

    $ tar -zxvf example_DR10.tar.gz

At some point all of this will be described in more detail in other sections...

First, the data must be prepared for use according to the specifications
laid out in the "Requirements for Input" section. ``TheCannon`` does have
some built-in options for SNR-independent continuum normalization, so it's 
okay if input data is not continuum normalized at this stage. 

Then a ``Dataset`` object can be initialized:

    >>> from TheCannon import dataset
    >>> dataset = dataset.Dataset(
    >>> ...wl, tr_ID, tr_flux, tr_ivar, tr_label, test_ID, test_flux, test_ivar)

APOGEE data has gaps, and in this example we want to continuum normalize each
section separately. So that ``TheCannon`` knows that operations should be
performed for each section separately, we specify the ranges of each segment:

    >>> dataset.ranges = [[371,3192], [3697,5997], [6461,8255]]

``TheCannon`` comes with a series of optional built-in diagnostic plots. 
Some of these plots require reading in label names. The user can specify
what the label names are, in LaTeX format. 

    >>> dataset.set_label_names(['T_{eff}', '\log g', '[Fe/H]'])

At this stage, two diagnotic plots can be produced, one with the distribution
of SNR in the training and test set (the training set should ideally have
a higher SNR than the test set) and one using ``triangle.py`` to plot
every label's set of training values against every other.  

    >>> dataset.diagnostics_SNR()
    >>> dataset.diagnostics_ref_labels()

If the data has not been continuum normalized, the user can use the continuum
normalization functions built into ``TheCannon``. First, the training set
is pseudo-continuum normalized using a running quantile. In this case, the
window size for calculating the median is set to 50 Angstroms and the quantile
level is set to 90\%. 

    >>> pseudo_tr_flux, pseudo_tr_ivar = dataset.continuum_normalize_training_q(
    >>> ...q=0.90, delta_lambda=50)

Once the pseudo continuum has been calculated, continuum pixels are identified
using a median and variance flux cut. In this case, we specify that we want
7% of the pixels in the spectrum to be identified as continuum.

    >>> contmask = dataset.make_contmask(
    >>> ...pseudo_tr_flux, pseudo_tr_ivar, frac=0.07)

Once a satisfactory set of continuum pixels has been identified, the dataset's
continuum can be established:

    >>> dataset.set_continuum(contmask)

Once the dataset has a continuum mask, the continuum is fit for using either
a sinusoid or chebyshev function. In this case, we use a sinusoid:

    >>> cont = dataset.fit_continuum(3, "sinusoid")

Once a satisfactory continuum has been fit, the normalized training and test
spectra can be calculated:

    >>> norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
    >>> dataset.continuum_normalize(cont)

If these normalized spectra look acceptable, then they can be set:

    >>> dataset.tr_flux = norm_tr_flux
    >>> dataset.tr_ivar = norm_tr_ivar
    >>> dataset.test_flux = norm_test_flux
    >>> dataset.test_ivar = norm_test_ivar

Now, the data munging is over and we're ready to run ``TheCannon``!

For the training step (fitting for the spectral model) all the user needs to 
specify is the polynomial order of the spectral model. In this case, we use
a quadratic model: order = 2

>>> model = model.CannonModel(dataset, 2) 
>>> model.fit() 

equivalently,

>>> model.train()

At this stage, more optional diagnostic plots can be produced to examine
the spectral model:

>>> model.diagnostics()

If the model fitting worked, then we can proceed to the test step. This 
command automatically updates the dataset with the fitted-for test labels,
and returned the corresponding covariance matrix.

>>> label_errs = model.infer_labels(dataset)

And a final set of diagnostic plots:

>>> dataset.diagnostics_test_step_flagstars()
>>> dataset.diagnostics_survey_labels()

If the test step = the training step, then you can do this: 
>>> dataset.diagnostics_1to1()
