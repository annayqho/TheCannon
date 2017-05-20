.. _lamost_tutorial:

*********************************
Tutorial with LAMOST DR2 Spectra
*********************************

In this tutorial, we're going to use The Cannon 
to transfer a system of labels from APOGEE to LAMOST. 
More specifically, we're going to model LAMOST spectra as a function of 
four labels from APOGEE DR12 (the 12th data release): 
effective temperature T_eff, surface gravity logg, metallicity [Fe/H], 
and alpha enhancement [alpha/Fe].
To fit this model, we will use a *reference set*,
a set of stars observed in common between APOGEE
and LAMOST.
We will then be able to use this model 
to determine these four APOGEE-scale labels
from any new LAMOST spectrum, 
provided that the parameters of that star falls 
within the range of the reference set.
For more details on this procedure,
see the accompanying paper `Ho et al. 2017`_.

As described in that paper,
there are 11,057 objects measured in common between APOGEE and LAMOST.
In our work, we used around 10,000 of those objects,
after making cuts due to poor-quality data (flags, etc).
For the purpose of this tutorial, to speed things up,
I provide only the highest-SNR subset of the LAMOST spectra,
those with SNR > 100. This SNR cut leaves 1936 stars.

To run this example, download the folder ``lamost_spectra`` by clicking 
:download:`here <lamost_spectra.zip>`
and unzip it using the command

    $ unzip lamost_spectra.zip

Navigate into the spectra directory and count the number of files using

    $ ls | wc -l

There should be 11057 files, corresponding to 11057 stellar spectra.

Next, download the reference labels by clicking :download:`here <lamost_labels.fits>`.
Let's use the ``astropy`` module to examine the contents of this file.

>>> from astropy.table import Table
>>> data = Table.read("lamost_labels.fits")
>>> print(data.colnames)

You'll see that the first column is called ``LAMOST_ID``,
and the rest are ``RA``, ``Dec``, ``APOGEE_ID``,
``TEFF``, ``LOGG``, ``PARAM_M_H`` and ``PARAM_ALPHA_M``.
All of these stars were observed by both LAMOST and APOGEE,
which is why they have a LAMOST ID as well as an APOGEE ID.
The Teff, logg, [M/H], and [alpha/M] values are taken from
APOGEE. In this tutorial, we will use the APOGEE values because
it is the higher quality (higher SNR, higher resolution) survey.
For our model, we will be using LAMOST spectra and APOGEE labels,
and modeling the LAMOST spectra as a function of APOGEE labels.

Next, let's plot one spectrum. We will use the ``load_spectra``
module in ``TheCannon`` code.

>>> from TheCannon.lamost import load_spectra

The filenames of the spectra correspond to the IDs in the LAMOST_ID column
described above. Let's pick the first one:

>>> filename = data['LAMOST_ID'][0]

To get rid of trailing white spaces:

>>> filename = data['LAMOST_ID'][0].strip()

And now load the spectrum by feeding the filename into the function:

>>> specdir = "spectra"
>>> wl, flux, ivar = load_spectra("%s/" %specdir + filename)

Now, plot the thing

>>> plt.step(wl, flux, where='mid', linewidth=0.5, color='k')
>>> plt.xlabel("Wavelength (Angstroms)")
>>> plt.ylabel("Flux")
>>> plt.show()

Now, get all of the files

>>> filenames = np.array([val.strip() for val in data['LAMOST_ID']])
>>> filenames_full = np.array([specdir+"/"+val.strip() for val in filenames])
>>> wl, flux, ivar = load_spectra(filenames_full)

We'll use the first 1000 stars as the training set.

>>> tr_flux = flux[0:1000]
>>> tr_ivar = ivar[0:1000]
>>> tr_ID = filenames[0:1000]

>>> inds = np.array([np.where(filenames==val)[0][0] for val in tr_ID])
>>> tr_teff = data['TEFF'][inds]
>>> tr_logg = data['LOGG'][inds]
>>> tr_mh = data['PARAM_M_H'][inds]
>>> tr_alpham = data['PARAM_ALPHA_M'][inds]

Let's look at the teff-logg diagram of the training labels,
color-coded by metallicity.

>>> plt.scatter(tr_teff, tr_logg, c=tr_mh, lw=0, s=7, cmap="viridis")
>>> plt.gca().invert_xaxis()
>>> plt.xlabel("Teff")
>>> plt.ylabel("logg")
>>> plt.colorbar(label="M/H")
>>> plt.savefig("teff_logg_training.png")
>>> plt.close()

Note that there are very few stars at low metallicity,
so it will probably be challenging to do as good of a job
or get as precise results here.

Before the data can be run through ``TheCannon``, it must be prepared
according to the specifications laid out in the "Requirements for Input"
section. One of the requirements is for data to be continuum normalized
in a SNR-independent way. ``TheCannon`` does have built-in 
options for continuum normalizing spectra, and we illustrate that here.

Here are the steps for reading in the data. In practice, the user would
write his own code; for this example, we provide the module ``apogee.py``. 
The procedure for reading in spectra and training labels of course depends on
the survey, the file type, etc, and it is up to the user to package this
all appropriately before feeding it into ``TheCannon``.

>>> filenames = np.array([val.strip() for val in data['LAMOST_ID']])
>>> filenames_full = np.array([specdir+"/"+val.strip() for val in filenames])
>>> wl, flux, ivar = load_spectra(filenames_full)

There should be XXXX spectra with 3626 pixels each. 
We'll choose the first 1000 stars for the training set, 
and use the rest for the test set.

>>> tr_flux = flux[0:1000]
>>> tr_ivar = ivar[0:1000]
>>> tr_ID = filenames[0:1000]

Let's get the reference labels

>>> inds = np.array([np.where(filenames==val)[0][0] for val in tr_ID])
>>> tr_teff = data['TEFF'][inds]
>>> tr_logg = data['LOGG'][inds]
>>> tr_mh = data['PARAM_M_H'][inds]
>>> tr_alpham = data['PARAM_ALPHA_M'][inds]

Take a look at the teff-logg diagram, color-coded by metallicity
>>> plt.scatter(tr_teff, tr_logg, c=tr_mh, lw=0, s=7, cmap="viridis")
>>> plt.gca().invert_xaxis()
>>> plt.xlabel("Teff")
>>> plt.ylabel("logg")
>>> plt.colorbar(label="M/H")
>>> plt.savefig("teff_logg_training.png")
>>> plt.close()

Note that there are very few stars at low metallicity,
so it will probably be challenging to do as good of a job
or get as precise results here.

>>> print(wl.shape)
>>> print(tr_ID.shape)
>>> print(tr_flux.shape)
>>> print(tr_ivar.shape)

[num_training_objects, num_pixels]
(1339, 3626)
Fine. Not normalized yet, but we will do that later.

Now we need a block of training labels
[num_training_objects, num_labels]
Right now we have them separate, combine into an array of this shape:

>>> tr_label = np.vstack((tr_teff, tr_logg, tr_mh, tr_alpham))

Note that that gives us (4,1339) which is (num_labels, num_tr_obj),
So we need to take the transpose

Now we need to define our "test set": a bunch of other
spectra whose labels we want to determine and don't know yet.
Let's use some of the other spectra in the dataset
Say, the ones with 80 < SNR < 100
>>> test_ID = filenames[1000:]
>>> test_flux = flux[1000:]
>>> test_ivar = ivar[1000:]

Check the sizes
>>> print(test_ID.shape)
>>> print(test_flux.shape)
>>> print(test_ivar.shape)


Now, all the input data has been packaged properly, and we can begin running
``TheCannon.``

The first step is to initialize a ``Dataset`` object:

>>> ds = dataset.Dataset(
>>> ...wl, tr_ID, tr_flux, tr_ivar, tr_label, test_ID, test_flux, test_ivar)

``TheCannon`` has a number of optional diagnostic plots built-in, to help the
user visualize the results. Some of these plots require knowing the names
of the labels. If the user wants to produce these diagnostic plots, he or
she must specify the label names in LaTeX format: 

>>> ds.set_label_names(['T_{eff}', '\log g', '[M/H]', '[alpha/M]'])

At this stage, two diagnotic plots can already be produced, 
one with the distribution
of SNR in the training and test set (in practice, the training set 
should consist of higher SNR spectra than the test set) 
and the other using ``triangle.py`` to plot
every label's set of training values against every other.  

    >>> fig = ds.diagnostics_SNR()

.. image:: images_lamost/SNRdist.png

We can also plot the reference labels against each other:

    >>> fig = ds.diagnostics_ref_labels()

That figure should look like this:

.. image:: images_lamost/ref_labels.png

Again, ``TheCannon`` requires incoming spectra to be normalized
in a way that is independent of signal to noise. If the data does not satisfy
this criteria already, the user can use the 
functions built into ``TheCannon``. 

>>> ds.continuum_normalize_gaussian_smoothing(L=50)

Let's take a look at a normalized spectrum.

>>> plt.step(ds.wl, ds.tr_flux[0], where='mid', linewidth=0.5, color='k')
>>> plt.xlabel("Wavelength (Angstroms)")
>>> plt.ylabel("Flux")

.. image:: images_lamost/norm_spec.png

Now, the data munging is over and we're ready to run ``TheCannon``!

For the training step (fitting for the spectral model) all the user needs to 
specify is the desired polynomial order of the spectral model. 
In this case, we use a quadratic model: order = 2

>>> m = model.CannonModel(2, useErrors=False) 
>>> m.fit(ds) 

At this stage, more optional diagnostic plots can be produced to examine
the spectral model:

>>> m.diagnostics_leading_coeffs(ds)

The second is a plot of the leading coefficients and scatter of the model
as a function of wavelength

.. image:: images_lamost/leading_coeffs.png

If the model fitting worked, then we can proceed to the test step. This 
command automatically updates the dataset with the fitted-for test labels,
and returns the corresponding covariance matrix.

>>> starting_guess = np.mean(ds.tr_label,axis=0)-m.pivots
>>> errs, chisq = m.infer_labels(ds, starting_guess)

You can access the new labels as follows:

>>> test_labels = ds.test_label_vals

A set of diagnostic output:

>>> ds.diagnostics_survey_labels()

The second generates a triangle plot of the survey (Cannon) labels,
shown below.

.. image:: images_lamost/survey_labels.png

Now we can compare the "real" values to the Cannon values, for the test objects.

>>> inds = np.array([np.where(filenames==val)[0][0] for val in ds.test_ID])
>>> test_teff = data['TEFF'][inds]
>>> test_logg = data['LOGG'][inds]
>>> test_mh = data['PARAM_M_H'][inds]
>>> test_alpham = data['PARAM_ALPHA_M'][inds]
>>> test_label = np.vstack((test_teff, test_logg, test_mh, test_alpham)).T
>>> ds.tr_label = test_label

>>> ds.diagnostics_1to1()

.. image:: images_lamost/1to1_label_0.png

.. image:: images_lamost/1to1_label_1.png

.. image:: images_lamost/1to1_label_2.png

.. _Ho et al. 2017: http://iopscience.iop.org/article/10.3847/1538-4357/836/1/5/pdf

