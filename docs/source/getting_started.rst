***************
Getting Started
***************

``TheCannon`` can be installed using ``pip``:

``pip install TheCannon``

Here is an overview of the basic workflow. All of this is described
in more detail in other sections...

First, you have to prepare the data for use in whatever way is 
required from the particularly spectroscopic survey and the training labels

The basic input to ``TheCannon`` is the following: and write the list 
Read them in whatever way necessary. 

    >>> from TheCannon import dataset
    >>> dataset = dataset.Dataset(
    >>> ...wl, tr_ID, tr_flux, tr_ivar, tr_label, test_ID, test_flux, test_ivar)

Other optional setup things...:

    >>> dataset.ranges = [[371,3192], [3697,5997], [6461,8255]]
    >>> dataset.set_label_names(['T_{eff}', '\log g', '[Fe/H]'])

Some optional diagnostic plots

    >>> dataset.diagnostics_SNR()
    >>> dataset.diagnostics_ref_labels()

``TheCannon`` includes its own continuum normalization.

    >>> pseudo_tr_flux, pseudo_tr_ivar = dataset.continuum_normalize_training_q(
    >>> ...q=0.90, delta_lambda=50)
    >>> contmask = dataset.make_contmask(
    >>> ...pseudo_tr_flux, pseudo_tr_ivar, frac=0.07)

Have a look, and if you're OK with it then

    >>> dataset.set_continuum(contmask)

Then you can fit for the continuum using

    >>> cont = dataset.fit_continuum(3, "sinusoid")

And if you're cool with how the continuum looks, then

    >>> norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
    >>> dataset.continuum_normalize_f(cont)

And if you're cool with the new continuum, then

    >>> dataset.tr_flux = norm_tr_flux
    >>> dataset.tr_ivar = norm_tr_ivar
    >>> dataset.test_flux = norm_test_flux
    >>> dataset.test_ivar = norm_test_ivar

Now, the data munging is over. 

Training step:

>>> model = model.CannonModel(dataset, 2) # 2 = quadratic model
>>> model.fit() # model.train would work equivalently.
>>> or model.train()

Optional: 
>>> model.diagnostics()

Test step:
>>> label_errs = model.infer_labels(dataset)

Optional:
>>> dataset.diagnostics_test_step_flagstars()
>>> dataset.diagnostics_survey_labels()

If the test step = the training step, then you can do this: 
>>> dataset.diagnostics_1to1()
