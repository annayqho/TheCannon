.. _input_requirements:

Requirements for Input
======================

As mentioned in the previous section,
the most difficult part of using *The Cannon*
is getting your data into a suitable format.
For a detailed description of the requirements,
you are encouraged to read  `Ness et al. 2015`_. 
Here are some basics:

  * All spectra must come from the same dataset 
    (same telescope, same reduction pipeline)
  * Reference labels (the high-fidelity labels you use for the training set) 
    must come from a consistent source. To be more concrete:
    imagine that you want to measure Teff and logg,
    and you think that the most reliable Teff comes from APOGEE,
    while the most reliable logg comes from Kepler.
    Then *all* of your reference Teff values must come from the same APOGEE
    dataset, and *all* of your reference logg values must come from the same
    Kepler dataset.
  * Spectra must be normalized in a consistent way that is independent of
    signal-to-noise (more precisely: 
    the normalization procedure should be a linear operation on the data, 
    so that it is unbiased as (symmetric) noise grows)
  * Spectra must be radial velocity shifted (so, spectral features
    should line up at the same wavelength values when two objects are compared)
  * Spectra must be sampled onto a common wavelength grid 
    (all spectra must have the same start and stop wavelengths,
    and the same wavelength values)
  * Spectra must have a common line-spread function
    (cannot have different resolution)
  * Each flux value in the spectrum must be accompanied by an error bar
    (or an inverse variance)
  * A bad datapoint must be assigned an inverse variance of zero 
    or very close to zero. 
    ``Bad Data`` includes regions where there are many skylines, 
    regions where sky subtraction is known to be an issue, 
    and telluric regions.

The spirit behind all of this is that the model you build with *The Cannon*, 
and which you ultimately use to fit all of the spectra,
only knows about the labels and the symmetric (Gaussian) noise (S/N).
It does not know about any other effects,
such as those resulting from artifacts of using two different telescopes.
The model does not know about all of the factors that go into the overall 
shape of a spectrum, nor does it know about line broadening due to rotation,
nor the shifting of lines due to a nonzero radial velocity.
It also doesn't know about bad data or skylines -- it will simply weight
every pixel in the spectrum using its associated formal uncertainty.

When you are sure that your data obey all of the above criteria,
you need to create four different Python ``numpy`` arrays:

* **Wavelength Grid**

    * an array of wavelength values with shape [num_pixels] corresponding to all      of the spectra (both training spectra and test spectra)

* **Training Set**

  * Training Spectra

    * a block of continuum-normalized pixel intensity (flux) values with shape
      [num_training_objects x num_pixels]
    * a block of inverse variance values corresponding to the block of 
      pixel intensity values described above

  * Training Labels

    * a block of training labels with shape [num_training_objects x num_labels]


* **Test Set**

  * Test Spectra

    * a block of continuum-normalized pixel intensity (flux) values with shape
      [num_test_objects x num_pixels]
    * a block of inverse variance values corresponding to the block of
      pixel intensity values described above 


.. _Ness et al. 2015: http://arxiv.org/abs/1501.07604 
