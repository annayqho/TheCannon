Requirements for Input
======================

Required input to ``TheCannon`` are as follows:

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

These input spectra and labels 
must satisfy the criteria laid out in `Ness et al. 2015`_. 
The spirit of these requirements is that any differences between two
spectra should, to the extent possible, be due to differences in label
values rather than measurement procedure. 
In practice: 

  * Spectra must come from the same dataset 
  * Input labels must come from a consistent source
  * Spectra must be continuum normalized in a consistent way that is independent of
    signal-to-noise (more precisely, the normalization procedure should be a 
    linear operation on the data, so that it is unbiased as (symmetric) noise grows)
  * Spectra must be radial velocity shifted
  * Spectra must be sampled onto a common wavelength grid with a common line-spread function
    (so, all spectra must have the same start and stop wavelengths)
  * Each flux value in the spectrum must be accompanied by error bars / an inverse variance
    array
  * Bad or masked data should be assigned inverse variances of zero or very close to zero. 
    ``Bad Data`` includes: regions where there are many skylines, regions where sky
    subtraction is known to be an issue, telluric regions

.. _Ness et al. 2015: http://arxiv.org/abs/1501.07604 
