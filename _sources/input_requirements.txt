Requirements for Input
======================

Required input to ``TheCannon`` are as follows:

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
In particular, the spectra must:

  * Come from the same dataset (measured in a consistent way)
  * Be continuum normalized in a consistent way that is independent of
    signal-to-noise
  * Radial velocity shifted
  * Sampled onto a common wavelength grid with a common line-spread function

.. _Ness et al. 2015: http://arxiv.org/abs/1501.07604 
