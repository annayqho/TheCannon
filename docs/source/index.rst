****************************
*The Cannon* (``TheCannon``)
****************************
Stellar Labels from Large Spectroscopic Datasets
************************************************

.. toctree::
   :maxdepth: 2

Introduction
============

``TheCannon`` provides functionality for *The Cannon*,
a data-driven approach to determining stellar labels (parameters
and abundances) from stellar spectra in the context of large
spectroscopic surveys. 

For a detailed overview of *The Cannon* and a description of its
successful application to determining labels for APOGEE DR10 spectra,
see `Ness et al. 2015`_. 

This documentation includes two examples for implementation, 
the first a very simple implementation using APOGEE spectra in
which where the test set is identical to the training set, and
the second an implementation of inferring APOGEE-scale labels for 
LAMOST spectra. 

The code is open source and `available on github`_. 

.. _Ness et al. 2015: http://arxiv.org/abs/1501.07604 
.. _available on github: https://github.com/annayqho/TheCannon

Requirements for Input
======================

In order to use ``TheCannon``, the input spectra and training labels
must satisfy the criteria laid out in `Ness et al. 2015`_. They consist
of the following:

* **Training Set**

  * Training Spectra

    * a block of continuum-normalized pixel intensity (flux) values with shape
      [num_training_objects x num_pixels]
    * a block of inverse variance values corresponding to the block of 
      pixel intensity values described above


* **Input Training Spectra**: a block of continuum-normalized 
  pixel intensity (flux) values with shape [num_training_objects x num_pixels]

  * Come from the same dataset (measured in a consistent way)
  * Continuum normalized in a consistent way that is independent of
    signal-to-noise
  * Radial velocity shifted
  * Sampled onto a common wavelength grid with a common line-spread function

.. _Ness et al. 2015: http://arxiv.org/abs/1501.07604 

