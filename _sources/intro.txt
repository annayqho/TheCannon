Introduction
============
``TheCannon`` provides functionality for *The Cannon*,
a data-driven approach to determining stellar labels (parameters
and abundances) from stellar spectra in the context of large
spectroscopic surveys. 

For a detailed overview of *The Cannon* and a description of its
successful application to determining labels for APOGEE DR10 spectra,
see `Ness et al. 2015`_. 
For an application of *The Cannon* to bringing two spectroscopic surveys
(APOGEE and LAMOST) onto the same physical scale,
see `Ho et al. 2016`_.

Features include:

* Diagnostic output to help the user monitor and evaluate the process 
* SNR-independent continuum normalization 
* Training step: fit for the spectral model given training spectra and labels,
  with the polynomial order for the spectral model decided by the user
* Test step: infer labels for the test spectra

This documentation includes two tutorials for implementation,
one using APOGEE spectra in which the test set is identical to the training set,
the other using LAMOST spectra in which we perform a leave-1/8-out cross-validation.

The code is open source and `available on github`_. It is compatible with
Python 2 and 3, numpy versions 1.7-1.9 and scipy versions 0.13-0.15. 

.. _Ness et al. 2015: http://arxiv.org/abs/1501.07604 
.. _Ho et al. 2016: https://arxiv.org/abs/1602.00303
.. _available on github: https://github.com/annayqho/TheCannon
