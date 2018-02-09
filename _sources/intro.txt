============
Introduction
============

*The Cannon* is a data-driven method for determining stellar parameters 
and chemical abundances from stellar spectra in the context of large 
spectroscopic surveys. 
*The Cannon* uses no physical stellar models, is very fast, 
and achieves comparable accuracy to existing survey pipelines 
using significantly lower SNR spectra; 
it requires only a set of objects observed in common between the surveys. 
Because of these strengths, it has been used to
bring different stellar surveys onto a consistent physical scale,
and even transfer information from one survey to another.

For a detailed overview of *The Cannon*,
see `Ness et al. 2015`_. 

Applications
------------
(Current as of 8 Feb. 2018)

- Re-analysis of APOGEE DR10: `Ness et al. 2015`_
- Measuring mass (and inferring age) from spectra: `Ness et al. 2016`_
- Chemical tagging: `Hogg et al. 2016`_
- Tying LAMOST to the APOGEE label scale: `Ho et al. 2017a`_
- Detailed element abundances (with regularization): `Casey et al. 2016`_
- Largest catalog to-date of stellar masses, ages, and individual abundances 
  (alpha enhancement, carbon, and nitrogen): `Ho et al. 2017b`_


Documentation
-------------

A version of the code for *The Cannon*
is open source and `available on github`_. 
It is compatible with Python 2 and 3, 
numpy versions 1.7-1.9 and scipy versions 0.13-0.15. 

Features include:

* Diagnostic output to help the user monitor and evaluate the process 
* SNR-independent normalization of spectra
* Training step: fit for the spectral model given training spectra and labels,
  with the polynomial order for the spectral model decided by the user
* Test step: infer labels for the test spectra

This documentation includes two tutorials for implementation,
one using APOGEE spectra in which the test set is identical to the training set,
the other using LAMOST spectra in which we perform a leave-1/8-out cross-validation.

Before you use the code, you need to make sure you have all the requirements
for input. Click "Next" below to see what those are.

.. _Ness et al. 2015: http://arxiv.org/abs/1501.07604 
.. _Ness et al. 2016: https://arxiv.org/abs/1511.08204
.. _Hogg et al. 2016: https://arxiv.org/abs/1601.05413
.. _Ho et al. 2017a: https://arxiv.org/abs/1602.00303
.. _Casey et al. 2017: https://arxiv.org/abs/1603.03040
.. _Ho et al. 2017b: https://arxiv.org/abs/1609.03195
.. _available on github: https://github.com/annayqho/TheCannon
