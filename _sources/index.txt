****************************
*The Cannon* (``TheCannon``)
****************************
Stellar Labels from Large Spectroscopic Datasets
************************************************

Introduction
============
``TheCannon`` provides functionality for *The Cannon*,
a data-driven approach to determining stellar labels (parameters
and abundances) from stellar spectra in the context of large
spectroscopic surveys. 

For a detailed overview of *The Cannon* and a description of its
successful application to determining labels for APOGEE DR10 spectra,
see `Ness et al. 2015`_. 

Features include:

* Diagnostic output to help the user monitor and evaluate the process 
* SNR-independent continuum normalization 
* Training step: fit for the spectral model given training spectra and labels,
  with the polynomial order for the spectral model decided by the user
* Test step: infer labels for the test spectra

This documentation includes a very simple example for implementation 
using APOGEE spectra in which the test set is identical to the training set. 

The code is open source and `available on github`_. 

.. _Ness et al. 2015: http://arxiv.org/abs/1501.07604 
.. _available on github: https://github.com/annayqho/TheCannon

Table of Contents
=================

.. toctree::
   :maxdepth: 2

   input_requirements
   getting_started 
   api 
