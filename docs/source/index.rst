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

Notable features include:

* Continuum-normalize spectra using a running quantile
* Continuum-normalize spectra using cuts on median and variance flux
* Fit for the spectral model given training spectra and training labels
* Specify the order of the polynomial spectral model
* Infer labels for test spectra
* Various optional diagnostic plots

This documentation includes a very simple example for implementation 
using APOGEE spectra in which the test set is identical to the training set

The code is open source and `available on github`_. 

.. _Ness et al. 2015: http://arxiv.org/abs/1501.07604 
.. _available on github: https://github.com/annayqho/TheCannon

Table of Contents
=================

.. toctree::
   :maxdepth: 2

   input_requirements
   getting_started 
   using_tc
   api 
