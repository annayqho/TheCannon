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

This documentation includes two examples for implementation, 
the first a very simple implementation using APOGEE spectra in
which where the test set is identical to the training set, and
the second an implementation of inferring APOGEE-scale labels for 
LAMOST spectra. 

The code is open source and `available on github`_. 

.. _Ness et al. 2015: http://arxiv.org/abs/1501.07604 
.. _available on github: https://github.com/annayqho/TheCannon

The documentation is organized as follows:

.. toctree::
   :maxdepth: 2
   :caption: Preparing Data for Input

   input_requirements

* :ref:`input-requirements`

.. _input-requirements:

.. toctree::
   :maxdepth: 2
   :caption: Preparing Data for Input 

   input_requirements
