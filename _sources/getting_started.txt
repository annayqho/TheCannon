***************
Getting Started
***************

If you want to use *The Cannon*,
there are two options:
(1) You can ``git clone`` the code directly from `github`_,
or (2) You can use the public release version.
Option (2) is recommended, since this version is more stable.
Instructions for installing and using the public release version
are below.

.. The code should be compatible with Python 2 and 3, 
.. numpy versions 1.7-1.9 and scipy versions 0.13-0.15
.. (and if it's not, please let me know!)


Installation
------------

``TheCannon`` can be installed using ``pip``:

    $ pip install TheCannon


Basic Workflow
--------------

Typically, you have some large dataset of stellar spectra.
You want to measure some set of labels from those spectra.
For a subset -- called the *reference set* of objects, 
or the *reference objects* --
you already know those labels with high fidelity
(for any reason, perhaps because they were measured by a different survey
or at higher SNR).
The procedure is roughly:

1. Get your dataset into the right format,
   which may involve normalizing the spectra (see below)
2. Use the reference set to fit a spectral model (*training step*)
3. To make sure that the model is sensible, 
   perform a leave-n-out cross-validation
4. Once you feel comfortable with the model, 
   use it to infer labels from all the other spectra in the dataset 
   (*test step*)

The most difficult part of using *The Cannon* (by far)
is Step 1: getting your data into the right format.
So, before you jump into using the code,
make sure you understand
the input requirements:

Requirements
````````````
:ref:`input_requirements`


Tutorials
---------

Once you have installed the package and understood the input requirements,
you can work through two applications of the workflow outlined above. 
One tutorial uses APOGEE DR10 data to infer labels for the same objects
used to train the model (to keep things simple),
without doing any cross-validation (Step 3). 
The second tutorial uses LAMOST data to perform a full 
leave-1/8-out cross-validation.


APOGEE Tutorial
```````````````
:ref:`apogee_tutorial`

LAMOST Tutorial
```````````````
:ref:`lamost_tutorial`

.. _github: https://github.com/annayqho/TheCannon
.. _Ho et al. 2016: https://arxiv.org/abs/1602.00303
