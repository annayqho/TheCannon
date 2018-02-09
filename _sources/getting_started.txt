***************
Getting Started
***************

If you want to use *The Cannon*,
there are two options.
(1) You can ``git clone`` the code directly from `github`_.
(2) You can use the public release version.
Options (2) is recommended, since it is more stable 
and more likely to be compatible with the tutorials on this website.
Instructions for installing and using the public release version
are below.


Installation
------------

``TheCannon`` can be installed using ``pip``:

    $ pip install TheCannon


Requirements
------------

It is compatible with Python 2 and 3, 
numpy versions 1.7-1.9 and scipy versions 0.13-0.15. 

Before using the code, make sure you understand
the input requirements.

:ref:`input_requirements`


Tutorials
---------

Once you have installed the package, you can get
an overview of the basic workflow using two simple illustrations:
one with APOGEE DR10 data in which the test set is identical to the training set,
and one with LAMOST data in which we perform a leave-1/8-out cross-validation
(as in `Ho et al. 2016`_).


APOGEE Tutorial
```````````````
:ref:`apogee_tutorial`

LAMOST Tutorial
```````````````
:ref:`lamost_tutorial`

.. _github: https://github.com/annayqho/TheCannon
.. _Ho et al. 2016: https://arxiv.org/abs/1602.00303
