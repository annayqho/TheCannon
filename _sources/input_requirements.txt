.. _input_requirements:

Requirements for Input
======================

As mentioned in the previous section,
the most difficult part of using *The Cannon*
is getting your data into a suitable format.
For a detailed description of the requirements,
you are encouraged to read  `Ness et al. 2015`_. 
Here are some basics:

  * All spectra must come from the same dataset 
    (same telescope, same reduction pipeline)
  * Reference labels (the high-fidelity labels you use for the training set) 
    must come from a consistent source. To be more concrete:
    imagine that you want to measure Teff and logg,
    and you think that the most reliable Teff comes from APOGEE,
    while the most reliable logg comes from Kepler.
    Then *all* of your reference Teff values must come from the same APOGEE
    dataset, and *all* of your reference logg values must come from the same
    Kepler dataset.
  * Spectra must be normalized in a consistent way that is independent of
    signal-to-noise (more precisely: 
    the normalization procedure should be a linear operation on the data, 
    so that it is unbiased as (symmetric) noise grows)
  * Spectra must be radial velocity shifted (so, spectral features
    should line up at the same wavelength values when two objects are compared)
  * Spectra must be sampled onto a common wavelength grid 
    (all spectra must have the same start and stop wavelengths,
    and the same wavelength values)
  * Spectra must have a common line-spread function
    (cannot have different resolution)
  * Each flux value in the spectrum must be accompanied by an error bar
    (or an inverse variance)
  * A bad datapoint must be assigned an inverse variance of zero 
    or very close to zero. 
    ``Bad Data`` includes regions where there are many skylines, 
    regions where sky subtraction is known to be an issue, 
    and telluric regions.

The spirit behind all of this is that the model you build with *The Cannon*, 
and which you ultimately use to fit all of the spectra,
only knows about the labels and the symmetric (Gaussian) noise (S/N).
It does not know about any other effects,
such as those resulting from artifacts of using two different telescopes.
The model does not know about all of the factors that go into the overall 
shape of a spectrum, nor does it know about line broadening due to rotation,
nor the shifting of lines due to a nonzero radial velocity.
It also doesn't know about bad data or skylines -- it will simply weight
every pixel in the spectrum using its associated formal uncertainty.

When you are sure that your data obey all of the above criteria,
you need to create six different Python ``numpy`` arrays.
They are described below, together with the required shape.

In case you're new to Python, here's a quick note on how to check the shape
of a numpy array called ``my_array``:

>> import numpy as np
>> my_array = np.array([1,2,3,4,5])
>> print(my_array.shape)

That 1-D array has a shape (length) of 5.
It is the same for a 2-D array:

>> my_array = np.array([[1,2,3,4,5],[6,7,8,9,10]])
>> print(my_array.shape)

That 2-D array is a shape (2,5): 2 rows, 5 columns.

Now that you know how to check the shapes of your arrays,
here is what you need:

1. **Wavelength Grid**

    * This is a 1-D array of wavelength values with length [num_pixels].
      This is the wavelength grid for all of the spectra in your dataset.
      It will look something like

      np.array([3600, 3605, 3610, 3615, ...]) 

      where each value is the wavelength in Angstroms 
      (or whatever unit you would like).

2. **Flux Values of Training Spectra**

    * This is a 2-D array of flux (pixel intensity) values with the shape
      [num_training_objects x num_pixels]

3. **Uncertainties in Flux Values of Training Spectra**

    * This is a 2-D array of inverse variance values with an identical
      shape to the training spectra:
      [num_training objects x num_pixels].
    * The objects must be in the same order as in the Training Spectra array.
      That is, if Object A has its spectrum in Row 10
      of the training spectra array, then it should have its uncertainties
      in Row 10 of the uncertainties array.
    * Each inverse variance encodes the uncertainty on the corresponding flux 
      value in the training spectra.

4. **Training Labels**

    * This is a 2-D array of reference (high-fidelity) labels with shape 
      [num_training_objects x num_labels]
    * Again, the row order must correspond to the row order
      in the training spectra arrays.

5. **Flux Values of Test Spectra**

    * Same as (2), except these are the objects you want to measure labels for,
      not the ones you have labels for already. So the shape should be
      [num_test_objects x num_pixels]

6. **Uncertainties in Flux Values of Test Specra**

    * Same as (3), except these correspond to the flux values in (5).


Phew! Once you have all of those spectra in hand,
and they meet the requirements listed above,
you are ready to use **The Cannon**.


.. _Ness et al. 2015: http://arxiv.org/abs/1501.07604 
