""" Dummy package that allows cleaner imports """
try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits
