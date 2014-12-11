# Initialize all of the data: spectra and training labels

class Star(object):
    'Common base class for all spectra'
    def __init__(self, spectrum, labels=None):
        pixels = spectrum[0]
        fluxes = spectrum[1]
        flux_errs = spectrum[2]
        
        self.pixels = pixels
        self.fluxes = fluxes
        self.flux_errs = flux_errs

        self.labels = labels # only for training set

    # You can imagine there being some kind of def sanity_check() here, where the self-consistency of all of the input data is tested, ex. that the length of the arrays match, etc
