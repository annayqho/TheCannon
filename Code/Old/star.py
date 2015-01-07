# Initialize all of the data: spectra and training labels

class Star(object):
    ' Common base class for all stars, regardless of survey, training and test '
    def __init__(self, ID, spectrum, labels=None):
        self.ID = ID
        self.spectrum = spectrum
        self.labels = labels
    
    def getID(self):
        return self.ID

    def getPixels(self):
        return self.spectrum[0]

    def getFluxes(self):
        return self.spectrum[1]

    def getFluxErrs(self):
        return self.spectrum[2]

    def getLabelNames(self):
        return self.labels[0]

    def getLabelValues(self):
        return self.labels[1]

    # You can imagine there being some kind of def sanity_check() here, where the self-consistency of all of the input data is tested, ex. that the length of the arrays match, etc
