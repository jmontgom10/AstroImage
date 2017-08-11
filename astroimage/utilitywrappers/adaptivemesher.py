import copy

class AdaptiveMesher(object):
    """
    Rebins Image (or StokesParameter) objects using the provided statistic.

    Whatever object is supplied to instantiate the AdaptiveMesher object, it
    must have a rebin method which is available via the `__getitem__` method.

    Parameters
    ----------
    binningStatistic: `function`
        A unser defined function which takes the input object (e.g., image
        or StokesParameter object) as the only argument and which returns the
        statistic used to determine if an optimal binning level has been
        reached. In the case of polarization percentage (for which this class
        was designed), the thresholding statistic is the polarimetric
        signal-to-noise ratio.

    thresholdValue : `int` or `float`
        The minimum value for the binningStatistic. Any cells which do not meet
        this thresholding value will continue to be rebinned until the
        binningStatistic function returns a value greater than this minimum
        thresholding value.

    Methods
    -------
    run

    Examples
    --------
    Read in Stokes images
    >>> from astroimage import ReducedScience
    >>> Iimg = ReducedScience.read('Iimg.fits')
    >>> Qimg = ReducedScience.read('Qimg.fits')
    >>> Uimg = ReducedScience.read('Uimg.fits')
    >>> stokesDict = {'I':Iimg, 'Q':Qimg, 'U':Uimg}

    Construct a StokesParameters object
    >>> from astroimage.utilitiywrappers import StokesParameters
    >>> stokesParams = StokesParameters(stokesDict)

    Define a binning statistic function and threshold value
    >>> def Psnr(stokesPar):
    ...     P, _ = stokesPar.compute_polarization_images(
    ...         P_estimator='NAIVE', minimum_SNR=0.0)
    ...     return P.snr
    ...
    >>> minimum_SNR = 2.5

    Create an AdaptiveMesher object from the StokesParameters object
    >>> from astroimage.utilitiywrappers import AdaptiveMesher
    >>> adaptiveMesher = AdaptiveMesher(
    ...     stokesParams, binningStatistic=Psnr, thresholdValue=minimum_SNR)

    Run the adaptive mesh routine
    >>> adaptedMesh = adaptiveMesher.run()

    Construct the polarization map from the adaptively meshed Stokes parameters
    >>> P, PA = adaptedMesh.compute_polarization_images(
    ...     P_estimater='ASYMPTOTIC', minimum_SNR=minimum_SNR)

    Look at the beautiful maps
    >>> P.show(vmin=0, vmax=0.3, cmap='inferno')
    >>> PA.show(vmin=0, vmax=180, cmap='rainbow')
    """

    def __init__(self, inputObject, binningStatistic, thresholdValue):
        """Builds the AdaptiveMesher object."""
        # Test if the input object has a __getitem__ method defined
        if not hasattr(inputObject, '__getitem__'):
            raise TypeError('`inputObject` must have a `__getitem__` method defined')

        # Test if the input binningStatistic function is callable
        if not callable(binningStatistic):
            raise TypeError('`binningStatistic` must be a callable function')

        # Test if the input thresholdValue scalar is in fact a scalar
        if hasattr(thresholdvalu, '__iter__'):
            raise TypeError('`thresholdValue` must be a scalar')

        # Store the input object (of any type)
        self.__inputObject = copy.deepcopy(inputObject)

        # Store the binningStatistic function
        self.__binningStatistic = binningStatistic

        # Store the thresholdValue
        self.__thresholdValue = thresholdValue

    @property
    def data(self):
        """Provides a quick reference to the input data in its current state"""
        return self._AdaptiveMesher__inputObject

    def run(self):
        """Executes an adaptive mesh algorithm using the provided function and values"""
        raise NotImplementedError
