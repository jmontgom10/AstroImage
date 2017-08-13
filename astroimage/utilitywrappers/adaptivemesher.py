# Import core modules
import copy

# Import scipy modules
import numpy as np

# Import the ReducedScience class to provide rebinning functionality
from astroimage.reduced import ReducedScience

class AdaptiveMesher(object):
    """
    Rebins Image (or StokesParameter) objects using the provided statistic.

    Whatever object is supplied to instantiate the AdaptiveMesher object, it
    must have a rebin method which is available via the `__getitem__` method.

    Parameters
    ----------
    binningStatistic: `function`
        A user defined function which takes the input object (e.g., image or
        StokesParameter object) as the only argument and which returns an array
        containing the statistic used to determine if an optimal binning level
        has been reached. In the case of polarization percentage (for which this
        class was designed), the thresholding statistic is the polarimetric
        signal-to-noise ratio for each pixel.

        Example format:

        def binningStatistic(img):
            return img.data/img.uncertainty

    dataCopier: `function`
        A user defined funciton which takes two input objects (e.g., image or
        StokesParameter object) and a set of indices and copies data from one
        object to the other. Since the AdaptiveMesher class has no way of
        knowing exactly where and how the data are stored inside the input
        objects, the user is responsible for providing this information. This
        function must take the source object as its first argument, the
        destination object as its second argument, and the indices to copy as
        its third argument.

        Example format:

        def copyData(destImg, sourceImg, inds):
            img1.data[inds] = img2.data[inds]

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

    Define a data copying function
    >>> def stokesCopy(sourceStokesPar, destinationStokesPar, inds):
    ...     destinationStokesPar.I.data[inds] = sourceStokesPar.I.data[inds]
    ...     destinationStokesPar.Q.data[inds] = sourceStokesPar.Q.data[inds]
    ...     destinationStokesPar.U.data[inds] = sourceStokesPar.U.data[inds]

    Create an AdaptiveMesher object from the StokesParameters object
    >>> from astroimage.utilitiywrappers import AdaptiveMesher
    >>> adaptiveMesher = AdaptiveMesher(
    ...     stokesParams,
    ...     binningStatistic=Psnr,
    ...     dataCopy=stokesCopy)

    Run the adaptive mesh routine
    >>> adaptedMesh = adaptiveMesher.run(minimum_SNR)

    Construct the polarization map from the adaptively meshed Stokes parameters
    >>> P, PA = adaptedMesh.compute_polarization_images(
    ...     P_estimater='ASYMPTOTIC', minimum_SNR=minimum_SNR)

    Look at the beautiful maps
    >>> P.show(vmin=0, vmax=0.3, cmap='inferno')
    >>> PA.show(vmin=0, vmax=180, cmap='rainbow')
    """

    def __init__(self, inputObject, binningStatistic, dataCopier):
        """Builds the AdaptiveMesher object."""
        # Test if the input object has a __getitem__ method defined
        if not hasattr(inputObject, '__getitem__'):
            raise TypeError('`inputObject` must have a `__getitem__` method defined')

        # Test if the input binningStatistic function is callable
        if not callable(binningStatistic):
            raise TypeError('`binningStatistic` must be a callable function')

        # Store the input object (of any type)
        self.__inputObject = copy.deepcopy(inputObject)

        # Store the binningStatistic function
        self.__binningStatistic = binningStatistic

        # Store the cdataCpier function
        self.__dataCopier = dataCopier

    @property
    def data(self):
        """Provides a quick reference to the original input"""
        return self._AdaptiveMesher__inputObject

    def _prepare_data(self, maxRebins):
        """Crops data so rebinning produces an integer number of cells"""
        # Begin by performing all the necessary rebinnings and storing the
        # results in a dictionary. The first step of this is to pad or crop the
        # image accordingly.
        maxBinning = 2**(maxRebins)
        ny, nx = self.data.shape
        cutRt  = nx % maxBinning
        cutTp  = ny % maxBinning

        # If the current image dimensions are not correct, then crop the data!
        if cutRt > 0 or cutTp > 0:
            print('Cropping {} pix from right and {} pix from top of image(s)'.format(
                cutRt, cutTp
            ))
            # Apply the crops to the inputObject
            tmpData = self.data[0:ny-cutTp, 0:nx-cutRt]
        else:
            tmpData = self.data

        # Store the cropped data
        self.croppedData = tmpData

    def _compute_rebinnings(self, maxRebins):
        """Computes all possible rebinnings of the input data"""
        # Having cropped the Image (or StokesParameters), perform all possible
        # rebinnings and store them in a dictionary
        rebinDict  = {}
        rebinnings = 2**np.arange(maxRebins + 1)
        for rebin in rebinnings:
            # Down-size the array via rebinning
            tmpData = self.croppedData[::rebin, ::rebin]

            # Store the data in the rebinDict
            rebinDict[rebin] = tmpData

        # Store the rebinned data
        self.rebinnedData = rebinDict

    def _test_refine_binning(self, finerBinningBool):
        """
        Returns an array indicating which pixels can use a finer rebinning

        To pass this test, clusters of 2x2 pixels must have at least 3 pixels
        pass the test statistic at the *finer* binning.

        Parameters
        ----------
        finerBinningBool: numpy.array
            Array indicating which individual pixels passed the threshold test
            at the finer binning level.
        """
        # Convert the arrays to ReducedScience images for rebinning
        fineBoolImg = ReducedScience(finerBinningBool.astype(float))

        # Grab the shape of the current finerBinningBool array
        ny, nx = fineBoolImg.shape

        # Rebin the fineTest image to see which 2x2 clusters have at least 3
        # pixels passing the threshold value test at the finer binning level.
        binnedTest       = fineBoolImg.rebin((ny//2, nx//2))
        binnedTest       = 4*binnedTest.rebin((ny, nx))
        finerBinningPass = binnedTest.data >= 3

        return finerBinningPass

    def _parse_rebinnings(self, thresholdValue):
        """Determines the minimum rebinning which meets thresholdValue."""
        # Grab the available rebinning levels
        rebinnings      = np.array([k for k in self.rebinnedData.keys()])
        sortInds        = rebinnings.argsort()
        reverseSortInds = sortInds[::-1]
        forwardBinnings = rebinnings[sortInds]
        reverseBinnings = rebinnings[reverseSortInds]

        # Loop through each rebinning (from greatest to least) and test if
        # the supplied statistic meets the selected threshold value
        testResults = {}
        for rebin in reverseBinnings:
            # If the statistic meets the threshold, then store that rebinning
            testStat           = self._AdaptiveMesher__binningStatistic(self.rebinnedData[rebin])
            testResults[rebin] = (testStat >= thresholdValue).astype(int)

        # Grab the original shape of the unbinned cropped data
        unbinnedShape = self.croppedData.shape

        # Start by assuming the maximal binning level, and only permit
        # refinement if 3-out-of-4 sub-pixels pass the threshold at the finer
        # binning level
        finalBinnings = np.ones(unbinnedShape, dtype=int)

        # Loop back through the rebinnings and determine the ideal binning
        for rebin in reverseBinnings[1:]:
            # Test whether the next finer rebinning level passes the 3-pixel test
            finerBinningPass = self._test_refine_binning(testResults[rebin])

            # Now that it has been determined if this finer binning passes the
            # test, expand the test results up to the original unbined shape
            finerBinningPass = ReducedScience(finerBinningPass.astype(int))
            finerBinningPass = finerBinningPass.rebin(unbinnedShape)

            # Store the results of this rebinning test
            # First, figure out which parts of the image *can* have their
            # binning level updated. Any pixels which have already had a binning
            # level assignd *cannot* have their binning level updated. If the
            # propesd finer binning level *failed* the finerBinningPass test,
            # then assign it the *previous* level of binning (i.e., rebin//2)

            # Locate the pixels which passed this proposed binning level
            failedPixels = (finerBinningPass.data == 0)

            # Locate any pixels which have already had a binning level assined
            unassignedPixels = (finalBinnings == 1)

            # The pixels to be assigned at this pass are those which failed and
            # are currently 'unassigned' to a specific binning level
            assignPixels = np.logical_and(failedPixels, unassignedPixels)

            # Assign the identified pixels to the *previous* binning level
            finalBinnings[np.where(assignPixels)] = rebin*2

        # Locate pixels which do not ever pass the *maximal* binning
        maximalBinningPass = self._test_refine_binning(testResults[forwardBinnings.max()])
        maximalBinningPass = ReducedScience(maximalBinningPass.astype(int))
        maximalBinningPass = maximalBinningPass.rebin(unbinnedShape)

        # Find pixels assigned to the maximal binning but which do not pass the
        # maximal binning test
        totalFailurePix = np.logical_and(
            finalBinnings == forwardBinnings.max(),
            np.logical_not(maximalBinningPass.data)
        )

        # Assign those pixels which fail even at the maximal binning to actually
        # have the finest binning (since no value was obtained with binning up
        # pixels)
        finalBinnings[np.where(totalFailurePix)] = 1

        # Store the final binings for the user to check out if they want
        self.amrBinnings = finalBinnings

        # Copy the rebinned data over to the final output object
        adaptiveMeshedData = copy.deepcopy(self.croppedData)
        for rebin in forwardBinnings[1:]:
            # Up-bin the data to its original shape
            tmpData = (self.rebinnedData[rebin])[::np.float(1.0/rebin), ::np.float(1.0/rebin)]

            # Locate the indices assigned to this rebinning level
            copyInds = np.where(finalBinnings == rebin)

            # Use the supplied dataCopier function to copy data from this
            # rebinning level to the final output data structure
            self._AdaptiveMesher__dataCopier(
                tmpData,
                adaptiveMeshedData,
                copyInds
            )

        self.adaptiveMeshedData = adaptiveMeshedData

    def run(self, thresholdValue, maxRebins=3):
        """
        Executes an adaptive mesh algorithm using the provided function and values

        Parameters
        ----------
        thresholdValue : `int` or `float`
            The minimum value for the binningStatistic (defined by the function)
            provided when initalizing the objet). Any cells which do not meet
            this thresholding value will continue to be rebinned until the
            binningStatistic function returns a value greater than this minimum
            thresholding value.

        maxRebins: int, opotional, default: 3
            The maximum number of 2x2 rebinning to perform. If this level of
            rebinning does not meet the `thresholdValue` provided, then the data
            can be assumed to be noise dominated.
        """
        # Test if the input thresholdValue scalar is in fact a scalar
        if hasattr(thresholdValue, '__iter__'):
            raise TypeError('`thresholdValue` must be a scalar')

        # Test if maxRebins is an integer
        if not np.float(maxRebins).is_integer():
            raise TypeError('`maxRebins` must be an integer')

        # Crop data so that rebinning always provides an integer number of cells
        self._prepare_data(maxRebins)

        # Compute all possible rebinnings which will be later evaluated
        self._compute_rebinnings(maxRebins)

        # Parse the rebinned data to decite which binning level is best
        self._parse_rebinnings(thresholdValue)

        # Return the pseudo-AMR data to the user
        return self.adaptiveMeshedData
