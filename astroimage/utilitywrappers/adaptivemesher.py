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

    upbinFunction: `function`
        A user defined funciton which returns the binned input object to the
        original binning scale. This function should be used to guarantee that
        all the data are correctly rebinned rather than simply assuming that the
        inverse `__getitem__` procedure will produce the correct results.

        Example format:

        def upbinData(inputImg, rebinFactor):
            # Compute the output image shape
            ny, nx = [rebinFactor*d for d in inputImg.shape]
            outputImg = inputImg.rebin((ny, nx))

            # Perhaps I need to adjust the rebinned uncertainties
            outputImg.uncertainty = outputImg.uncertainty/np.sqrt(rebinFactor**2)

            return outputImg

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

    def __init__(self, inputObject, binningStatistic, upbinFunction, dataCopier):
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

        # Store the upbinning function
        self.__upbinFunction = upbinFunction

        # Store the cdataCpier function
        self.__dataCopier = dataCopier

    @property
    def data(self):
        """Provides a quick reference to the original input"""
        return self._AdaptiveMesher__inputObject

    def _crop_data(self):
        """Crops data so rebinning produces an integer number of cells"""
        # Begin by performing all the necessary rebinnings and storing the
        # results in a dictionary. The first step of this is to pad or crop the
        # image accordingly.
        maxBinning = np.max(self.rebinLevels)
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

    def _compute_rebinnings(self):
        """Computes all possible rebinnings of the input data"""
        # Having cropped the Image (or StokesParameters), perform all possible
        # rebinnings and store them in a dictionary
        rebinDict  = {}
        for rebin in self.rebinLevels:
            # Down-size the array via rebinning
            tmpData = self.croppedData[::rebin, ::rebin]

            # Store the data in the rebinDict
            rebinDict[rebin] = tmpData

        # Store the rebinned data
        self.rebinnedData = rebinDict

    def _test_rebinnings(self, thresholdValue):
        """
        Tests the SNR of each rebin level

        To pass this test, clusters of 2x2 pixels must have at least all
        four sub-pixels pass the test statistic at the *finer* binning.

        Parameters
        ----------
        thresholdValue : `int` or `float`
            The minimum value for the binningStatistic (defined by the function)
            provided when initalizing the objet). Any cells which do not meet
            this thresholding value will continue to be rebinned until the
            binningStatistic function returns a value greater than this minimum
            thresholding value.
        """
        # Grab the shape of the *unbinned* data
        unbinnedShape = self.croppedData.shape

        # Loop through each rebin level and test if all four of its
        # sub-pixels pass the SNR test.
        binningTestDict = {}
        for rebin in self.rebinLevels[:-1]:
            # Compute the supplied test statistic
            testStatistic = self._AdaptiveMesher__binningStatistic(self.rebinnedData[rebin])

            # Rebin the fineTest image to see which 2x2 clusters have all four
            # pixels passing the threshold value test at the finer binning level.
            # Convert the arrays to ReducedScience images for rebinning
            testStatisticImg = ReducedScience(
                (testStatistic  > thresholdValue).astype(float)
            )

            # Grab the shape of the current array
            ny, nx = testStatistic.shape

            # Check if all four pixels in each cluster pass the binning test
            binnedTest  = testStatisticImg.rebin((ny//2, nx//2))
            binnedTest  = binnedTest.rebin((ny, nx))
            binningPass = (binnedTest.data == 1).astype(int)

            # Expand the binnig test using the ReducedScience `rebin` method
            binningPass = ReducedScience(binningPass).rebin(unbinnedShape)

            # Store the results in the binningTestDict dictionary
            binningTestDict[rebin] = binningPass.data
        else:
            # Treat the final rebin level
            rebin = self.rebinLevels[-1]

            # Compute the supplied test statistic
            testStatistic = self._AdaptiveMesher__binningStatistic(self.rebinnedData[rebin])

            # Determine if this statistic passed the test
            binningPass = (testStatistic  > thresholdValue).astype(int)
            binningPass = ReducedScience(binningPass).rebin(unbinnedShape)

            # Store the results of this final binning level
            binningTestDict[rebin] = binningPass.data

        # Store the binning tests in the object for later reference
        self.binningTestResults = binningTestDict

    def _assign_final_binnings(self):
        """"""
        # Generate an array in which to store the final binning levels
        self.amrBinnings = -1*np.ones(self.croppedData.shape)

        # Create a *reverse* order array of rebin levels
        reverseSortInds    = self.rebinLevels.argsort()[::-1]
        reverseRebinLevels = self.rebinLevels[reverseSortInds]

        # Loop through each binning level (except the 1x1 binning) and test if
        # the binning passed the minimum threshold value
        for rebin in reverseRebinLevels[:-1]:
            # Determine if this binning level passes the test
            thisRebinTest = self.binningTestResults[rebin]

            # Determine if the next level down passes the test
            nextRebinTest = self.binningTestResults[rebin//2]

            # If this level passed the test and the next level down passed the
            # test, then keep digging. However, if this level passed the test,
            # but the next level down did not pass the test, then use this level
            proposedPixelsToAssign = np.logical_and(
                thisRebinTest,
                np.logical_not(nextRebinTest)
            )

            # Locate which pixels have *already* been assigned to a binning
            unassignedPixels = self.amrBinnings < 0

            # Locate which pixels have not yet been assigned but should be
            assignPixels = np.logical_and(
                proposedPixelsToAssign,
                unassignedPixels
            )
            assignInds = np.where(assignPixels)

            # Assign these pixels to the current rebin level
            self.amrBinnings[assignInds] = rebin
        else:
            # Determine which pixels to permit with the *finest* binning level
            rebin = reverseRebinLevels[-1]

            # Determine if this binning level passes the test
            thisRebinTest = self.binningTestResults[rebin]

            # Locate which pixels have *already* been assigned to a binning
            unassignedPixels = self.amrBinnings < 0

            # Locate which pixels have not yet been assigned but should be
            assignPixels = np.logical_and(
                thisRebinTest,
                unassignedPixels
            )
            assignInds = np.where(assignPixels)

            # Assign these pixels to the current rebin level
            self.amrBinnings[assignInds] = rebin

    def _copy_final_data(self):
        """Uses the provided copy function to copy over final, binned data"""
        # Re-compute the available rebninning levels
        # Copy the rebinned data over to the final output object
        adaptiveMeshedData = copy.deepcopy(self.croppedData)

        # Loop through each binning level and copy over the data to the final
        # output object
        for rebin in self.rebinLevels[1:]:
            # Up-bin the data to its original shape
            tmpData = self._AdaptiveMesher__upbinFunction(
                self.rebinnedData[rebin], rebin
            )

            # Locate the indices assigned to this rebinning level
            copyInds = np.where(self.amrBinnings == rebin)

            # Use the supplied dataCopier function to copy data from this
            # rebinning level to the final output data structure
            self._AdaptiveMesher__dataCopier(
                tmpData,
                adaptiveMeshedData,
                copyInds
            )

        # Store the final, rebinned output data
        self.adaptiveMeshedData = adaptiveMeshedData

    def run(self, thresholdValue, maxRebins=3):
        """
        Executes an adaptive mesh algorithm using the provided function and values

        Parameters
        ----------
        thresholdValue: `int` or `float`
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

        # Compute each of the rebin levels to use and store for later use
        self.rebinLevels = 2**np.arange(maxRebins + 1)

        # Crop data so that rebinning always provides an integer number of cells
        self._crop_data()

        # Compute all possible rebinnings which will be later evaluated
        self._compute_rebinnings()

        # Parse the rebinned data to decite which binning level is best
        self._test_rebinnings(thresholdValue)

        # Now that the binning levels have been tested, determine which
        # binning levels should be used to fill which pixels
        self._assign_final_binnings()

        # Now that all the optimal binning levels have been computed, copy the
        # data over from the rebinned data objects into the final output object
        self._copy_final_data()

        # Return the pseudo-AMR data to the user
        return self.adaptiveMeshedData

    def reapply_amr(self, inArray):
        """
        Applies the AMR results to additional arrays.

        This can be usefor for computing the rebinning based on statistics from
        one set of objects but applying the identical binning scheme to an
        alternative set of arrays.

        Parameters
        ----------
        inArray : numpy.ndarray
            An array of numbers to be adaptive mesh rebinned

        Returns
        -------
        outArray : numpy.ndarray
            The adaptive mesh rebinned version of the input array
        """
        # Test if amrBinning has already been produced
        if not hasattr(self, 'amrBinnings'):
            raise RuntimeError('AdaptiveMesher has not yet been run')

        # Grab the shape of the input array
        ny, nx = inArray.shape

        # Identify which pixels are finite
        finitePix = np.isfinite(inArray).astype(int)

        # Loop through each binning level and compute the AMR outArray
        outArray = inArray.copy()
        for rebinLevel in np.unique(self.amrBinnings).astype(int):
            # Skip over the trivial versions
            if rebinLevel <= 1: continue

            # Compute the rebinning shape for this level
            sh = (
                ny//rebinLevel, rebinLevel,
                nx//rebinLevel, rebinLevel
            )

            # Compute the rebinned xx and yy arrays
            # First down-sample...
            tmpArray = np.nansum(np.nansum(inArray.reshape(sh), axis=-1), axis=1)

            # then up-sample
            tmpArray = np.kron(tmpArray, np.ones((rebinLevel, rebinLevel)))

            # Divide by

            # Replace the corresponding indices with these rebinned values
            copyInds = np.where(self.amrBinnings == rebinLevel)
            outArray[copyInds] = tmpArray[copyInds]

        return outArray
