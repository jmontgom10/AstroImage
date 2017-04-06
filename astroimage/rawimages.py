# This tells Python 2.x to handle imports, division, printing, and unicode the
# way that `future` versions (i.e. Python 3.x) handles those things.
from __future__ import absolute_import, division, print_function, unicode_literals

# Core imports
import copy
import warnings
from functools import lru_cache

# Scipy imports
import numpy as np

# Astropy imports
from astropy.nddata import NDDataArray

# AstroImage imports
from .baseimage import BaseImage, ClassProperty
from .reducedimages import ReducedImage, MasterBias, MasterDark, MasterFlat
from .reducedscience import ReducedScience

# Define which functions, classes, objects, etc... will be imported via the command
# >>> from .rawimages import *
__all__ = ['RawImage', 'RawBias', 'RawDark', 'RawFlat', 'RawScience']

class RawImage(BaseImage):
    """
    The base class for raw data from telescope/instrument.

    Provides methods for performing data reduction techniques.

    Properties
    ----------
    airmass         The airmass of the observation
    axes            The Axes instance storing the plotted image (if plotted)
    data            The actual 2D numpy array of the image in the fits file
    binning         The binning of the image as a tuple, returned as binning
                    along the height axis, then width axis
    date            The UTC date and time of the observation in the format
                    YYYY-MM-DD   HH:MM:SS.SS
    dec             The declination of the observation in the format
                    +DD:(AM)(AM):(AS)(AS).(AS)(AS)
    dtype           The data type of the image array (see `numpy.ndarray`)
    expTime         The total exposure time, in seconds
    figure          The Figure instance storing the plotted axes (if plotted)
    filename        The image filename on disk
    filter          The filter through which the image was obtained
    header          The header info for the associated fits file
    height          The height of the image, in pixels
    image           The AxesImage storing the plotted data (if plotted)
    instrument      The instrument from which the image was obtained
    overscan        The overscan region of the FITS image
    overscanPix     The width of the overscan region
    prescan         The prescan region of the FITS image
    prescanPix      The width of the prescan region
    ra              The right ascension of the observation in the format
                    HH:MM:SS.SS
    shape           Dimensions of the image as a tuple, returned as height, then
                    width, in pixels, in keeping with the behavior of the
                    `numpy.ndarray` size attribute
    unit            The units of the numpy 2D array stored in `data`
    width           The width of the image, in pixels


    Class Methods
    -------------
    set_headerKeywordDict
    read

    Methods
    -------
    set_arr
    set_header
    copy
    write
    rebin
    show

    Examples
    --------
    Read in two simple fits files
    >>> from astroimage import RawScience
    >>> img1 = RawScience('Rimg1.fits')
    >>> img2 = RawScience('Rimg2.fits')

    Check that the images are the same dimensions
    >>> img1.shape, img2.shape
    ((500, 500), (500, 500))

    Now compute the mean of those two images
    >>> img3 = 0.5*(img1 + img2)

    Rebin the resulting image
    >>> img3.rebin(250, 250)

    Check that the new image has the correct dimensions
    >>> img3.shape
    (250, 250)

    Display the resultant average, rebinned image
    >>> fig, ax, axImg = img3.show()
    """

    ##################################
    ### START OF CLASS VARIABLES   ###
    ##################################

    # Extend the list of acceptable properties for this class
    __properties = copy.deepcopy(BaseImage.properties)
    __properties.extend([
        'prescanArray',
        'prescanWidth',
        'overscanArray',
        'overscanWidth'
    ])

    ##################################
    ### END OF CLASS VARIABLES     ###
    ##################################

    ##################################
    ### START OF CLASS METHODS     ###
    ##################################

    @ClassProperty
    @classmethod
    def properties(cls):
        return cls.__properties

    ##################################
    ### END OF CLASS METHODS       ###
    ##################################

    def __init__(self, *args, **kwargs):
        # Invoke the parent class constructor
        super(RawImage, self).__init__(*args, **kwargs)

        # Indicate that the array has not yet been overscan corrected
        self.__overscanCorrected = False

        # If either prescan or overscan areas were defined, then ensure the
        # `data` attribute does not include those regions
        if (self.prescanArray is not None) or (self.overscanArray is not None):
            # Copy the full data array
            data = self.data.copy()

            # Overwrite the array so that it doesn't include the precscan data
            self._BaseImage__fullData = NDDataArray(
                data[:, self.prescanWidth:-self.overscanWidth],
                uncertainty=self._BaseImage__fullData.uncertainty,
                unit=self._BaseImage__fullData.unit,
                wcs=self._BaseImage__fullData.wcs
            )

            # If an overscan region was defined, then do the overscan correction
            if (self.overscanArray is not None) and (not self.overscanCorrected):
                self._apply_overscan_correction()

        # Raw images with no units should be assumed to have ADU units
        if self.has_dimensionless_units:
            # Overwrite the array and give it ADU units
            self._BaseImage__fullData = NDDataArray(
                self._BaseImage__fullData.data,
                uncertainty=self._BaseImage__fullData.uncertainty,
                unit='adu',
                wcs=self._BaseImage__fullData.wcs
            )

            # Attempt to add the ADU unit to the image header
            try:
                unitKey = self.headerKeywordDict['UNIT']
                self._BaseImage__header[unitKey] = '{0:FITS}'.format(self.unit)
            except: pass

    ##################################
    ### START OF PROPERTIES        ###
    ##################################

    @property
    def prescanArray(self):
        """The array of prescan values"""
        return self.__prescanArray

    @property
    def prescanWidth(self):
        """The width of the region trimmed for the prescan data"""
        if self.prescanArray is not None:
            return self.__prescanWidth
        else:
            return None

    @property
    def overscanArray(self):
        """The array of overscan values"""
        return self.__overscanArray

    @property
    def overscanWidth(self):
        """The width of the region trimmed for the overscan data"""
        if self.overscanArray is not None:
            return self.__overscanWidth
        else:
            return None

    @property
    def overscanCorrected(self):
        """Boolean flag signaling if the array has been overscan corrected"""
        return self.__overscanCorrected

    ##################################
    ### END OF PROPERTIES          ###
    ##################################

    ##################################
    ### START OF STATIC METHODS    ###
    ##################################

    @staticmethod
    def _where_jumps(arr, uncert=None, sig=3):
        """
        Method to detect jumps in a 1D numpy.ndarray.

        Parameters
        ----------
        arr : array_like
            The array in which the user wants to find any significant jumps

        uncert : scalar or array_like, optional, default: None
            The uncertainty in the array values

        sig : int or float, optional, default: 3
            The minimum significance of an array jump to be considered a
            detection (uncertainty units)

        Returns
        -------
        out : tuple
            The indices of the significantly detected jump locations
        """
        # Check that the input array in fact an array
        try:
            arr1 = np.array(arr)
        except:
            raise TypeError('`arr` must be a array_like')

        if arr1.ndim != 1:
            raise ValueError('`arr` must be a 1D array')

        # Check if the uncert value is a scalar
        uncertIsScalar = isinstance(uncert,
            (int, np.int8, np.int16, np.int32, np.int64,
            float, np.float32, np.float64))
        if uncert is None:
            # If no uncertainty was provided, then estimate it from the array.
            def rolling_window(a, window):
                shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
                strides = a.strides + (a.strides[-1],)
                return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

            # Compute a rolled sampling of the input array
            rolledArr = rolling_window(arr1, 5)
            rolledArr = np.vstack([
                rolledArr[0], rolledArr[0],
                rolledArr,
                rolledArr[-1], rolledArr[-1]
                ])

            # Compute the rolled standard deviation as the uncertainty
            uncert1 = np.std(rolledArr, axis=1)

        elif uncertIsScalar:
            # If a scalar uncertainty was provided, then broadcast into array
            uncert1 = uncert*np.ones(arr1.shape)

        elif type(uncert) is np.ndarray:
            uncert1 = uncert

        else:
            raise TypeError('`uncert` must be a scalar or array_like value')

        # Now that all the pre-requisites are met, let's compute the 1D gradient
        arrGrad    = np.abs(np.convolve(arr, [-1,0,1], mode='same'))
        gradUncert = np.sqrt(np.convolve(uncert**2, [1, 0, 1], mode='same'))

        # Kill of the edge effects
        arrGrad[1]  = 0.0
        arrGrad[-1] = 0.0

        # Now, simply test for where the gradient exceeds the sigma-cut
        return np.where(arrGrad > sig*gradUncert)

    ##################################
    ### START OF OTHER METHODS     ###
    ##################################

    def _dictionary_to_properties(self, propDict):
        # Call the parent method
        super(RawImage, self)._dictionary_to_properties(propDict)

        # Extend the method to include prescan and overscan properties
        ###
        # Extract the prescan region
        ###
        # Test if a prescan area has been defined
        prescanTest = 'prescanWidth' in propDict
        if prescanTest:
            # Estimate the prescan region from the keyword in the header
            try:
                prescanWidth = int(propDict['prescanWidth'])
            except:
                raise TypeError('`prescanWidth` property must be convertible to an integer')

            # Find the binning-specific prescan width and store it for reference
            prescanPix1 = (prescanWidth + 40)//self.binning[0]
            self.__prescanWidth = prescanPix1

            # Grab the prescan data
            prescan1    = self.data[:, 0:prescanPix1]

            # Find the jump (if possible) from prescan to sky background
            prescanRowValues = np.median(prescan1, axis=0)
            prescanRowUncert = np.std(prescan1, axis=0)

            # Find the inds of the jumps
            jumpInds = RawImage._where_jumps(
                prescanRowValues, uncert=prescanRowUncert
            )
            jumpInds = jumpInds[0]

            if jumpInds.size > 1:
                # If at least 2 jumps were found, then find the largest region
                # between jumps
                widthsBetweenJumps = jumpInds[1:] - jumpInds[:-1]
                maximumWidthInd    = widthsBetweenJumps.argmax()

                # Grab JUST the relevant indices
                jumpInds = jumpInds[maximumWidthInd:maximumWidthInd+2]

                # Reassign the prescan array
                prescan2 = prescan1[:, jumpInds[0]:jumpInds[1]]

            elif jumpInds.size == 1:
                # If only one jump was found, then trim that jump and store it
                jumpInds = jumpInds[0]

                # Grab the LARGER of the two halves of the prescan region
                # (hopefully that is correct!)
                if jumpInds < prescanPix1//2:
                    prescan2 = prescan1[:, jumpInds:]
                if jumpInds >= prescanPix1//2:
                    prescan2 = prescan1[:, :jumpInds]

            else:
                prescan2 = prescan1

            # Store the prescan array in the prescan attribute
            self.__prescanArray = prescan2

        else:
            # If no prescan region was defined at all, then simply set a null.
            self.__prescanArray = None

        ###
        # Extract the overscan region
        ###
        # Test if an overscan area has been defined
        overscanTest = 'overscanWidth' in propDict
        if overscanTest:
            # Estimate the prescan region from the keyword in the header
            try:
                overscanWidth = int(propDict['overscanWidth'])
            except:
                raise TypeError('`overscanWidth` property must be convertible to an integer')

            # Find the binning-specific prescan width and store it for reference
            overscanPix1 = (overscanWidth + 40)//self.binning[0]
            self.__overscanWidth = overscanPix1

            # Grab the overscan data
            overscan1    = self.data[:, -overscanPix1:]

            # Find the jump (if possible) from overscan to sky background
            overscanRowValues = np.median(overscan1, axis=0)
            overscanRowUncert = np.std(overscan1, axis=0)

            # Find the inds of the jumps
            jumpInds = RawImage._where_jumps(
                overscanRowValues, uncert=overscanRowUncert)
            jumpInds = jumpInds[0]

            if jumpInds.size > 1:
                # If at least 2 jumps were found, then find the largest region
                # between jumps
                widthsBetweenJumps = jumpInds[1:] - jumpInds[:-1]
                maximumWidthInd    = widthsBetweenJumps.argmax()

                # Grab JUST the relevant indices
                jumpInds = jumpInds[maximumWidthInd:maximumWidthInd+2]

                # Reassign the overscan array
                overscan2 = overscan1[:, jumpInds[0]:jumpInds[1]]

            elif jumpInds.size == 1:
                # If only one jump was found, then trim that jump and store it
                jumpInds = jumpInds[0]

                # Grab the LARGER of the two halves of the prescan region
                # (hopefully that is correct!)
                if jumpInds < overscanPix1//2:
                    overscan2 = overscan1[:, jumpInds:]
                if jumpInds >= overscanPix1//2:
                    overscan2 = overscan1[:, :jumpInds]

            else:
                overscan2 = overscan1

            # Store the overscan array in the overscan attribute
            self.__overscanArray = overscan2

        else:
            # If no prescan region was defined at all, then simply set a null.
            self.__overscanArray = None

    def _properties_to_header(self):
        # Call the parent method
        super(RawImage, self)._properties_to_header()

        # Extend the method to include prescan and overscan properties
        # If a prescan region was defined, then set it in the header
        if self.prescanWidth is not None:
            try:
                prescanKey = self.headerKeywordDict['PRESCANWIDTH']
                self._BaseImage__header[prescanKey] = self.prescanWidth
            except: pass
        else:
            try:
                prescanKey = self.headerKeywordDict['PRESCANWIDTH']
                del self._BaseImage__header[prescanKey]
            except: pass

        # If an overscan region was defined, then set it in the header
        if self.overscanWidth is not None:
            try:
                overscanKey = self.headerKeywordDict['OVERSCANWIDTH']
                self._BaseImage__header[overscanKey] = self.overscanWidth
            except: pass
        else:
            try:
                overscanKey = self.headerKeywordDict['OVERSCANWIDTH']
                del self._BaseImage__header[overscanKey]
            except: pass

    def _apply_overscan_correction(self):
        """Internal "private" method to apply overscan correction"""
        # Test if an overscan has been defined. If not, then just return
        if self.overscanArray is None:
            raise AttributeError('There is no overscan in this image.')

        # Exit early if the overscan correction has already been applied
        if self.overscanCorrected:
            return

        try:
            # Try to use sklearn for regularized linear regression
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.pipeline import make_pipeline
            from sklearn.linear_model import Lasso
            regularizedLinearRegression = True
        except:
            # Otherwise simply use a 3rd order polynomial and hope for the best.
            regularizedLinearRegression = False

        # Compute the median row behavior of prescan/overscan
        # medianPrescanCol  = np.median(self.prescanArray,  axis=1)
        medianOverscanCol = np.median(self.overscanArray, axis=1)

        if regularizedLinearRegression:
            # Generate an estimator using up to 9th degree polynomial
            degree = 12
            alpha = 5e-3

            # Generate sapmling values
            X = np.arange(medianOverscanCol.size)[:, np.newaxis]
            y = medianOverscanCol[:, np.newaxis]

            # Build an estimator using the LASSO procedure
            est = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=alpha))

            # Peform the estimation
            est.fit(X, y)

            # Get the fitted overscan column
            fittedOverscanCol = est.predict(X)

        else:
            # If sklearn is not installed, then just do 3rd order polynomial
            x = np.arange(medianOverscanCol.size)
            y = medianOverscanCol

            overscanPolynom   = np.polyfit(x, y, 3)
            fittedOverscanCol = np.polyval(overscanPolynom, x)

        # Extend the overscan sampling along the x-axis
        overscanCol = fittedOverscanCol[:, np.newaxis]
        overscanArr = overscanCol.repeat(self.shape[1], 1)

        # Subtract the overscan shape from the array and store it
        correctedArr = self.data.copy() - overscanArr

        # Build an NDDataArray instance to store the corrected array
        outFullData = NDDataArray(
            correctedArr,
            uncertainty=self._BaseImage__fullData.uncertainty,
            unit=self._BaseImage__fullData.unit,
            wcs=self._BaseImage__fullData.wcs
        )

        # Store the corrected data in the hidden "fullData" attribute
        self._BaseImage__fullData = outFullData

        # Set the flag to indicate that the overscan was successfully removed.
        self.__overscanCorrected = True

    def process_image(self, bias=None, dark=None, flat=None):
        """
        Subtracts bias, removes dark current, and divides by the flat field. If
        any of the bias, dark, or flat images are not provided, then those steps
        of the processing will be skipped.

        Parameters
        ----------
        bias : astroimage.MasterBias
            An averaged bias frame

        dark : astroimage.MasterDark
            An average of bias-free dark frames with units of counts/sec to be
            scaled up by the exposure time of the current RawImage instance.

        flat : astroimage.MasterFlat
            An average of bias-free, mode-normalized flat frames

        Returns
        -------
        outImg : astroimage.ReducedImage (or subclass)
            The bias subtracted, and/or dark current subtracted, and/or flat
            field normalized image.
        """
        # Catch the type of image being processed.
        selfType = type(self)

        # By default, do NOTHING (set with these flags)
        subtractBias, subtractDark, divideFlat = False, False, False
        if selfType is RawBias:
            pass
        if selfType is RawDark:
            subtractBias = True
        if selfType is RawFlat:
            subtractBias = True
            subtractDark = True
        if selfType is RawScience:
            subtractBias = True
            subtractDark = True
            divideFlat   = True

        # Make a copy of the arr attribute for manipulating
        outArr = self.data.copy()

        # Apply each correction type in the proper order
        if bias is not None and subtractBias:
            if type(bias) is MasterBias:
                outArr   -= bias.data
                outUncert = bias.uncertainty
            else:
                raise TypeError('`bias` must be a MasterBias image')

        if dark is not None and subtractDark:
            if type(dark) is MasterDark:
                if dark.is_significant:
                    outArr   -= self.expTime*dark.data
                    outUncert = np.sqrt(
                        outUncert**2 +
                        (self.expTime*dark.uncertainty)**2
                    )
                else:
                    warnings.warn('Skipping insignificant dark current levels')
            else:
                raise TypeError('`dark` must be a MasterDark image')

        # TODO: Add uncertainty for RawScience images at this point
        #
        # 1) Compute poisson uncertainty.
        # 2) Include poisson uncertainty in new output uncertainty
        # 3) Proceed to divide by MasterFlat, which has its own uncertainty

        if flat is not None and divideFlat:
            if type(flat) is MasterFlat:
                A = outArr
                B = flat.data
                outArr    = A/B
                outUncert = np.abs(outArr)*np.sqrt(
                    (outUncert/A)**2 +
                    (flat.uncertainty/B)**2
                )
            else:
                raise TypeError('`flat` must be a MasterFlat image')

        # Copy the header for manipulation
        outHeader = self.header.copy()

        # Remove the Prescan and Overscan keywords from the header
        headerKeywordDictKeys = self.headerKeywordDict.keys()
        if 'PRESCAN' in headerKeywordDictKeys:
            prescanKey = headerKeywordDict['PRESCAN']
            try: del outHeader[prescanKey]
            except: pass

        if 'OVERSCAN' in headerKeywordDictKeys:
            overscanKey = headerKeywordDict['OVERSCAN']
            try: del outHeader[overscanKey]
            except: pass

        # Ensure the NAXIS keywords are correct
        outHeader['NAXIS1'] = outArr.shape[1]
        outHeader['NAXIS2'] = outArr.shape[0]

        # TODO: determine which TYPE of output should be written

        # Initalize a blank instance for returning to the user
        outImage = ReducedImage(
            outArr,
            uncertainty=outUncert,
            header=outHeader,
            properties={'unit': self.unit}
        )

        return outImage

    ##################################
    ### END OF OTHER METHODS       ###
    ##################################

##################################
### START OF SUBCLASSES        ###
##################################

class RawBias(RawImage):
    """
    A class for handling raw bias frames.

    Provides a method for computing the read-noise of the bais frames.
    """
    def __init__(self, *args, **kwargs):
        super(RawBias, self).__init__(*args, **kwargs)

        if self.obsType != 'BIAS':
            raise IOError('Cannot instantiate a RawBias with a {0} type image.'.format(
                self.obsType
            ))

    ##################################
    ### START OF PROPERTIES        ###
    ##################################

    @property
    def readNoise(self):
        """The estimated read noise of this bias frame"""
        if self.overscanCorrected:
            return np.std(self.data) * self.unit
        else:
            raise RuntimeError('Must apply overscan correction before estimating read noise')

    ##################################
    ### END OF PROPERTIES          ###
    ##################################

class RawDark(RawImage):
    """
    A class for handling raw dark frames.
    """
    def __init__(self, *args, **kwargs):
        super(RawDark, self).__init__(*args, **kwargs)

        if self.obsType != 'DARK':
            raise IOError('Cannot instantiate a RawDark with a {0} type image.'.format(
                self.obsType
            ))

class RawFlat(RawImage):
    """
    A class for handling raw flat frames.

    Provides a method for computing the mode of the flat frame.
    """
    def __init__(self, *args, **kwargs):
        super(RawFlat, self).__init__(*args, **kwargs)

        if self.obsType != 'FLAT':
            raise IOError('Cannot instantiate a RawFlat with a {0} type image.'.format(
                self.obsType
            ))

    @property
    @lru_cache()
    def mode(self):
        """An estimate of the statistical mode of this image"""
        # Compute the number of bins that will be needed to find mode
        numBins = np.int(np.ceil(0.1*(np.max(self.data) - np.min(self.data))))

        # Loop through larger and larger binning until find unique solution
        foundMode = False
        while not foundMode:
            # Generate a histogram of the flat field
            hist, flatBins = np.histogram(self.data.flatten(), numBins)

            # Locate the histogram maximum
            maxInds = (np.where(hist == np.max(hist)))[0]
            if maxInds.size == 1:
                # Grab the index of the maximum value and shrink
                maxInd = maxInds[0]
                foundMode = True
            else:
                # Shrink the NUMBER of bins to help find a unqiue maximum
                numBins *= 0.9

        # Estimate flatMode from histogram maximum
        flatMode = np.mean(flatBins[maxInd:maxInd+2])*self.unit

        return flatMode

class RawScience(RawImage):
    """
    A class for handling raw science frames.
    """
    def __init__(self, *args, **kwargs):
        super(RawScience, self).__init__(*args, **kwargs)

        if self.obsType != 'OBJECT':
            raise IOError('Cannot instantiate a RawScience with a {0} type image.'.format(
                self.obsType
            ))

##################################
### END OF SUBCLASSES          ###
##################################
