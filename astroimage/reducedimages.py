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
from astropy.io import fits

# AstroImage imports
from .baseimage import BaseImage, ClassProperty

# Define which functions, classes, objects, etc... will be imported via the command
# >>> from .reducedimage import *
__all__ = ['ReducedImage', 'MasterBias', 'MasterDark', 'MasterFlat']

class ReducedImage(BaseImage):
    """
    The base class for reduced calibration and science data.

    Defines additional mathematical operations (e.g., trig and log functions)

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
    header          The header info for the associated fits file.
    height          The height of the image, in pixels.
    image           The AxesImage storing the plottted data (if plotted)
    instrument      The instrument from which the image was obtained
    ra              The right ascension of the observation in the format
                    HH:MM:SS.SS
    shape           Dimensions of the image as a tuple, returned as height, then
                    width, in pixels, in keeping with the behavior of the
                    `numpy.ndarray` size attribute.
    uncertainty     The array of uncertainties associated with `data`
    units           The units of the numpy 2D array stored in `data`
    width           The width of the image, in pixels

    Class Methods
    -------------
    set_headerKeywordDict
    read

    Methods
    -------
    set_arr
    set_uncertainty
    set_header
    copy
    write
    rebin
    show

    Examples
    --------
    Read in calibrated files
    >>> from astroimage import ReducedScience
    >>> img1 = ReducedScience.read('img1.fits')
    >>> img2 = ReducedScience.read('img2.fits')

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
        'uncertainty' # This property and below are for the ReducedImage class
    ])

    ##################################
    ### END OF CLASS VARIABLES     ###
    ##################################

    ##################################
    ### END OF CLASS METHODS       ###
    ##################################

    @ClassProperty
    @classmethod
    def properties(cls):
        return cls.__properties

    ##################################
    ### END OF CLASS METHODS       ###
    ##################################

    def __init__(self, *args, **kwargs):
        """
        Constructs a `ReducedImage` instance from provided arguments.

        More properly implemented by subclasses
        Parameters
        ----------
        data : `numpy.ndarray`, optional
            The array of values to be stored for this image

        uncertainty : `numpy.ndarray`, optional
            The uncertainty of the values to be stored for this image

        header : `astropy.io.fits.header.Header`, optional
            The header to be associated with the `arr` attribute

        properties : `dict`, optional
            A dictionary of properties to be set for this image
            (e.g. {'unit': u.adu, 'ra': 132.323, 'dec': 32.987})

        Returns
        -------
        outImg : `ReducedImage` (or subclass)
            A new instance containing the supplied data, header, and
            properties
        """
        # Start by instantiating the basic BaseImage type information
        super(ReducedImage, self).__init__(*args, **kwargs)

    ##################################
    ### START OF PROPERTIES        ###
    ##################################

    @property
    def uncertainty(self):
        """The uncertainties associated with the `data` values"""
        if self.has_uncertainty:
            return self._BaseImage__fullData.uncertainty.array
        else:
            return None

    @property
    def has_uncertainty(self):
        """Boolean flag if the `uncertainty` property exists"""
        return (self._BaseImage__fullData.uncertainty is not None)

    ##################################
    ### END OF PROPERTIES        ###
    ##################################

    ##################################
    ### START OF MAGIC METHODS     ###
    ##################################

    def __getitem__(self, key):
        """
        Implements the slice getting method.

        Parameters
        ----------
        key: slice
            The start, stop[, step] slice of the pixel locations to be returned

        Returns
        -------
        outImg: `ReducedImage` (or subclass)
            A sliced copy of the original image

        Examples
        --------
        This method can be used to crop and rebin data. So that minimal data is
        lost, the optional step element of the `key` slice(s) are interpreted
        as a rebinning factor for the flux conservative `frebin` method.

        >>> from astroimage import ReducedImage
        >>> img1 = ReducedImage(np.arange(100).reshape((10, 10)))
        >>> img1.shape
        (10, 10)
        >>> img2 = img1[1:9:2, 1:9:2]
        >>> img2.shape
        (4, 4)

        In this instance, the default *average* rebinning method is used, but
        that can be corrected through a simple multiplicative factor.
        """
        raise NotImplementedError


        # TODO: Catch if the key is just a slice (i.e. not along both axes)
        if type(key) is slice:
            key1 = (key,)
        else:
            key1 = key

        # TODO: Compute the rebinning factors along each axis
        binnings = [k.step if k.step is not None else 1 for k in key]

        # TODO: Recompute the crop boundaries using the rebinning factors


        # TODO: Crop the array

        # TODO: Rebin the array if necessary

        # Copy the image
        outImg = self.copy()

        # # Grab the output portion of the array
        # outArr = self.data[key]
        # outImg.set_arr(outArr)
        #
        # # Check if there is an uncertainty, and grab the output portion
        # if self.has_sigma:
        #     outSig = self.sigma[key]
        #     outImg.set_sigma(outSig)
        #
        # # Update the header
        # outImg.header['NAXIS1'] = outArr.shape[1]
        # outImg.header['NAXIS2'] = outArr.shape[0]

        return outImg

    def __setitem__(self, key, value):
        """
        Implements the slice setting method.

        Parameters
        ----------
        key: slice
            The start, stop[, step] slice of the pixel locations to set

        value : int, float, or array_like
            The values to place into the specified slice of the stored array

        Returns
        -------
        out: None
        """

        raise NotImplementedError

        # TODO: finish this implementation ??? OR Just get rid of it!


    ##################################
    ### END OF MAGIC METHODS     ###
    ##################################

























    ##################################
    ### START OF CUSTOM SETTERS    ###
    ##################################

    def set_uncertainty(self, uncert):
        """
        Used to replace the private `sigma` attribute.

        Parameters
        ----------
        uncert : numpy.ndarray
            An array containing the array to be placed in the private `uncertainty`
            attribute

        Returns
        -------
        out : None
        """

        # Test if arr is a numpy array
        if not isinstance(uncertArr, np.array):
            raise TypeError('`uncert` must be an instance of numpy.ndarray')

        # Test if the replacement array matches the previous array's shape
        if self.shape != uncertArr.shape:
            raise ValueError('`uncert` is not the expected shape')

        import pdb; pdb.set_trace()
        # TODO: set the uncertainty

        return None

    ##################################
    ### END OF CUSTOM SETTERS      ###
    ##################################

    ##################################
    ### START OF OTHER METHODS     ###
    ##################################

    def _build_HDUs(self):
        # Invoke the parent method to build the basic HDU
        HDUs = super(ReducedImage, self)._build_HDUs()

        if self.uncertainty is not None:
             # Bulid a secondary HDU
            sigmaHDU = fits.ImageHDU(data = self.uncertainty,
                                     name = 'UNCERTAINTY',
                                     do_not_scale_image_data=True)
            HDUs.append(sigmaHDU)

        return HDUs

    def divide_by_expTime(self):
        """Divides the image by its own exposure time and sets expTime to 1"""
        # Divide by the exposure time
        outImg = self/self.expTime

        # Modify the expTime value
        outImg._BaseImage__expTime = 1.0

        # Make sure the header is updated, too
        outImg._properties_to_header()

        return outImg

    # def rebin(self, nx1, ny1, copy=False, total=False):
    #     """
    #     Rebins the image using sigma attribute to produce a weighted average.
    #     The new image shape must be integer multiples of fractions of the
    #     original shape. Default behavior is to use inverse variance weighting
    #     for the average if a "sigma" attribute is present. Otherwise, simply
    #     do straight averaging or summing.
    #
    #     Parameters
    #     ----------
    #
    #     copy : bool, optional, default: False
    #         If set to true, then returns a copy of the image with a rebinned
    #         array. Otherwise, the image will be rebinned in place.
    #
    #     total : bool, optional, default: False
    #         If set to true, then returned array is total of the
    #         binned pixels rather than the average.
    #
    #     Returns
    #     -------
    #     outImg : `BaseImage` (or subclass) or None
    #         If copy was set to True, then a rebinned copy of the original image
    #         is returned. Otherwise None is returned and the original image is
    #         rebinned in place.
    #     """
    #
    #     # TODO -- rewrite all baseclass methods so that they ONLY reference the
    #     # attributes defined for the BaseImage class.
    #
    #     # TODO -- write the subclass methods to simply invoke the analogous
    #     # BaseImage method and then tack on additional behaviour (e.g. updating
    #     # the wcs in an ReducedScience instance or reading in the image uncertainty
    #     # for a ReducedImage instance)
    #
    #     # Grab the shape of the initial array
    #     ny, nx = self.shape
    #
    #     # Test for improper result shape
    #     goodX = ((nx % nx1) == 0) or ((nx1 % nx) == 0)
    #     goodY = ((ny % ny1) == 0) or ((ny1 % ny) == 0)
    #     if not (goodX and goodY):
    #         raise ValueError('Result dimensions must be integer factor of original dimensions')
    #
    #     # First test for the trivial case
    #     if (nx == nx1) and (ny == ny1):
    #         if copy:
    #             return self.copy()
    #         else:
    #             return None
    #
    #     # Compute the pixel ratios of upsampling and down sampling
    #     xratio, yratio = np.float(nx1)/np.float(nx), np.float(ny1)/np.float(ny)
    #     pixRatio       = np.float(xratio*yratio)
    #     aspect         = yratio/xratio         #Measures change in aspect ratio.
    #
    #     if ((nx % nx1) == 0) and ((ny % ny1) == 0):
    #         # Handle integer downsampling
    #         # Get the new shape for the array and compute the rebinning shape
    #         sh = (ny1, ny//ny1,
    #               nx1, nx//nx1)
    #
    #         # Build the appropriate weights for the averaging procedure
    #         if hasattr(self, 'sigma'):
    #             # Catch the zeros uncertainty points and null them out.
    #             tmpSig = self.sigma.copy()
    #             zeroInds = np.where(self.sigma == 0)
    #             if len(zeroInds[0]) > 0:
    #                 tmpSig[zeroInds] = 1.0
    #
    #             # Now actually compute the weights
    #             wts    = tmpSig**(-2)
    #
    #             # Finally replace "zero-uncertainty" values with zero weight.
    #             if len(zeroInds[0]) > 0:
    #                 wts[zeroInds] = 0.0
    #
    #         else:
    #             wts = np.ones_like(self.data)
    #
    #         # Build the weighted array
    #         tmpArr = wts*self.data
    #
    #         # Perform the actual rebinning
    #         # rebinWts1 = wts.reshape(sh).mean(-1).mean(1)
    #         rebinWts = wts.reshape(sh).sum(-1).sum(1)
    #
    #         # Catch division by zero
    #         zeroInds   = np.where(rebinWts == 0)
    #         noZeroInds = np.where(
    #             np.logical_and(
    #             (rebinWts != 0),
    #             np.isfinite(rebinWts)))
    #
    #         # Computed weighted rebinning
    #         rebinArr = (tmpArr.reshape(sh).sum(-1).sum(1))
    #         rebinArr[noZeroInds] /= rebinWts[noZeroInds]
    #
    #         # Compute uncertainyt in weighted rebinning
    #         rebinSig = np.zeros(rebinArr.shape) + np.NaN
    #         rebinSig[noZeroInds] = np.sqrt(1.0/rebinWts[noZeroInds])
    #
    #         # Check if total flux conservation was requested
    #         if total:
    #             # Re-normalize by pixel area ratio
    #             rebinArr /= pixRatio
    #
    #             # Apply the same re-normalizing to the sigma array
    #             if hasattr(self, 'sigma'):
    #                 rebinSig /= pixRatio
    #
    #     elif ((nx1 % nx) == 0) and ((ny1 % ny) == 0):
    #         # Handle integer upsampling
    #         rebinArr = np.kron(self.data, np.ones((ny1//ny, nx1//nx)))
    #         if hasattr(self, 'sigma'):
    #             rebinSig  = np.kron(self.sigma, np.ones((ny1//ny, nx1//nx)))
    #
    #         # Check if total flux conservation was requested
    #         if total:
    #             # Re-normalize by pixel area ratio
    #             rebinArr /= pixRatio
    #
    #             if hasattr(self, 'sigma'):
    #                 rebinSig /= pixRatio
    #
    #     # Check if there is a header needing modification
    #     if hasattr(self, 'header'):
    #         outHead = self.header.copy()
    #
    #         # Update the NAXIS values
    #         outHead['NAXIS1'] = nx1
    #         outHead['NAXIS2'] = ny1
    #
    #         # Update the CRPIX values
    #         outHead['CRPIX1'] = (self.header['CRPIX1'] + 0.5)*xratio + 0.5
    #         outHead['CRPIX2'] = (self.header['CRPIX2'] + 0.5)*yratio + 0.5
    #         if self.wcs.wcs.has_cd():
    #             # Attempt to use CD matrix corrections, first
    #             # Apply updates to CD valus
    #             thisCD = self.wcs.wcs.cd
    #             # TODO set CDELT value properly in the "astrometry" step
    #             outHead['CD1_1'] = thisCD[0,0]/xratio
    #             outHead['CD1_2'] = thisCD[0,1]/yratio
    #             outHead['CD2_1'] = thisCD[1,0]/xratio
    #             outHead['CD2_2'] = thisCD[1,1]/yratio
    #         elif self.wcs.wcs.has_pc():
    #             # Apply updates to CDELT valus
    #             outHead['CDELT1'] = outHead['CDELT1']/xratio
    #             outHead['CDELT2'] = outHead['CDELT2']/yratio
    #
    #             # Adjust the PC matrix if non-equal plate scales.
    #             # See equation 187 in Calabretta & Greisen (2002)
    #             if aspect != 1.0:
    #                 outHead['PC1_1'] = outHead['PC1_1']
    #                 outHead['PC2_2'] = outHead['PC2_2']
    #                 outHead['PC1_2'] = outHead['PC1_2']/aspect
    #                 outHead['PC2_1'] = outHead['PC2_1']*aspect
    #
    #         # Adjust BZERO and BSCALE for new pixel size, unless these values
    #         # are used to define unsigned integer data types.
    #         # TODO handle special cases of unsigned integers, where BSCALE may
    #         # be used to define integer data types.
    #         if not total:
    #             if 'BSCALE' in self.header.keys():
    #                 bscale = self.header['BSCALE']
    #                 # If BSCALE has been set to something reasonable, then adjust it
    #                 if (bscale != 0) and (bscale != 1):
    #                     outHead['BSCALE'] = (bscale/pixRatio,
    #                         'Calibration Factor')
    #
    #             if 'BZERO' in self.header.keys():
    #                 bzero  = self.header['BZERO']
    #                 # If BZERO has been set to something reasonable, then adjust it
    #                 if (bzero != 0):
    #                     outHead['BZERO'] = (bzero/pixRatio,
    #                         'Additive Constant for Calibration')
    #
    #     else:
    #         # If no header exists, then buil a basic one
    #         keywords = ['NAXIS2', 'NAXIS1']
    #         values   = (ny1, nx1)
    #         headDict = dict(zip(keywords, values))
    #         outHead  = fits.Header(headDict)
    #
    #     # Reread the WCS from the output header
    #     outWCS = WCS(outHead)
    #
    #     if copy:
    #         # If a copy was requested, then return a copy of the original image
    #         # with a newly rebinned array
    #         outImg         = self.copy()
    #         outImg.arr     = rebinArr
    #
    #         # Update the uncertainty attribute
    #         # This may be a newly computed "uncertainty of the mean"
    #         outImg.sigma = rebinSig
    #
    #         # Update the header if it exists
    #         if hasattr(self, 'header'):
    #             outImg.header  = outHead
    #
    #         # Update the binning attribute to match the new array
    #         outImg.binning = (outImg.binning[0]/xratio,
    #                           outImg.binning[1]/yratio)
    #
    #         # Update the wcs attribute to match the new header data
    #         if outWCS.has_celestial: outImg.wcs = outWCS
    #
    #         # Return the updated image object
    #         return outImg
    #     else:
    #         # Otherwise place the rebinned array directly into the Image object
    #         self.data     = rebinArr
    #
    #         if hasattr(self, 'sigma'):
    #             self.sigma   = rebinSig
    #
    #         if hasattr(self, 'header'):
    #             self.header  = outHead
    #
    #         # Update the binning attribute to match the new array
    #         self.binning = (self.binning[0]/xratio,
    #                         self.binning[1]/yratio)
    #
    #         # Update the wcs attribute to match the new header data
    #         if outWCS.has_celestial: self.wcs = outWCS

    ##################################
    ### END OF OTHER METHODS       ###
    ##################################

##################################
### START OF SUBCLASSES        ###
##################################

class MasterBias(ReducedImage):
    """A class for reading in reduced master bias frames."""

    def __init__(self, *args, **kwargs):
        super(MasterBias, self).__init__(*args, **kwargs)

        if self.obsType != 'BIAS':
            raise IOError('Cannot instantiate a RawBias with a {0} type image.'.format(
                self.obsType
            ))

class MasterDark(ReducedImage):
    """A class for reading in reduced master dark frames."""

    def __init__(self, *args, **kwargs):
        super(MasterDark, self).__init__(*args, **kwargs)

        if self.obsType != 'DARK':
            raise IOError('Cannot instantiate a RawDark with a {0} type image.'.format(
                self.obsType
            ))

    ##################################
    ### START OF PROPERTIES        ###
    ##################################

    @property
    @lru_cache()
    def is_significant(self):
        """Boolean flag if the dark current is greater than 2x(read noise)"""
        return (np.median(self.data)/np.std(self.data)) > 2.0

    ##################################
    ### END OF PROPERTIES        ###
    ##################################

class MasterFlat(ReducedImage):
    """A class for reading in reduced master flat frames."""

    def __init__(self, *args, **kwargs):
        super(MasterFlat, self).__init__(*args, **kwargs)

        if self.obsType != 'FLAT':
            raise IOError('Cannot instantiate a RawFlat with a {0} type image.'.format(
                self.obsType
            ))

    ##################################
    ### START OF PROPERTIES        ###
    ##################################

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

    ##################################
    ### END OF PROPERTIES          ###
    ##################################


##################################
### END OF SUBCLASSES          ###
##################################
