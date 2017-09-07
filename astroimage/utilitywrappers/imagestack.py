# Core imports
import copy
import warnings
import psutil
from functools import lru_cache

# Scipy imports
import numpy as np
from scipy import ndimage, signal

# Astropy imports
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from astropy.nddata import NDDataArray, StdDevUncertainty
from photutils import make_source_mask, data_properties

# AstroImage imports
from ..baseimage import BaseImage
from ..raw import RawBias, RawDark, RawFlat, RawScience
from ..reduced import MasterBias, MasterDark, MasterFlat, ReducedScience
from .astrometrysolver import AstrometrySolver
from .inpainter import Inpainter

# Define which functions, classes, objects, etc... will be imported via the command
# >>> from imagestack import *
__all__ = ['ImageStack']

class ImagePairOffsetGetter(object):
    """
    A class for computing the offsets between two images.

    This is essentially a helper class for the ImageStack alignment methods.
    This class assumes that there is no significant rotation between the two
    input images.

    Properties
    ----------
    image1         A BaseImage (or subclass) instance
    image2         A BaseImage (or subclass) instance

    Methods
    -------
    get_cross_correlation_integer_pixel_offset    Computes the offset between
                                                  two images with integer pixel
                                                  accuracy
    get_cross_correlation_subpixel_offset         Computes the offset between
                                                  two images with subpixel
                                                  accuracy
    """

    def __init__(self, image1, image2):
        """
        Constructs the ImagePairOffsetGetter from two supplied images.
        """
        if not (issubclass(type(image1), ReducedScience)
            and(issubclass(type(image2), ReducedScience))):
            raise TypeError('Both images must be `ReducedScience` instances for proper alignment')

        # Store two copies of the image data arrays... the rest doesn't matter!
        self.image1 = image1.copy()
        self.image2 = image2.copy()

    ##################################
    ### START OF OTHER METHODS     ###
    ##################################

    def _replace_negatives_and_nans_with_medians(self):
        """
        Replaces negatives and nans with non-problematic values.

        Uses a median-filter to estimate the expected values at the location of
        negative/nan pixels.
        """
        # Loop through each of the arrays and perform the cleanup
        arrayList = [self.image1.data, self.image2.data]
        for array in arrayList:
            # Find the negative pixels
            negPix = np.nan_to_num(array) < 0

            # If there are some negative pixels, then replace them with the
            # median value of the WHOLE image
            if np.sum(negPix.astype(int)) > 0:
                # Find the indices of the bad and good pixels
                badInds  = np.where(negPix)
                goodInds = np.where(np.logical_not(negPix))

                # Replace the bad pixels with the median of the good pixels
                array[badInds] = np.median(np.nan_to_num(array[goodInds]))

            # Find the Nan pixels
            nanPix = np.logical_not(np.isfinite(array))

            # If there are some NaN pixels, then replace them with the local
            # median value.
            if np.sum(nanPix.astype(int)) > 0:
                # Find the indices of the bad and good pixels
                badInds  = np.where(nanPix)
                goodInds = np.where(np.logical_not(nanPix))

                # Replace the bad pixels with the median of the good pixels
                array[badInds] = np.median(np.nan_to_num(array[goodInds]))

                # Compute the median filtered image
                medianImage =  ndimage.median_filter(array, size=(9,9))

                # Replace the bad pixels with their local median
                array[badInds] = medianImage[badInds]

        # Return the fixed arrays to the user
        return tuple(arrayList)

    @staticmethod
    def _fix_bad_correlation_image_pixels(corrImage):
        """Repairs any deviant pixels in the cross-correlation image"""
        # Do a little post-processing to block out bad points in corrImage
        # Copy the input for manipulation
        outCorrImage = corrImage.copy()

        # First filter with the median
        medCorr = ndimage.median_filter(corrImage, size=(9,9))

        # Compute sigma_clipped_stats of the correlation image
        mean, median, stddev = sigma_clipped_stats(corrImage)

        # Then check for significant deviations from median.
        deviations = (np.abs(corrImage - medCorr) > 2.0*stddev)

        # Count the number of masked neighbors for each pixel
        neighborCount = np.zeros_like(corrImage, dtype=np.int16)
        for dx1 in range(-1,2,1):
            for dy1 in range(-1,2,1):
                    neighborCount += np.roll(np.roll(deviations, dy1, axis=0),
                        dx1, axis=1).astype(np.int16)

        # Find isolated deviant pixels (these are no good!)
        deviations = np.logical_and(deviations, neighborCount <= 4)

        # If some deviating pixels were found, then replace them with their
        # local median
        if np.sum(deviations > 0):
            badInds = np.where(deviations)
            outCorrImage[badInds] = medCorr[badInds]

        return outCorrImage

    @staticmethod
    def _extract_integer_offset_from_correlation_image(corrImage):
        """
        Extracts the image offset values from the cross correlation image

        Parameters
        ----------
        corrImage : numpy.ndarray
            A clean (defect free) version of the cross correlation image

        Returns
        -------
        dx, dy : int
            The image ofset values based on the cross correlation image
        """
        # Check for the maximum of the cross-correlation image function
        correlationPeak = np.unravel_index(corrImage.argmax(), corrImage.shape)
        dy, dx = np.array(correlationPeak) - np.array(corrImage.shape)//2

        return int(dx), int(dy)

    def get_wcs_integer_pixel_offset(self):
        """
        Computes the offset between image1 and image2 using wcs alignment.

        Provides integer pixel accuracy.

        Returns
        -------
        dx, dy : int
            The offset of self.image1 with respect to self.image2
        """
        # Grab the WCS of the first image
        refWCS = self.image1.wcs

        # Estimate the central location for this image
        refX, refY = self.image1.shape[1]//2, self.image1.shape[0]//2

        # Convert pixels to sky coordinates
        refRA, refDec = refWCS.all_pix2world(refX, refY, 0, ra_dec_order=True)

        # Compute the pixel location of the reference RA and Dec
        dx, dy = self.image2.wcs.all_world2pix(refRA, refDec, 0)

        # Convert these relativeoffsets into integer values
        dx, dy = int(dx - refX), int(dy - refY)

        # Return the image offsets
        return (dx, dy)

    @lru_cache()
    def get_cross_correlation_integer_pixel_offset(self):
        """
        Computes the offset between image1 and image2 using cross-correlation.

        Provides integer pixel accuracy.

        Returns
        -------
        dx, dy : int
            The offset of self.image1 with respect to self.image2
        """
        # Replace any suprious values with local median values
        array1, array2 = self._replace_negatives_and_nans_with_medians()

        # Do an array flipped convolution, which is a correlation.
        corrImage = signal.fftconvolve(
            array2,
            array1[::-1, ::-1],
            mode='same'
        )

        # Fix any suprious pixel values
        corrImage = self._fix_bad_correlation_image_pixels(corrImage)

        # Extract the integer pixel offesets from this correlation image
        dx, dy = self._extract_integer_offset_from_correlation_image(corrImage)

        return dx, dy

    @staticmethod
    def _parse_star_cutouts(starCutouts1, starCutouts2):
        """
        Decides which cutouts to use for sub-pixel alignment.

        The provided star cutouts will be quality checked. Those cutouts which
        do not meet the minimum quality criteria will be tossed from the stacks.

        Parameters
        ----------
        starCutouts1, starCutouts2 : array_like
            A stack of star cutouts, each center on on a star

        Returns
        -------
        outStarCutouts1, outStarCutouts2 : array_like
            A quality-cut stack of star cutouts, each center on on a star.
        """
        # Start by parsing the properties of ALL the cutouts
        cutoutCorrelationCoeffs1 = []
        cutoutCorrelationCoeffs2 = []
        cutoutElongations1       = []
        cutoutElongations2       = []
        normalizedStarCutouts1   = []
        normalizedStarCutouts2   = []
        averageCutoutFlux        = []
        for starCutout1, starCutout2 in zip(starCutouts1, starCutouts2):
            # Measure the properties of this cutout
            cutoutProperties1 = data_properties(starCutout1)
            cutoutProperties2 = data_properties(starCutout2)

            # Grab the elongation property
            cutoutElongations1.append(cutoutProperties1.elongation)
            cutoutElongations2.append(cutoutProperties2.elongation)

            # Grab the correlation coefficient property
            correlationCoeff1 = (
                cutoutProperties1.covar_sigxy /
                (
                    cutoutProperties1.semimajor_axis_sigma *
                    cutoutProperties1.semiminor_axis_sigma
                )
            )
            correlationCoeff2 = (
                cutoutProperties2.covar_sigxy /
                (
                    cutoutProperties2.semimajor_axis_sigma *
                    cutoutProperties2.semiminor_axis_sigma
                )
            )

            # Store the correlation coefficients in lists
            cutoutCorrelationCoeffs1.append(correlationCoeff1)
            cutoutCorrelationCoeffs2.append(correlationCoeff2)

            # Compute the total flux of each star
            cutoutFlux1 = starCutout1.sum()
            cutoutFlux2 = starCutout2.sum()

            # Normalize the cutouts to have a total of one
            normalizedStarCutouts1.append(starCutout1/cutoutFlux1)
            normalizedStarCutouts2.append(starCutout2/cutoutFlux2)

            # Compute an average flux for this star and store it
            starCutoutFlux = 0.5*(cutoutFlux1 + cutoutFlux2)
            averageCutoutFlux.append(starCutoutFlux)

        # Convert these to arrays
        cutoutCorrelationCoeffs1 = np.array(cutoutCorrelationCoeffs1)
        cutoutCorrelationCoeffs2 = np.array(cutoutCorrelationCoeffs2)
        cutoutElongations1       = np.array(cutoutElongations1)
        cutoutElongations2       = np.array(cutoutElongations2)
        normalizedStarCutouts1   = np.array(normalizedStarCutouts1)
        normalizedStarCutouts2   = np.array(normalizedStarCutouts2)
        averageCutoutFlux        = np.array(averageCutoutFlux)

        # Sort the cutouts from brightest to dimmest
        # Grab the sorting index ordering
        sortInds = averageCutoutFlux.argsort()
        sortInds = sortInds[::-1]

        # Apply the sorting array.
        cutoutCorrelationCoeffs1 = cutoutCorrelationCoeffs1[sortInds]
        cutoutCorrelationCoeffs2 = cutoutCorrelationCoeffs2[sortInds]
        cutoutElongations1       = cutoutElongations1[sortInds]
        cutoutElongations2       = cutoutElongations2[sortInds]
        normalizedStarCutouts1   = normalizedStarCutouts1[sortInds]
        normalizedStarCutouts2   = normalizedStarCutouts2[sortInds]
        averageCutoutFlux        = averageCutoutFlux[sortInds]

        # Find the cutouts with good correlation coefficients
        goodCorrelationCoeffs = np.logical_and(
            np.abs(cutoutCorrelationCoeffs1 - np.nanmedian(cutoutCorrelationCoeffs1)) < 0.15,
            np.abs(cutoutCorrelationCoeffs2 - np.nanmedian(cutoutCorrelationCoeffs2)) < 0.15
        )

        # Find the cutouts with good elogation values
        goodElongations = np.logical_and(
            cutoutElongations1 < 1.4,
            cutoutElongations2 < 1.4
        )

        # Find the cutouts with good everything....
        goodCutouts = np.logical_and(
            goodCorrelationCoeffs,
            goodElongations
        )

        # Cull the cutouts to only include the good cutouts
        goodCutoutInds = np.where(goodCutouts)

        return (normalizedStarCutouts1[goodCutoutInds],
            normalizedStarCutouts2[goodCutoutInds])

    @staticmethod
    def _build_star_cutout_mosaic(starCutouts):
        """
        Constructs a mosaic of star cutouts.

        Parameters
        ----------
        starCutouts : array_like
            A list of star cutouts, each centered on a star

        Returns
        -------
        starCutoutMosaic : numpy.ndarray
            An array containing the star cutout each of the brightest stars
        """
        # Make sure starCutouts can be handled properly
        try:
            starCutouts = np.array(starCutouts)
        except:
            raise TypeError('`starCutouts` must be an array-like object')

        if starCutouts.ndim != 3:
            raise ValueError('`starCutouts` must be a (numbor of stars X cutout size x cutout size) array')


        # Get the number and shape of the remaining star cutouts
        numberOfStars, ny, nx = starCutouts.shape

        # Cull the list to the brightest square number of stars
        if numberOfStars >= 25:
            keepStarCount = 25
        elif numberOfStars >= 16:
            keepStarCount = 16
        elif numberOfStars >= 9:
            keepStarCount = 9
        elif numberOfStars >= 4:
            keepStarCount = 4
        else:
            raise RuntimeError('Fewer than 9 stars found: cannot build star cutout mosaic')

        # Chop out the sections around each star, and build a mosaic of cutouts
        numZoneSide  = np.int(np.round(np.sqrt(keepStarCount)))
        cutoutMosaic = np.zeros((numZoneSide*ny, numZoneSide*nx))

        # Loop through each star to be placed in the mosaic
        for iStar, starCutout in enumerate(starCutouts[0:keepStarCount]):
            # Compute the zone for this star
            yZone, xZone = np.unravel_index(iStar, (numZoneSide, numZoneSide))

            # Establish the pasting boundaries
            btPaste = np.int(np.round(ny*yZone))
            tpPaste = np.int(np.round(ny*(yZone + 1)))
            lfPaste = np.int(np.round(nx*xZone))
            rtPaste = np.int(np.round(nx*(xZone + 1)))

            # Paste the cutout into the star mosaic
            cutoutMosaic[btPaste:tpPaste, lfPaste:rtPaste] = starCutout

        return cutoutMosaic

    @staticmethod
    def _extract_subpixel_offset_from_correlation_image(corrImage):
        """
        Extracts the subpixel offset from the cross-correlation image

        Parameters
        ----------
        corrImage : numpy.ndarray
            The cross-correlation image of the starCutoutMosaic images

        Returns
        -------
        dx, dy : float
            The subpixel correction to be added to the integer pixel offset of
            the two images
        """

        # Define a plane fitting function for use within this method only
        def planeFit(points):
            """
            p, n = planeFit(points)

            Given an array, points, of shape (d,...)
            representing points in d-dimensional space,
            fit an d-dimensional plane to the points.
            Return a point, p, on the plane (the point-cloud centroid),
            and the normal, n.
            """

            points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
            assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
            ctr = points.mean(axis=1)
            x = points - ctr[:,np.newaxis]
            M = np.dot(x, x.T) # Could also use np.cov(x) here.

            return ctr, np.linalg.svd(M)[0][:,-1]

        # Check for the maximum of the cross-correlation function
        yPeak, xPeak = np.unravel_index(corrImage.argmax(), corrImage.shape)

        # Compute the corners of the central region to analyze
        peakSz = 6
        btCorr = yPeak - peakSz
        tpCorr = btCorr + 2*peakSz + 1
        lfCorr = xPeak - peakSz
        rtCorr = lfCorr + 2*peakSz + 1

        # Chop out the central region
        corrImagePeak = corrImage[btCorr:tpCorr, lfCorr:rtCorr]

        # Get the gradient of the cross-correlation function
        Gx = ndimage.sobel(corrImagePeak, axis=1)
        Gy = ndimage.sobel(corrImagePeak, axis=0)

        # Grab the index of the peak
        yPeak, xPeak = np.unravel_index(
            corrImagePeak.argmax(),
            corrImagePeak.shape
        )

        # Chop out the central zone and grab the minimum of the gradient
        cenSz = 3
        bt    = yPeak - cenSz//2
        tp    = bt + cenSz
        lf    = xPeak - cenSz//2
        rt    = lf + cenSz

        # Grab the region near the minima
        yy, xx   = np.mgrid[bt:tp, lf:rt]
        Gx_plane = Gx[bt:tp, lf:rt]
        Gy_plane = Gy[bt:tp, lf:rt]

        # Fit planes to the x and y gradients...Gx
        px_init = models.Polynomial2D(degree=1)
        py_init = models.Polynomial2D(degree=1)
        fit_p   = fitting.LinearLSQFitter()
        px      = fit_p(px_init, xx, yy, Gx_plane)
        py      = fit_p(py_init, xx, yy, Gy_plane)

        # TODO: speed this up by getting the plane solutions from the
        # planeFit(points) function.

        # Solve these equations using NUMPY
        # 0 = px.c0_0 + px.c1_0*xx_plane + px.c0_1*yy_plane
        # 0 = py.c0_0 + py.c1_0*xx_plane + py.c0_1*yy_plane
        #
        # This can be reduced to Ax = b, where
        #
        A = np.matrix([[px.c1_0.value, px.c0_1.value],
                       [py.c1_0.value, py.c0_1.value]])
        b = np.matrix([[-px.c0_0.value],
                       [-py.c0_0.value]])

        # Now we can use the build in numpy linear algebra solver
        x_soln = np.linalg.solve(A, b)

        # Extract the shape of the corrImage and compute final relative offset
        ny, nx = corrImage.shape

        # Finally convert back into an absolute image offset
        dx1 = lfCorr + (x_soln.item(0) - (ny)//2)
        dy1 = btCorr + (x_soln.item(1) - (nx)//2)

        return dx1, dy1

    def get_cross_correlation_subpixel_offset(self, satLimit=16e3,
        cutoutSize=21):
        """
        Computes the offset between image1 and image2 using cross-correlation.

        Provides subpixel accuracy.

        Parameters
        ----------
        satLimit : int or float, optional, default: 16e3
            Sources which contain any pixels with more than this number of
            counts will not be used to perform cross-correlation alignment.

        cutoutSize : int, optional, default: 21
            The size of the cutout array to extract for matching PSFs through
            cross-correlation. This will also set a limit for the nearest
            neighbors allowed at sqrt(2)*cutoutSize.

        Returns
        -------
        dx, dy : float
            The precise offset of self.image1 with respect to self.image2
        """
        # TODO: Test if rough pixel-level alignment is required
        pass

        # # Test if a quick WCS integer pixel alignment is possible.
        # if self.image1.has_wcs and self.image2.has_wcs:
        #     # Compute the integer pixel offsets using WCS
        #     dx, dy = self.get_wcs_integer_pixel_offset()
        # else:
        #     # Compute the integer pixel offsets using cross-correlation
        #     dx, dy = self.get_cross_correlation_integer_pixel_offset()
        #
        # # Shift image2 array to approximately match image1
        # shiftedImage2 = self.image2.shift(-dx, -dy)

        # Compute a combined image and extract stars from that combined image
        combinedImage  = 0.5*(self.image1 + self.image2)

        xStars, yStars = combinedImage.get_sources(
            satLimit = satLimit,
            crowdLimit = np.sqrt(2)*cutoutSize,
            edgeLimit = cutoutSize + 1
        )

        # Grab the list of star cutouts from image one
        starCutouts1 = self.image1.extract_star_cutouts(xStars, yStars,
            cutoutSize = cutoutSize)

        # Grab the list of star cutouts from shifted image two
        starCutouts2 = self.image2.extract_star_cutouts(xStars, yStars,
            cutoutSize = cutoutSize)

        # Cull any bad cutouts from the cutout list
        starCutouts1, starCutouts2 = self._parse_star_cutouts(
            starCutouts1,
            starCutouts2
        )

        # Build the square mosaics of cutouts
        cutoutMosaic1 = self._build_star_cutout_mosaic(starCutouts1)
        cutoutMosaic2 = self._build_star_cutout_mosaic(starCutouts2)

        #
        # TODO: remove this code block if possible
        #
        # Construct a NEW ImagePair instance from these two mosaics
        mosaicPair = ImagePairOffsetGetter(
            ReducedScience(cutoutMosaic1),
            ReducedScience(cutoutMosaic2)
        )

        # Replace any suprious values with local median values
        array1, array2 = mosaicPair._replace_negatives_and_nans_with_medians()

        # Do an array flipped convolution, which is a correlation.
        corrImage = signal.fftconvolve(
            array2,
            array1[::-1, ::-1],
            mode='same'
        )

        # Fix any suprious pixel values
        corrImage = ImagePairOffsetGetter._fix_bad_correlation_image_pixels(corrImage)

        # Grab the subpixel precision offsets from the cross correlation image
        dx, dy = ImagePairOffsetGetter._extract_subpixel_offset_from_correlation_image(corrImage)

        # # Add the integer and subpixel offsets and return them to the user
        # dx += dx1
        # dy += dy1

        return dx, dy

    ##################################
    ### END OF OTHER METHODS     ###
    ##################################

class ImageStack(object):
    """
    A class for aligning and combining a list of AstroImage objects.

    Properties
    ----------
    imageList      A list containing all the images in the stack

    Methods
    -------
    add_image                            Appends the provided image instance to
                                         the end of the `imageList`
    pop_image                            Removes and returns the specified image
                                         from the `imageList`
    align_images_with_wcs                Aligns the images using the WCS
                                         solutions in the header
    align_images_with_cross_correlation  Aligns the images using a
                                         cross-correlation technique
    combine_images                       Combines the images to form a single,
                                         average output image
    """

    def __init__(self, imageList, gobble=True):
        """
        Constructs an `ImageStacks` from a list or tuple of AstroImage
        instances. The instances can be of any type as long as they are a
        subclass of the BaseImage class and all the images are of the same type.

        Parameters
        ----------
        imageList : iterable
            An iterable list of AstroImage instances (all of the same type)

        gobble : bool, optional, default: True
            If True, then `imageList` is *emptied* into the ImageStack storage
            attribute so that `imageList` is empty after constructing the
            ImageStack. If False, then the contens of `imageList` are simply
            coppied into the ImageStack storage content, and `imageList` is
            unaffected. The default value is True in order to save memory.

        Returns
        -------
        outStack : `ImageStack`
            A new instance containing the data from the image .
        """
        # Check that a list (or something close o it) was provided
        if not hasattr(imageList, '__iter__'):
            raise TypeError('`imageList` must be a list or iterable object containing image instances')

        # Start by counting the number of images
        numberOfImages = len(imageList)

        # Catch an empty list
        if numberOfImages < 1:
            raise ValueError('`imageList` must contain at least one image instance')

        # Check that the first element of the list is a subclass of `BaseImage`
        thisType = type(imageList[0])
        if not issubclass(type(imageList[0]), BaseImage):
            raise TypeError('{0} type not a recognized astroimage type'.format(thisType))

        # Check if all the images are of the same type
        typeList = [type(img) for img in imageList]
        if typeList.count(typeList[0]) != numberOfImages:
            raise TypeError('All instances in `imageList` must be the same type')

        # Check that the binning is all correct
        imageBinnings = np.array([img.binning for img in imageList])
        dx = imageBinnings[:, 0]
        dy = imageBinnings[:, 1]

        if ((np.sum(dx == dx[0]) != numberOfImages) or
            (np.sum(dy == dy[0]) != numberOfImages)):
            raise ValueError('All instances in `imageList` must have the same binning')

        # Check that the units have the same dimensions.
        unitList = [u.Quantity(1, img.unit).decompose() for img in imageList]
        if unitList.count(unitList[0]) != numberOfImages:
            raise ValueError('All instances in `imageList` must have the same units')

        # Grab the units of the first image and loop through the rest of the
        # images to make sure that they also have the same units.
        targetUnits = imageList[0].unit
        for i in range(numberOfImages):
            if imageList[i].unit != targetUnits:
                imageList[i].convert_units_to(targetUnits)

        # Store an immutable version of the image list
        if gobble:
            # If the gobble parameter is true, then transpose each element of
            # the intput list into the tupple.
            self.__imageList = tuple()
            while len(imageList) > 0:
                self.__imageList += (imageList.pop(0),)
        elif not gobble:
            # If the gobble parameter is false, then just copy the list...
            self.__imageList = tuple(imageList)

        # Store the image list type
        self.__imageType = thisType

        # Set an instance variable to indicate whether the images have been aligned
        if issubclass(self.imageType, (RawScience, ReducedScience)):
            # Test if these images are aligned
            self.__aligned = self._test_alignment()
        else:
            # Calibration images do not require alignment
            self.__aligned = True

        # Initalize a boolean value to indicate that this is NOT a supersky.
        # The boolean will be changed to "True" if-and-only-if the
        # "produce_supersky" method is executed.
        self.__is_supersky = False

        # Force all the image shapes to be the same
        if self.numberOfImages > 1:
            self.pad_images_to_match_shapes()

    ##################################
    ### START OF PROPERTIES        ###
    ##################################
    @property
    def aligned(self):
        """A boolean flag indicating the image alignment"""
        return self.__aligned

    @property
    def imageList(self):
        """The list of images in this stack"""
        return self.__imageList

    @property
    def imageType(self):
        """The type of images stored in this stack"""
        return self.__imageType

    @property
    def is_supersky(self):
        """Boolean flag indicating if this stack has yielded a supersky"""
        return self.__is_supersky

    @property
    def numberOfImages(self):
        """The number of images currently in this stack"""
        return len(self.imageList)

    @property
    def shape(self):
        """The shape of the image stack (nz, ny, nx)"""
        return (self.numberOfImages,) + self.imageList[0].shape

    ##################################
    ### END OF PROPERTIES        ###
    ##################################

    ##################################
    ### START OF OTHER METHODS     ###
    ##################################
    def pad_images_to_match_shapes(self):
        """
        Pads all the images in imageList to have the same shape.

        The padding is applied to the top and right sides of the images, so
        the WCS of those images will be unaffacted if they have WCS.

        Side Effects
        ------------
        Pads the images in place, so the ImageStack on which this is invoked
        will be modified.

        Returns
        -------
        outStack : ImageStack
            The SAME image stack on which this method was invoked but with its
            images now padded to have the same shapes.
        """
        # Force all the images to have the same shape
        imageShapes = np.array([img.shape for img in self.imageList])
        ny, nx      = imageShapes.max(axis=0)

        # Loop through each image and add padding if necessary
        for ny1nx1 in imageShapes:
            ny1, nx1 = ny1nx1
            padY = ny - ny1 if ny1 < ny else 0
            padX = nx - nx1 if nx1 < nx else 0

            # Extract the first image in the imageList
            thisImg = self.pop_image(0)

            if padX > 0 or padY > 0:
                # Pad the image as necessary
                thisImg = thisImg.pad(((0, padY), (0, padX)), 'constant')

            # Return the image to the imageList (at the END of the list)
            self.add_image(thisImg)

        # Hand the padded ImageStack back to the user
        return self

    def add_image(self, image):
        """
        Adds an image to the image stack.

        Parameters
        ----------
        image : `BaseImage` (or subclass)
            The image to be added to the ImageStack instance

        Returns
        -------
        out: None
        """
        if type(image) is not self.imageType:
            raise TypeError('`image` must be of type {0}'.format(self.imageType))

        listBinning = self.imageList[0].binning
        if image.binning != listBinning:
            raise ValueError('`image` must have binning ({0} x {1})'.format(*listBinning))

        self.__imageList = self.imageList + (image,)

        return None

    def pop_image(self, index=None):
        """
        Removes the image at `index` from the imageList and returns it.

        Parameters
        ----------
        index : int
            The index of the image to be removed and returned

        Returns
        -------
        outImg : BaseImage or subclass
            The image stored at `index` in the imageList
        """
        # Check if an index was provided, and if not, then grab the final index
        if index is None: index = self.numberOfImages-1

        # Grab the output image
        try:
            outImg = self.imageList[index]
        except:
            raise

        # Reconstruct the image list
        if index < self.numberOfImages-1:
            self.__imageList = self.imageList[:index] + self.imageList[index+1:]
        else:
            self.__imageList = self.imageList[:index]

        return outImg

    def get_wcs_offsets(self, subPixel=False):
        """
        Computes the relative offsets between `ReducedScience` images in the
        ImageStack using the WCS solutions provided in each image. If ANY of the
        images do not have a WCS solution, then this method returns an error.

        Parameters
        ----------
        subPixel : bool, optional, default: False
            If true, then the image offsets are returned with sub-pixel
            precision. In general, the WCS solution is not accurate enough to
            justify subpixel precision, so the default value is False.

        Returns
        -------
        dx, dy : numpy.ndarray
            The  horizontal (dx) and vertical (dy) offsets required to align the
            images.
        """
        # Check that there is at least ONE image for which to compute an offset
        if self.numberOfImages < 1:
            raise ValueError('Silly rabbit, you need some images to align!')

        # Check that the image list is storing `ReducedScience` type images.
        if not issubclass(self.imageType, ReducedScience):
            raise TypeError('WCS offsets can only be computed for `ReducedScience` type images')

        # Check that all the stored images have WCS
        imagesHaveWCS = [img.has_wcs for img in self.imageList]
        if not all(imagesHaveWCS):
            raise ValueError('All `ReducedScience` instances must have WCS solutions')

        # Grab the WCS of the first image
        refWCS = self.imageList[0].wcs

        # Estimate the central location for this image
        refX, refY = self.imageList[0].shape[1]//2, self.imageList[0].shape[0]//2

        # Convert pixels to sky coordinates
        refRA, refDec = refWCS.wcs_pix2world(refX, refY, 0, ra_dec_order=True)

        # Loop through all the remaining images in the list
        # Grab the WCS of the alignment image and convert back to pixels
        xPos = []
        yPos = []
        for img in self.imageList:
            imgX, imgY = img.wcs.all_world2pix(refRA, refDec, 0)

            # Store the relative center pointing
            xPos.append(float(imgX))
            yPos.append(float(imgY))

        # Compute the relative pointings from the median position
        dx = np.median(xPos) - np.array(xPos)
        dy = np.median(yPos) - np.array(yPos)

        if subPixel:
            # If sub pixel offsets were requested, then add a small `epsilon` to
            # ensure that none of the images have a zero offset.
            dx, dy = self._force_non_integer_offsets(dx, dy)
        else:
            # If integer pixel offsets were requested, then round each offset to
            # its nearest integer value.
            dx = np.round(dx).astype(int)
            dy = np.round(dy).astype(int)

        # Return the image offsets
        return (dx, dy)

    #TODO: break this up into two separate methods:
    # 1) integer_offsets
    # 2) subpixel_offsets

    def _test_alignment(self):
        """
        Tests if the images in the stack are aligned (using build in WCS)
        """
        # If there are less than 2 images, then things are aligned by definition
        if self.numberOfImages < 2: return True

        try:
            # Attempt to get the image offsets using the WCS
            dx, dy = self.get_wcs_offsets()

            # Compute the radial offsets
            radOffsets = np.sqrt(
                (dx[1:] - dx[0])**2 +
                (dy[1:] - dy[0])**2
            )

            # Test if all the radial offsets are less than half a pixel
            aligned = all(radOffsets < 0.5)

            return aligned
        except:
            # If offsets could not be computed, then assume unaligned images
            return False

    def get_cross_correlation_offsets(self, subPixel=False, satLimit=16e3):
        """
        Computes the relative offsets between the images in the ImageStack

        Parameters
        ----------
        subPixel : bool, optional, default: False
            If true, then the image offsets are returned with sub-pixel
            precision. In general, the WCS solution is not accurate enough to
            justify subpixel precision, so the default value is False.

        satLimit : int or float, optional, default: 16e3
            The maximum number of pixel counts permitted for any of the
            reference stars to be used for alignment. Any stars containing a
            pixel brighter than this amount will be omitted from the list of
            permissible reference stars.

        Returns
        -------
        dx, dy : numpy.ndarray
            The horizontal (dx) and vertical (dy) offsets required to align
            the images.
        """
        # Catch the truly trivial case
        numberOfImages = self.numberOfImages
        if self.numberOfImages <= 1:
            return (0, 0)

        # Grab the appropriate reference image depending on whether or not the
        # image stack has already been aligned...
        if self.aligned:
            # If subPixel accuracy was requested, then start by constructing
            # a reference image to be used in sub-pixel alignment.
            referenceImage = self.build_median_image()

            # Initalize lists for storing offsets and shapes, and use the FIRST
            # image in the list as the reference image for now.
            xPos = []
            yPos = []

            # Which image should be the FIRST image to align? Start with the 0th
            # image because we've constructed a separate `referenceImage`
            startInd = 0

        else:
            referenceImage = self.imageList[0]
            # Initalize lists for storing offsets and shapes, and use the FIRST
            # image in the list as the reference image for now.
            xPos = [0]
            yPos = [0]

            # Which image should be the FIRST image to align? Skip the 0th image
            # in this case because it is serving as the referenc image.
            startInd = 1

        # Loop through the rest of the images.
        # Use cross-correlation to get relative offsets,
        # and accumulate image shapes
        progressString = 'Aligning image {0} of {1}'
        for imgNum, image in enumerate(self.imageList[startInd:]):
            # Update the user on the progress
            print(progressString.format(imgNum+startInd+1, numberOfImages), end='\r')

            # Construct an image pair using the reference image
            imgPair = ImagePairOffsetGetter(
                referenceImage,
                image
            )

            # Grab subpixel or integer offsets depending on what was requested
            if subPixel:
                dx, dy = imgPair.get_cross_correlation_subpixel_offset(
                    satLimit=satLimit)
            else:
                dx, dy = imgPair.get_cross_correlation_integer_pixel_offset()

            # Append cross_correlation values for non-reference image
            xPos.append(dx)
            yPos.append(dy)

        # Print a new line for shell output
        print('')

        # TODO: delete these lines if everything is working correctly.
        # Compute the relative pointings from the median position
        dx = np.median(xPos) - np.array(xPos)
        dy = np.median(yPos) - np.array(yPos)

        if subPixel:
            # If sub pixel offsets were requested, then add a small `epsilon` to
            # ensure that none of the images have a zero offset.
            dx, dy = self._force_non_integer_offsets(
                np.array(dx),
                np.array(dy)
            )
        else:
            # If integer pixel offsets were requested, then round each offset to
            # its nearest integer value.
            dx = np.round(dx).astype(int)
            dy = np.round(dy).astype(int)

        return (dx, dy)

    @staticmethod
    def _force_non_integer_offsets(dx, dy, epsilon=1e-4):
        """
        Forces any offset values to be non-integer values by adding epsilon.

        Continues to add epsilon to the offsets until NONE of them are integers.

        Parameters
        ----------
        dx, dy : array_like
            The offset values to force to be non-integer

        epsilon: float, optional, default: 1e-4
            The ammount to add to the offsets in order to make them non-integer

        Returns
        -------
        dx, dy : array_like
            Arrays of offset values, none of which will be integers
        """
        # Copy the input offsets
        outDx = np.array(copy.deepcopy(dx), dtype=float)
        outDy = np.array(copy.deepcopy(dy), dtype=float)

        # Repeatedly add epsilon to the offsets until none of them are integers
        addEpsilon = True
        while addEpsilon:
            # Check for any perfectly integer shifts
            for dx1, dy1 in zip(outDx, outDy):
                # If an integer pixel shift is found, then add tiny shift and
                # try again.
                if dx1.is_integer() or dy1.is_integer():
                    addEpsilon = True
                    outDx += epsilon
                    outDy += epsilon
                    break
            else:
                # If the loop completed, then no epsilon addition necessary!
                addEpsilon = False

        return outDx, outDy

    def apply_image_shift_offsets(self, dx, dy, padding=0):
        """
        Shifts each image in the stack by the ammount specified.

        Parameters
        ----------
        dx, dy : int or float
            The amount to shift each image along the horizontal (dx) and
            vertical (dy) axes.

        padding : int or float, optional, default: 0
            The value to use for padding the edges of the shifted images.
        """
        numberOfOffsets = len(dx)
        if numberOfOffsets != len(dy):
            raise ValueError('`dx` and `dy` must have the same number of elements')

        if numberOfOffsets != self.numberOfImages:
            raise ValueError('There must be one (dx, dy) pair for each image')

        # Recompute the offsets so that all images are shifted up and right
        dx1, dy1 = dx - np.floor(dx.min()), dy - np.floor(dy.min())

        # Compute the required padding
        padX, padY = np.int(np.ceil(dx1.max())), np.int(np.ceil(dy1.max()))

        # Loop through each offset and apply it to the images
        for dx11, dy11 in zip(dx1, dy1):
            # Extract the first image in the imageList
            thisImg = self.pop_image(0)

            # Pad this image with so that shifts do not delete any data
            thisImg = thisImg.pad(
                ((0, padY), (0, padX)),
                mode='constant',
                constant_values=padding
            )

            # Shift the image as necessary
            thisImg = thisImg.shift(dx11, dy11, padding=padding)

            # Return the image to the imageList (at the END of the list)
            self.add_image(thisImg)

    def build_median_image(self):
        """
        Computes fast median image of an aligned image stack.

        Returns
        -------
        medianImage : `~astroimage.reduced.ReducedScience`
            The median of the image stack.
        """
        # Check if the image stack has been aligned
        if not self.aligned:
            raise RuntimeError('ImageStack must be aligned before a median image can be computed')

        # Stack the data arrays
        dataStack = np.array([img.data for img in self.imageList])

        # Compute the median of the data stack
        medianData = np.nanmedian(dataStack, axis=0)

        # Copy the first image in the image stack and replace its data
        medianImage = self.imageList[0].copy()
        medianImage.data = medianData

        # Store the median image in the medianImage attribute
        self.medianImage = medianImage

        return medianImage

    def align_images_with_wcs(self, subPixel=False, padding=0):
        """
        Aligns the whole stack of images using the astrometry in the header.

        NOTE: (2016-06-29) This function *DOES NOT* match image PSFs.
        Perhaps this functionality will be implemented in future versions.

        Parameters
        ----------
        subPixel : bool
            If True, then non-integer pixel shifts will be applied. If False,
            then all shift amounts will be rounded to the nearest integer pixel.

        padding : int or float, optional, default: 0
            The value to use for padding the edges of the aligned images.
        """
        # Catch the case where imageList has only one image
        if self.numberOfImages == 1:
            return imageList[0]

        # If no offsets were supplied, then retrieve them
        dx, dy = self.get_wcs_offsets(subPixel=subPixel)

        if subPixel == True:
            # Make sure there are no integers in the dx, dy list
            dx, dy = self._force_non_integer_offsets(dx, dy)
        else:
            # If non-subpixel alignment was requested, then FORCE all the
            # offsets to the nearest integer value.
            dx = np.round(dx).astype(int)
            dy = np.round(dy).astype(int)


        # Apply the shifts to the images in the stack
        self.apply_image_shift_offsets(dx, dy, padding=padding)

        # Set the alignment flag to True
        self.__aligned = True

        # Set the is_supersky flag to False (in case it was previously set True)
        self.__is_supersky = False

    def align_images_with_cross_correlation(self, subPixel=False,
        satLimit=16e3, padding=0):
        """
        Aligns the whole stack of images using the astrometry in the header.

        NOTE: (2016-06-29) This function *DOES NOT* math image PSFs.
        Perhaps this functionality will be implemented in future versions.

        Parameters
        ----------
        subPixel : bool
            If True, then non-integer pixel shifts will be applied. If False,
            then all shift amounts will be rounded to the nearest integer pixel.

        satLimit : int or float, optional, default: 16e3
            Sources which contain any pixels with more than this number of
            counts will not be used to perform cross-correlation alignment.

        padding : int or float, optional, default: 0
            The value to use for padding the edges of the aligned images.
        """
        # Catch the case where imageList has only one image
        if self.numberOfImages == 1:
            return imageList[0]

        # Check if approximate alignment has already been achieved
        if not self.aligned:
            # Start by retrieving the integer pixel offsets
            print('Aligning images to the integer-pixel level')
            dx, dy = self.get_cross_correlation_offsets(subPixel=False,
                satLimit=satLimit)

            # Align the images to an integer pixel level.
            self.apply_image_shift_offsets(dx, dy, padding=padding)

            # Set the alignment flag to True
            self.__aligned = True

        # If approximate alignment has already been achieved, then simply
        # proceed to get sub-pixel alignment level.
        if subPixel == True:
            # Get the sub-pixel corrections to the alignment.
            print('Aligning images to the sub-pixel level')
            dx, dy = self.get_cross_correlation_offsets(subPixel=True,
                satLimit=satLimit)

            # Make sure there are no integers in the dx, dy list
            dx, dy = self._force_non_integer_offsets(dx, dy)

            # Apply the shifts to the images in the stack
            self.apply_image_shift_offsets(dx, dy, padding=padding)

            # Set the alignment flag to True
            self.__aligned = True

        # else:
        #     # If non-subpixel alignment was requested, then FORCE all the
        #     # offsets to the nearest integer value.
        #     dx = np.round(dx).astype(int)
        #     dy = np.round(dy).astype(int)

        # Set the is_supersky flag to False (in case it was previously set True)
        self.__is_supersky = False

    ####################################
    ### START OF COMBINATION HELPERS ###
    ####################################

    def _get_number_of_rows_to_process(self, bitsPerPixel):
        """
        Computes the number of rows to process at a given time.

        Parameters
        ----------
        bitsPerPixel : int
            The number of bits used to store each pixel

        Returns
        -------
        numRows : int
            The number of rows to include in each section

        numSections : int
            The total number of sections in the ImageStack instance
        """
        # TODO: do a better job estimating the number of rows to process.
        # Compute the number of pixels that fit under the memory limit.
        memLimit    = (psutil.virtual_memory().available/
                      (bitsPerPixel*(1024**2)))
        memLimit    = int(50*np.floor(memLimit/10.0))
        numStackPix = memLimit*(1024**2)*8/bitsPerPixel

        # Grab the number of images and the shape of those image
        numImg, ny, nx  = self.shape

        # Compute the number of rows to be processed in each chunk
        numRows = int(np.floor(numStackPix/(numImg*nx)))

        # Catch the case where ALL rows get handled at once
        if numRows > ny: numRows = ny
        numSections = int(np.ceil(ny/numRows))

        # Recompute the number of rows to be evenly spaced
        numRows = int(np.ceil(ny/numSections))

        return numRows, numSections

    def _produce_individual_star_masks(self, dilationWidth=4):
        """
        Finds the stars in the image stack and builds masks to protect or omit.

        Parameters
        ----------
        dilationWidth : int or float, optional, default: 4
            The amount to circularly dilate outward from masked pixels. These
            roughly translate to pixel values so that `dilationWidth=4` will
            mask any locations within 4 pixels of a crudely identified star
            pixel. This actually depends on the number of other factors, such as
            the number of crudely identified star pixels in a given star, etc...

        Returns
        -------
        starMasks : numpy.ndarray
            A (numberOfImages, ny, nx) array where each slice along the 0th axis
            represents the star mask for the image located at the corresponding
            index in the imageList attribute.
        """
        # TODO: REWRITE THIS METHOD USING THE ASTROPY SEGMENTATION METHODS???
        # Yes, I THINK so...

        # Grab binning
        binX, binY = self.imageList[0].binning

        # Compute kernel shape
        medianKernShape = (np.int(np.ceil(9.0/binX)), np.int(np.ceil(9.0/binY)))

        # Grab the number of images (for user updates)
        numImg = self.numberOfImages

        # Construct a blank array to populate with masks
        starMasks = np.zeros(self.shape, dtype=int)

        # Loop through the images and compute individual star masks
        for imgNum, img in enumerate(self.imageList):
            print('Building star mask for image {0:g} of {1:g}'.format(imgNum + 1, numImg), end='\r')
            # Grab the image array
            thisData = img.data.copy()

            # Replace bad values with zeros
            badInds = np.where(np.logical_not(np.isfinite(thisData)))
            thisData[badInds] = -1e6

            # Filter the image
            medImg = ndimage.median_filter(thisData, size = medianKernShape)

            # get stddev of image background
            mean, median, stddev = img.sigma_clipped_stats()

            # Look for deviates from the filter (positive values only)
            # starMask1 = np.logical_and(np.abs(thisData - medImg) > 2.0*stddev,
            #                            thisData > 0)
            starMask1 = (np.abs(thisData - medImg) > 2.0*stddev)

            # Use the scipy ndimage opening and closing to clean the mask
            starMask1 = ndimage.binary_opening(starMask1)
            starMask1 = ndimage.binary_closing(starMask1)

            # Clean out some edge effects.
            starMask1[0:6,   :] = 0
            starMask1[-6:-1, :] = 0
            starMask1[:, 0:6  ] = 0
            starMask1[:, -6:-1] = 0

            #
            # NOTE: This doesn't work when there are nebulae and galaxies in the image!
            #
            # starMask1 = make_source_mask(
            #     thisData,
            #     snr=2,
            #     npixels=5,
            #     dilate_size=11,
            #     mask_value=-1e6
            # )

            # Try using guassian kernel convolution instead
            from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

            # Initalize a dilatingKernel
            gaussian_2D_kernel = Gaussian2DKernel(10.0)

            # Normalize the kernel
            gaussian_2D_kernel.normalize()

            # If the dialation kernel is larger than 10 pixels, then use FFT
            # convolution.
            starMask11 = convolve_fft(
                starMask1.astype(float),
                gaussian_2D_kernel
            )

            # Mask any pixels with values greater than 0.04 (which seems to
            # produce a reasonable result.)
            peakValue     = 1/(200*np.pi)
            maskThreshold = 10 * peakValue * np.exp(-0.5*((dilationWidth+0.5)/10.0)**2)

            starMask1 = (starMask11 > maskThreshold).astype(np.int8)

            # TODO: delete this code if convolution works out
            #
            # # Finally, liberally EXPAND the mask with four dilations
            # starMask1 = ndimage.binary_dilation(
            #     starMask1,
            #     iterations=starMaskIters
            # ).astype(np.int8)

            # TODO: delete this code once I verify everything is working
            #
            # # Count the number of masked neighbors for each pixel
            # neighborCount = np.zeros(thisData.shape, dtype=int)
            # for dx in range(-1,2,1):
            #     for dy in range(-1,2,1):
            #         neighborCount += np.roll(np.roll(starMask1, dy, axis=0),
            #                                  dx, axis=1).astype(np.int8)
            #
            # # Find pixels with more than two masked neighbor (including self)
            # # starMask1 = np.logical_and(starMask1, neighborCount > 2)
            # starMask1 = (neighborCount > 2).astype(np.int8)

            # Place the final mask into its respective slice of the 3D array
            starMasks[imgNum, :, :] = starMask1

        # Print a newline character to preserve star mask updates
        print('')

        # Once ALL of the star masks have been computed, return them to the user
        return starMasks


    def _construct_star_mask(self):
        """
        Finds stars in the image stack and builds masks to protect or omit.

        Returns
        -------
        mask : numpy.ndarray or bool
            An array of star positions to mask. If no stars were found, then
            simply returns False. This output can be used as the mask in a
            numpy.ma.core.MaskedArray object.
        """
        # Produce a separate star mask for EACH image in the stack
        starMasks = self._produce_individual_star_masks()

        # Accumulate these pixels into the final star mask
        starMask = starMasks.sum(axis=0)

        # Cleanup temporary variables
        del starMasks

        # Compute final star mask based on which pixels were masked more than
        # 10% of the time.
        numImg   = self.numberOfImages
        starMask = (starMask > np.ceil(0.1*numImg)).astype(float)

        # Check that at least one star was detected (more than 15 pixels masked)
        if np.sum(starMask) > 15:
            # Now smooth the star mask with a gaussian to dialate it
            starMask1 = ndimage.gaussian_filter(starMask, (4, 4))

            # Grab any pixels (and indices) above 0.05 value post-smoothing
            starMask     = (starMask1 > 0.05)
            numInStarPix = np.sum(starMask)

            # Notify user how many "in-star pixels" were masked
            print('\n\nMasked a total of {0} pixels'.format(numInStarPix))
        else:
            print('\n\nNo pixels masked as "in-star" pixels')
            starMask = False

        return starMask

    def _get_sigma_clip_start_and_steps(self, iters, backgroundClipSigma=5.0,
         backgroundClipStep=0.5, starClipSigma=40.0, starClipStep=1.0):
        """
        Computes the sigma clipping step sizes

        Parameters
        ----------
        iters : int
            The number of iterations to be used by the sigma-clipping

        backgroundClipSigma : int or float, optional, default: 5.0
            The number of standard-deviations from the median value a pixel
            located outside of a star can deviate from the median before it is
            marked as a bad pixel.

        backgroundClipStep : float, optional, default: 0.5
            The step-size to use for each iteration of the sigma-clipping in
            stellar pixels. If this value forces backgroundClipStart below 0.1,
            then it is reset to 0.5.

        starClipSigma : int or float, optional, default: 40.0
            The number of standard-deviations from the median value a pixel
            located within a star can deviate from the median before it is
            marked as a bad pixel.

        starClipStep: float, optional, default: 1.0
            The step-size to use for each iteration of the sigma-clipping in
            non-stellar background pixels. If this value forces starClipStart
            below 30.0, then it is reset to 1.0.

        Returns
        -------
        backgroundClipStart : float
            The starting point for sigma-clipping of the background pixels

        backgroundClipStep : float
            The step-size to use for each iteration of the sigma-clipping in
            stellar pixels. This is the same as the input value unless it was
            remapped to prevent a backgroundClipStart value below 0.1.

        starClipStart : float
            The starting point for sigma-clipping of the star pixels

        starClipStep: float
            The step-size to use for each iteration of the sigma-clipping in
            non-stellar background pixels. This is the same as the input value
            unless it was remapped to prevent too a starClipStart value below
            30.0.
        """
        # Compute how to iterate through sigma clipping
        # Compute the expected starting point
        backgroundClipStart = backgroundClipSigma - backgroundClipStep*iters
        starClipStart       = starClipSigma - starClipStep*iters

        # Double check that these values are legal
        # (otherwise adjust sigmaStep values)
        if backgroundClipStart < 0.1:
            backgroundClipStart = 0.5
            backgroundClipStep  = (backgroundClipSigma - backgroundClipStart)/iters

        if starClipStart < 30.0:
            starClipStart = 30.0
            starClipStep  = 1.0

        return (
            backgroundClipStart,
            backgroundClipStep,
            starClipStart,
            starClipStep
        )

    def _get_start_and_end_rows(self, sectionNumber, numberOfRows):
        """
        Compute the start and end rows for a given section number.

        Parameters
        ----------
        sectionNumber : int
            The section number to extract (starts at 0)

        numberOfRows : int
            The number of rows to extract from the imageList

        Returns
        -------
        startRow : int
            The index of the first row for the selected region

        endRow : int
            The index of the last row for the selected region
        """
        # Test if the input is logical
        if not issubclass(type(sectionNumber),
            (int, np.int, np.int8, np.int16, np.int32, np.int64)):
            raise TypeError('`sectionNumber` must be an int type')

        if not issubclass(type(numberOfRows),
            (int, np.int, np.int8, np.int16, np.int32, np.int64)):
            raise TypeError('`numberOfRows` must be an int type')

        # Grab the shape of the image stack
        nz, ny, nx = self.shape

        # Compute the range of rows to extract
        startRow = sectionNumber*numberOfRows
        endRow   = (sectionNumber + 1)*numberOfRows

        # Just to be safe, catch the case where we attempt to index BEYOND
        # the last row in the image stack.
        if endRow > ny: endRow = ny-1

        return startRow, endRow

    def _extract_data_sub_stack(self, startRow, endRow):
        """
        Extracts and returns a 3D numpy.ndarray containing the image data.

        Parameters
        ----------
        startRow : int
            The index of the first row in the selected sub stack

        endRow : int
            The index of the last row in the selected sub stack

        Returns
        -------
        outData : numpy.ma.ndarray
            An masked array containing the data
        """
        # Grab the shape of the image stack
        nz, ny, nx = self.shape

        # Compute the number of rows in this sub stack
        numberOfRows = endRow - startRow

        # Build an array for storing output
        outData = np.zeros((nz, numberOfRows, nx))

        # Loop through each image and extract its data
        for zInd, img in enumerate(self.imageList):
            outData[zInd, :, :] = img.data[startRow:endRow, :]

        return np.ma.array(outData)

    def _extract_uncert_sub_stack(self, startRow, endRow):
        """
        Extracts and returns a 3D numpy.ndarray containing the image uncertainty.

        Parameters
        ----------
        startRow : int
            The index of the first row in the selected sub stack

        endRow : int
            The index of the last row in the selected sub stack

        Returns
        -------
        outUncert : numpy.ndarray
            An array containing the uncertainty
        """
        # Grab the shape of the image stack
        nz, ny, nx = self.shape

        # Build a list of which of these images have uncertainty
        numWithUncert = np.sum([img.has_uncertainty for img in self.imageList])

        # If not ALL of the images have uncertainty, then there is no simple
        # algebraic means of treating the uncertainties for SOME of the images,
        # so simply return None for the uncertainty
        if numWithUncert > 0 and numWithUncert < nz:
            # Issue a warning so that the user knows this is happening.
            warnings.warn(
                'Not all images in the ImageStack have associated '
                'uncertainties: estimating uncertainty from data variance. '
                'This will overestimate the uncertainty in stellar pixels.'
            )

        if (numWithUncert != nz):
            outUncert = None
            return outUncert

        # If, however, ALL of the images DO have uncertainty, then proceed to
        # chop out that uncertainty subStack and return it

        # Compute the number of rows in this sub stack
        numberOfRows = endRow - startRow

        # Build an array for storing output
        outUncert = np.zeros((nz, numberOfRows, nx))

        # Loop through each image and extract its data
        for zInd, img in enumerate(self.imageList):
            outUncert[zInd, :, :] = img.uncertainty[startRow:endRow, :]

        return outUncert

    @staticmethod
    def _initalize_mask(dataSubStack):
        """
        Initalizes and output mask and masks NaNs and Infs in the input array

        Parameters
        ----------
        dataSubStack : numpy.ma.ndarray
            The data array in which the bad pixels are to be found

        Returns
        -------
        outMask : numpy.ndarray (bool type)
            An empty mask in which the final output will be stored

        outSubStack: numpy.ndarray
            An array containing all the same data as dataSubStack but with any
            NaNs or Infs masked
        """
        # Initalize an array to store the output mask values
        outMask = np.zeros(dataSubStack.shape, dtype=bool)

        # Start by masking out NaNs or Infs
        NaNsOrInfs        = np.logical_not(np.isfinite(dataSubStack.data))
        dataSubStack.mask = NaNsOrInfs

        return outMask, dataSubStack

    @staticmethod
    def _increment_sigma_clip_scale(inputClipScale, nextScale,
        backgroundClipStep, starClipStep=0, starMask=False):
        """
        Builds an array to indicate the sigma-clipping level for each column

        Parameters
        ----------
        inputClipScale : numpy.ndarray
            An array containing the sigmga-clipping scale BEFORE this iteration

        nextScale : numpy.ndarray (bool type)
            An array indicating which columns should have their sigma-clipping
            scale incremented by another step

        backgroundClipStep : int or float
            The amount to increment the sigma-clipping for background pixels

        starClipStep : int or float, optional, default: 0
            The amount to increment the sigma-clipping for stellar pixels. If
            this is not provided, then `starMask` must also be 0.

        starMask : numpy.ndaray (bool type), optional, default: False
            An array indicating which pixels are located in stars. True
            indicates stellar pixels. If this is provided, then `starClipStep`
            must also be provided.

        Returns
        -------
        outputClipScale : numpy.ndarrray
            An array containing the sigma-clipping scale AFTER this iteration
        """
        # If StarMask is false, then set starClipStep to zero.
        if starMask is False:
            starClipStep = 0

        # Generate simple arrays to indicate which pixels are background/star
        starPix       = np.array(starMask).astype(int)
        backgroundPix = np.logical_not(starPix).astype(int)

        # Convert the nextScale array into an integer array
        nextScale = np.array(nextScale).astype(int)

        # Copy the input sigma-clipping scale and add the next steps to it.
        outputClipScale  = inputClipScale
        outputClipScale += backgroundClipStep * nextScale * backgroundPix
        outputClipScale += starClipStep * nextScale * starPix

        return outputClipScale

    @staticmethod
    def _process_sigma_clip_iteration(dataSubStack, NaNsOrInfs, startNumMasked,
        sigmaClipScale):
        """
        Processes the next step in the sigma clip iteration

        Parameters
        ----------
        dataSubStack : numpy.ma.ndarray
            The 3D data array in which to located bad pixels

        NaNsOrInfs : numpy.ndarray
            A 2D array indicating the locations of NaNs or Infs in dataSubStack

        startNumMasked : numpy.ndarray
            A 2D array indicating the number of masked pixels in each column of
            the dataSubStack BEFORE this iteration. This should be equal to
            numpy.sum(dataSubStack.mask, axis=0)

        sigmaClipScale : numpy.ndarray
            A 2D array containing the current sigma clipping level of each
            column of pixels in dataSubStack

        Returns
        -------
        dataSubStack : numpy.ma.ndarray
            The same as the input dataSubStack array, but with its mask updated.

        nextScale : numpy.ndarray
            A 2D array indicating which columns should continue to the next step
            of iteration.

        endNumMasked : numpy.ndarray
            A 2D array indicating the number of masked pixels in each column of
            the datasubStack AFTER this iteration.
        """
        # Estimate the median and standard deviation of this subStack
        imgEstimate = np.ma.median(dataSubStack, axis = 0).data
        stackSigma  = np.ma.std(dataSubStack, axis = 0).data

        # Build a bool array for marking the outliers of this dataSubStack
        outliers = np.zeros(dataSubStack.shape, dtype=bool)

        # Loop through the stack, and find the outliers.
        for j in range(dataSubStack.shape[0]):
            deviation       = np.absolute(dataSubStack.data[j,:,:] - imgEstimate)
            outliers[j,:,:] = (deviation > sigmaClipScale*stackSigma)

        # Save the newly computed outliers to the mask
        dataSubStack.mask = np.logical_or(outliers, NaNsOrInfs)

        # Count the number of masked points after the iteration
        endNumMasked = np.sum(dataSubStack.mask, axis=0)

        # Determine which pixel columns experienced a CHANGE in the number of
        # masked pixels and mark those for continued iteration
        nextScale = endNumMasked != startNumMasked

        return dataSubStack, nextScale, endNumMasked

    def _construct_sub_stack_bad_pixel_mask(self, dataSubStack, starSubMask,
        iters=5, backgroundClipStart=2.5, backgroundClipStep=0.5,
        starClipStart=35.0, starClipStep=1.0):
        """
        Computes the bad pixels to be masked in the median filtered mean.

        Parameters
        ----------
        dataSubStack : numpy.ma.ndarray
            The array of values for which to identify outliers and compute masks
            to cover those bad pixes.

        starSubMask : array numpy.ndarray (bool type)
            An array of booleans indicating which pixels in the aligned images
            are inside stellar PSFs. True values indicate pixels inside a
            stellar PSF. A scalar False valeu indicates there are no stars to
            mask.

        iters : int, optional, default: 5
            The number of sigma-clipping iterations to perform when searching
            for bad pixels

        backgroundClipStart : int or float, optional, default: 2.5
            The sigma-clipping level for background pixels during the first
            iteration.

        backgroundClipStep : int or float, optional, default: 0.5
            The increment by whech to inclease the sigma-clipping level for
            background pixels after each iteration.

        starClipStart : int or float, optional, default: 35.0
            The sigma-clipping level for stellar pixels during the first
            iteration.

        starClipStep : int or float, optional, default: 1.0
            The increment by whech to inclease the sigma-clipping level for
            stellar pixels after each iteration.

        Returns
        -------
        dataSubStack : numpy.ma.ndarray
            The `mask` attribute of this objcet is a bool type array where True
            values indicate bad pixels based on the provided backgroundClipSigma
            and starClipSigma values.
        """
        # Initalize the output array and cover up any NaNs/Infs in dataSubStack
        outMask, dataSubStack = self._initalize_mask(dataSubStack)

        # At this stage, only NaNs or Infs are masked, so save that information
        # for use when processing the sigma-clipping.
        NaNsOrInfs = dataSubStack.mask

        # Initalize an array of sigma-clipping values
        sigmaClipScale = self._increment_sigma_clip_scale(
            0, # Start at zero sigma-clipping
            1, # Increment EVERY pixel up to its starting value
            backgroundClipStart,
            starClipStart,
            starSubMask
        )

        # Compute the starting number of pixels masked in each column
        startNumMasked = np.sum(dataSubStack.mask, axis=0)

        # This loop will iterate until the mask converges to an
        # unchanging state, or until clipSigma is reached.
        for iLoop in range(iters):
            print('\tProcessing section for (\u03C3(bkg), \u03C3(*)) = ({0:3.2g}, {1:3.2g})'.format(
                backgroundClipStart + backgroundClipStep*iLoop,
                starClipStart + starClipStep*iLoop))

            # Perform the next iteration in the sigma-clipping
            dataSubStack, nextScale, startNumMasked = self._process_sigma_clip_iteration(
                dataSubStack,
                NaNsOrInfs,
                startNumMasked,
                sigmaClipScale
            )

            if np.sum(nextScale) == 0:
                # If there are no new data included, then break out of loop
                break
            else:
                # Otherwise increment scale where new data are included
                sigmaClipScale = self._increment_sigma_clip_scale(
                    sigmaClipScale, # Start at the original sigma-clipping
                    nextScale, # Increment only columns with changed masking
                    backgroundClipStep, # Amount to increment bkg pixels
                    starClipStep,  # Amount to increment star pixels
                    starSubMask # Array indictating star pixels
                )

        # When finished processing each sub stack, return the final
        return dataSubStack

    @staticmethod
    def _propagate_masked_uncertainty(uncertainty, mask):
        """Computes the uncertainty in the masked array."""
        # Compute the variance of the total quantity
        varianceOfTheTotal = np.nansum(uncertainty**2, axis=0)

        # Count the number of unmasked pixels in each column of the stack
        goodPix = np.logical_and(
            np.logical_not(mask),
            np.isfinite(uncertainty)
        )
        numberOfUnmaskedPixels = np.sum(goodPix.astype(np.int16), axis=0)

        # Estimate the uncertainty by dividing the variance by the number of
        # unmasked pixels in each column, and then taking the square root.
        maskedUncertainty = np.sqrt(varianceOfTheTotal)/numberOfUnmaskedPixels

        return maskedUncertainty

    @staticmethod
    def _compute_masked_mean_and_uncertainty(maskedData, uncertainty):
        """
        Computes the mean and uncertainty in the mean of a masked array.
        """
        # Compute the mean of the unmasked pixels
        maskedMean = maskedData.mean(axis=0).data

        # Compute the masked uncertainty array
        if uncertainty is None:
            # If not all of the images had uncertainty arrays, then we must
            # resort to estimating the uncertainty from the data variance.
            maskedUncertainty = maskedData.std(axis=0).data
        else:
            # If an array of uncertainties was provided, then we can proceed by
            # propagating those uncertainties.
            maskedUncertainty = ImageStack._propagate_masked_uncertainty(
                uncertainty,
                maskedData.mask
                )

        return maskedMean, maskedUncertainty

    def _compute_stack_mean_and_uncertainty(self,  starMask, iters=5,
        backgroundClipSigma=5.0, starClipSigma=40.0):
        """
        Computes the mean and uncertainty of the complete stack.

        Takes a star mask and treats pixels inside stars with a more forgiving
        sigma-clipping than pixels outside stars.

        Parameters
        ----------
        starMask : numpy.ndarray (bool type)
            An array of booleans indicating which pixels in the aligned images
            are inside stellar PSFs. True values indicate pixels inside a
            stellar PSF.

        iters : int, optional, default: 5
            The number of sigma-clipping iterations to perform when searching
            for bad pixels

        backgroundClipSigma : int or float, optional, default: 5.0
            The number of standard-deviations from the median value a pixel
            located outside of a star can deviate from the median before it is
            marked as a bad pixel.

        starClipSigma : int or float, optional, default: 40.0
            The number of standard-deviations from the median value a pixel
            located within a star can deviate from the median before it is
            marked as a bad pixel.

        Returns
        -------
        outMean : numpy.ndarray
            The mean image

        outUncert : numpy.ndarray
            The uncertainty in that mean. If the images in the stack have
            associated uncertainties, then this will be a propagated
            uncertainty. If they do not have associated uncertainties, then this
            will be estimated from the variance in the stack pixel values.

            !WARNING! - Estimating uncertainty from the pixel variance will lead
            to unreasonably high uncertainties in stellar PSFs. Thus, it is
            much better to load the ReducedScience instances with an estimated
            detector gain so that a Poisson uncertainty can be used.
        """
        # Extract the number of images (nz) and the shape of the images (ny, nx)
        nz, ny, nx = self.shape

        # Test for the number of bits in each pixel (or just assum 64 bits)
        bitsPerPixel = 64

        # Compute the number of rows to process at a given time
        numberOfRows, numSections = self._get_number_of_rows_to_process(bitsPerPixel)
        print('Processing stack in {0} sections of {1} rows'.format(
            numSections, numberOfRows))

        # Compute the sigma-clipping starting points and increments
        tmp = self._get_sigma_clip_start_and_steps(
            iters=iters,
            backgroundClipSigma=backgroundClipSigma,
            starClipSigma=starClipSigma
        )
        backgroundClipStart, backgroundClipStep, starClipStart, starClipStep = tmp

        # Initalize an empty array to hold the output
        outMean   = np.zeros((ny, nx))
        outUncert = np.zeros((ny, nx))

        for sectionNumber in range(numSections):
            print('Starting section number {0}'.format(sectionNumber+ 1 ))
            # Compute the range of rows to extract
            startRow, endRow = self._get_start_and_end_rows(
                sectionNumber, numberOfRows
            )

            # Extract the data for this section
            dataSubStack = self._extract_data_sub_stack(startRow, endRow)

            # Extract the uncertainty for this section
            uncertSubStack = self._extract_uncert_sub_stack(startRow, endRow)

            # Extract the starSubMask for this section
            if issubclass(type(starMask), np.ndarray):
                starSubMask = starMask[startRow:endRow, :]
            elif issubclass(type(starMask), bool):
                starSubMask = starMask

            # Build the bad pixel mask for this subStack
            dataSubStack = self._construct_sub_stack_bad_pixel_mask(
                dataSubStack,
                starSubMask,
                iters=iters,
                backgroundClipStart=backgroundClipStart,
                backgroundClipStep=backgroundClipStep,
                starClipStart=starClipStart,
                starClipStep=starClipStep
            )

            # Compute the mean and uncertainty of the masked array
            mean, uncert = self._compute_masked_mean_and_uncertainty(
                dataSubStack, uncertSubStack)

            # Store the result in the output
            outMean[startRow:endRow, :] = mean
            outUncert[startRow:endRow, :] = uncert

        return outMean, outUncert

    def _parse_stars_according_to_image(self, starClipSigma=40.0):
        """
        Builds star mask for ReducedScience and turns of star masking for others

        Parameters
        ----------
        starClipSigma : int or float, optional, default: 40.0
            The number of standard-deviations from the median value a pixel
            located within a star can deviate from the median before it is
            marked as a bad pixel.

        Returns
        -------
        starMask : bool or numpy.ndarray (bool type)
            An array indicating which pixels are located in stars. True
            indicates stellar pixels. If this is provided, then `starClipStep`
            must also be provided. If no star clipping should happen (i.e. this
            image does not contain stars), then this is simply set to False.

        starClipSigma : int or float
            This is the same as the input starClipSigma value except that it is
            set to 0 if no star clipping sholud happen (i.e. this image does
            not contain stars)
        """
        if issubclass(self.imageType, ReducedScience):
            # Check if all the images were corrected to Airmass 0.0
            if np.sum([img.airmass for img in self.imageList]) > 0:
                raise ValueError('All images in the imageList must be corrected to airmass=0.0 before combining')

            # Compute the star masks for this image stack.
            starMask = self._construct_star_mask()

        else:
            starMask = False
            starClipSigma = 0

        return starMask, starClipSigma

    def _finalize_output(self, stackMean, stackUncert):
        """
        Places mean and uncertainty into an image object and solves astrometry

        Only attempts astrometric solution if the output image was an
        ReducedScience instance.

        Parameters
        ----------
        stackMean : numpy.ndarray
            The resulting median-filtered-mean of the stacked data

        stackUncert : numpy.ndarray
            The resulting uncertainty in the `stackMean` values

        Returns
        -------
        outImg : ReducedImage (or subclass)
            An image instance containing the mean and uncertainty and
            astrometric solution if possible.
        """
        # Select the type of output image to be built on the basis of the image
        # obsType.
        outImageClassDict = {
            'BIAS': MasterBias,
            'DARK': MasterDark,
            'FLAT': MasterFlat,
            'OBJECT': ReducedScience
        }
        outImageClass = outImageClassDict[self.imageList[0].obsType]

        # TODO: decide if it is a good idea to have an optional uncertainty...
        # # Properly handle the uncertainty provided
        # if stackUncert is not None:
        #     outUncert = StdDevUncertainty(stackUncert)
        # else:
        #     outUncert = None

        # Return that data to the user in a single AstroImage instance
        outImg = outImageClass(
            stackMean,
            uncertainty=StdDevUncertainty(stackUncert),
            header=self.imageList[0].header,
            properties={'unit': self.imageList[0].unit}
        )

        # Clean up any bad pixels in this image using the Inpointer class
        inpainter = Inpainter(outImg)
        outImg    = inpainter.inpaint_nans()

        # If the output image is an ReducedScience and is not a supersky image,
        # then clear out the old astrometry and solve it anew!
        if (outImageClass is ReducedScience) and not self.is_supersky:
            # Clear out the old astrometry
            outImg.clear_astrometry()

            # Initalize an astrometry solver object
            astroSolver = AstrometrySolver(outImg)

            # Attempt to perform an astrometric solution
            temporaryImage, success = astroSolver.run()

            # If astrometry solution was successful, then replace the output
            if success: outImg = temporaryImage

        return outImg

    def _compute_supersky(self, starMasks):
        """
        Computes the masked median of the unaligned image stack.

        Parameters
        ----------
        starMasks : numpy.ndarary
            A (numberOfImages, ny, nx) array containing a True value wherever
            there are stars and a False value in all the sky pixels.

        Returns
        -------
        supersky : numpy.ndarray
            A (ny, nx) array containing the median sky-counts in each pixel
        """
        # TODO: break this into more managably bite sized bits if necessary.

        # Construct a median normalized data stack
        dataStack = np.zeros(self.shape, dtype=np.float32)

        # Loop through each image, normalize and place in data stack
        for imgNum, img in enumerate(self.imageList):
            # Copy the data for this image
            thisData = img.data

            # Mask this image with its starMask
            starInds = np.where(starMasks[imgNum, :, :])
            thisData[starInds] = np.NaN

            # Compute the median of this image
            thisMedian = np.nanmedian(thisData)

            # Median normalize this image
            thisData /= thisMedian

            # Place the normalized image in its place
            dataStack[imgNum, :, :] = thisData

        # Compute the median image (ignore warnings because we'll fix those)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            medianArray = np.nanmedian(dataStack, axis=0)

            # Comptue uncertainty as standard deviation/sqrt(numOfUnmaskedPixels)
            stdArray    = np.nanstd(dataStack, axis=0)
            numPix      = np.nansum(dataStack, axis=0)
            uncertArray = stdArray/np.sqrt(numPix - 1)

            # Renormalize by this output median
            thisMedian = np.nanmedian(medianArray)
            medianArray /= thisMedian
            uncertArray /= np.abs(thisMedian)

        # Return to user
        return medianArray, uncertArray

    ####################################
    ### END OF COMBINATION HELPERS   ###
    ####################################

    ####################################
    ### START OF COMBINATION METHODS ###
    ####################################

    def produce_supersky(self, dilationWidth=4):
        """
        Computes the median of the unregistered image stack.

        Parameters
        ----------
        dilationWidth : int or float, optional, default: 4
            The amount to circularly dilate outward from masked pixels. These
            roughly translate to pixel values so that `dilationWidth=4` will
            mask any locations within 4 pixels of a crudely identified star
            pixel. This actually depends on the number of other factors, such as
            the number of crudely identified star pixels in a given star, etc...

        Returns
        -------
        outImg : ReducedImage (or subclass)
            The average sky image with stars masked.
        """
        # Catch if the images have been aligned and raise an error
        if self.aligned:
            raise RuntimeError('Cannot produce supersky with aligned images')

        # Catch if there are enough images to produce a supersky
        if self.numberOfImages < 2:
            raise RuntimeError('Cannot produce supersky with less than 2 images')

        # Produce individual star masks for each image in the ImageStack
        starMasks = self._produce_individual_star_masks(
            dilationWidth=dilationWidth
        )

        # Compute the mean and uncertainty given this star mask
        stackMedian, stackUncert = self._compute_supersky(starMasks)

        # Set the boolean supersky flag
        self.__is_supersky = True

        # Place the stack average and uncertainty into an array
        outImg = self._finalize_output(stackMedian, stackUncert)

        # Remove the units from the output image because superskys don't have units
        outImg /= (1.0*outImg.unit)

        # Return the resulting image to the user
        return outImg

    def combine_images(self, iters=5, double=False, backgroundClipSigma=5.0,
        starClipSigma=40.0):
        """
        Computes the median filtered mean of the image stack.

        Starts by identifying which pixels are located in stars and applies a
        more tolerant sigma-clipping procedure to those pixels.

        Parameters
        ----------
        iters : int, optional, default: 5
            The number of sigma-clipping iterations to perform when searching
            for bad pixels

        double : bool, optional, default: False
            If True, then the output image will be computed as a 64-bit float.
            If False, then the output image will be computed as a 32-bit float.

        backgroundClipSigma : int or float, optional, default: 5.0
            The number of standard-deviations from the median value a pixel
            located outside of a star can deviate from the median before it is
            marked as a bad pixel.

        starClipSigma : int or float, optional, default: 40.0
            The number of standard-deviations from the median value a pixel
            located within a star can deviate from the median before it is
            marked as a bad pixel.

        Returns
        -------
        outImg : ReducedImage (or subclass)
            The average image and uncertainty based on the input image list
        """
        # Catch the truly trivial case
        if self.numberOfImages <= 1:
            return self.imageList[0]

        # Catch if the images have not been aligned
        if not self.aligned:
            raise RuntimeError('This ImageStack has not yet been aligned')

        # If this is not an astroimage, then catch it and PREVENT star clipping
        tmp = self._parse_stars_according_to_image(starClipSigma)
        starMask, starClipSigma = tmp

        # Compute the mean and uncertainty given this star mask
        stackMean, stackUncert = self._compute_stack_mean_and_uncertainty(
            starMask,
            iters=iters,
            backgroundClipSigma=backgroundClipSigma,
            starClipSigma=starClipSigma
        )

        # Place the stack average and uncertainty into an array
        outImg = self._finalize_output(stackMean, stackUncert)

        # Return the resulting image to the user
        return outImg
