# Scipy imports
import numpy as np
from scipy import ndimage, signal

# Astropy imports
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDDataArray, StdDevUncertainty

# Define which functions, classes, objects, etc... will be imported via the command
# >>> from inpainter import *
__all__ = ['Inpainter']

class Inpainter(object):
    """
    A class to inpaint any masked or bad values for astroimage classes.
    """

    def __init__(self, image):
        """
        Constructs an Inpainter instance.

        Parameters
        ----------
        image : BaseImage or subclass
        """
        # simply store the image in the image attribute
        self.image = image

    @staticmethod
    def _prepare_array_for_inpainting(array, mask=False):
        """
        Applies mask with NaNs to prepare for inpainting

        Parameters
        ----------
        array : numpy.ndarray
            The array to be inpainted.

        Returns
        -------
        ourArray : numpy.ndarray
            A copy of the input array with all non-finite and masked pixels
            replaced with NaNs. This enables the inpainting method to locate all
            pixels marked for inpainting.

        proceed : bool
            A flag indicating whether any inpainting should be done. True
            indicates that there are pixels to inpaint and False indicates there
            are no pixels to inpaint.
        """
        # Locate any problematic or masked pixels
        badPix = np.logical_or(
            np.logical_not(np.isfinite(array)),
            mask
        )

        # Count the number of pixels to inpaint
        if np.sum(badPix) == 0:
            # If the number of pixels to inpaint is zero, then just rutren
            outArray = array
            proceed = False
            return array, proceed

        # If there are some pixels to inpaint, then proceed!
        proceed = True
        badInds = np.where(badPix)

        # Copy the input array
        outArray = array.copy()

        # Place a NaN in all the bad pixels to make them easier to find
        outArray[badInds] = np.NaN

        return outArray, proceed

    @staticmethod
    def _inpaint_array(array):
        """Inpaints the NaN values in the array"""

        # If there is no mask provided, then simply make one out of the NaNs
        badPix = np.isnan(array)

        # If no pixels were selected for inpainting, just return a copy of the image
        if np.sum(badPix) == 0:
            return array

        # First get the indices for the good and bad pixels
        goodInds = np.where(np.logical_not(badPix))
        badInds  = np.where(badPix)

        # Replace badInds with image median value
        repairedArr1 = array.copy()
        mean, median, stddev = sigma_clipped_stats(repairedArr1[goodInds])
        repairedArr1[badInds] = median

        # # On first pass, smooth the input image with kernel ~5% of image size.
        # ny, nx       = arr.shape
        # kernelSize   = np.int(np.round(0.05*np.sqrt(nx*ny)))
        # repairedArr1 = ndimage.gaussian_filter(repairedArr1, kernelSize)
        #
        # # Replace good pix with good values
        # repairedArr1[goodInds] = arr[goodInds]

        # Iterative kernel size
        iterKernelSize = 10

        # Loop through and keep smoothing the array
        meanDiff = 1.0
        while meanDiff > 0.1:
            # Smooth the image over with a smaller, 10 pixel kernel
            repairedArr = ndimage.gaussian_filter(repairedArr1, iterKernelSize)

            # Immediately replace the good pixels with god values
            repairedArr[goodInds] = array[goodInds]

            # Compute the mean pixel difference
            pixelDiffs  = np.abs(repairedArr1[badInds] - repairedArr[badInds])
            meanDiff    = np.mean(pixelDiffs)

            # Now that the image has been smoothed, swap out the saved array
            repairedArr1 = repairedArr

        # Do another iteration but this time on SMALL scales
        iterKernelSize = 4

        # Loop through and keep smoothing the array
        meanDiff = 1.0
        while meanDiff > 1e-5:
            # Smooth the image over with a smaller, 10 pixel kernel
            repairedArr = ndimage.gaussian_filter(repairedArr1, iterKernelSize)

            # Immediately replace the good pixels with god values
            repairedArr[goodInds] = array[goodInds]

            # Compute the mean pixel difference
            pixelDiffs  = np.abs(repairedArr1[badInds] - repairedArr[badInds])
            meanDiff    = np.mean(pixelDiffs)

            # Now that the image has been smoothed, swap out the saved array
            repairedArr1 = repairedArr

        # Return the actual AstroImage instance
        return repairedArr


    def inpaint_nans(self, mask=False):
        """
        Locates and replaces any NaN values in the image.

        Parameters
        ----------
        mask : numpy.ndarray or bool, optional, default: False
            A numpy array indicating the location of pixels to be inpainted.
            Any pixels with a True value will be inpainted.

        Returns
        -------
        outImg : astroimage.BaseImage subclass
            An image instance with the masked or NaN pixels repaired.
        """
        # Check if mask is the right type
        if not issubclass(type(mask), (bool, np.ndarray)):
            raise TypeError('`mask` must be an numpy.ndarray')

        # Check if mask is the right shape
        if issubclass(type(mask), np.ndarray):
            if mask.shape != self.image.shape:
                raise ValueError('`mask` must have the same shape as the image to be inpainted')

        # Prepare the data array for inpainting
        maskedData, proceed = self._prepare_array_for_inpainting(
            self.image.data,
            mask=mask
        )

        if not proceed:
            # If nothing to do, then just return to the user.
            return self.image

        print('Inpainting masked and NaN pixels.')

        # Inpaint the data array
        outData = self._inpaint_array(maskedData)

        # Handle the uncertainty array if it exists
        if self.image.uncertainty is not None:
            # Prepare the uncertainty array for inpainting
            maskedUncert, proceed = self._prepare_array_for_inpainting(
                self.image.uncertainty,
                mask=mask
            )

            # Inpaint the uncertainty array (using addition in quadrature)
            outUncert = np.sqrt(self._inpaint_array(maskedUncert**2))

            # Wrap the uncertainty in the StdDevUncertainty class
            outUncert = StdDevUncertainty(outUncert)
        else:
            outUncert = None

        # Construct the output image
        outImg = self.image.copy()
        outImg._BaseImage__fullData = NDDataArray(
            outData,
            uncertainty=outUncert,
            unit=self.image.unit,
            wcs=self.image.wcs
        )

        return outImg
