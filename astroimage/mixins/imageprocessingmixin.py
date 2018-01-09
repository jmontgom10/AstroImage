# Scipy imports
import numpy as np

# Astropy imports
from astropy.nddata import NDDataArray, StdDevUncertainty
from astropy.convolution import convolve

# Import the astroimage base class
from ..baseimage import BaseImage

__all__ = ['ImageProcessingMixin']

class ImageProcessingMixin(object):
    """
    A mixin class to handle image processing methods for ReducedScience class
    """

    def convolve(self, kernel):
        """Convolves the image with the provided kernel"""
        # Convolve the image data
        convalData = convolve(self.data, kernel)

        # Convolve the variance image to get the output variance
        convolVariance = convolve(self.uncertainty**2)

        # Convert the output variance to an uncertainty object
        convolUncert = StdDevUncertainty(np.sqrt(convalVariance))

        # Construct the output NDDataArray
        convolFullData = NDDataArray(
            convolData,
            uncertainty=convolUncert,
            unit=self._BaseImage__fullData.unit,
            wcs=self.wcs
        )

        # Store the rebinned FullData
        outImg._BaseImage__fullData = rebinFullData
