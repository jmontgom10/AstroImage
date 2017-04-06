# Scipy imports
import numpy as np

# Astropy imports
from astropy.nddata import NDDataArray, StdDevUncertainty
import astropy.units as u

class ImageNumericsMixin(object):
    """
    A mixin class to handle common numerical methods for ReducedScience class
    """

    ##################################
    ### START OF NUMERICAL METHODS ###
    ##################################

    def rad2deg(self):
        '''Converts the image data values from radians to degrees'''
        if self.unit != u.rad: raise u.UnitError('Units are not currently radians')
        return self.convert_units_to(u.deg, copy=True)

    def deg2rad(self):
        '''Converts the image data values from degrees to radians'''
        if self.unit != u.deg: raise u.UnitError('Units are not currently degrees')
        return self.convert_units_to(u.rad, copy=True)

    def sin(self):
        '''Computes the sine of the image data values'''
        # Check if the image has angle units
        if not self.has_angle_units:
            raise TypeError('Can only apply `sin` function to quantities with angle units')

        selfRadData = (self.data*self.unit).to(u.rad)
        selfData    = selfRadData.value

        # Propagate uncertainty if it exists
        if self._BaseImage__fullData.uncertainty is not None:
            selfRadUncert = (self.uncertainty*self.unit).to(u.rad)
            selfUncert    = selfRadUncert.value
            outUncert     = StdDevUncertainty(np.cos(selfData)*selfUncert)
        else:
            outUncert = None

        # Compute the sine and propagated uncertainty
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDatArray(
            np.sin(selfData),
            uncertainty=outUncert,
            unit=u.dimensionless_unscaled,
            wcs=self._BaseImage__fullData.wcs
        )

        return outImg

    def arcsin(self):
        '''Computes the arcsin of the image data values'''
        # Check if the image is a dimensionless quantity
        if not self.has_dimensionless_units:
            raise TypeError('Can only apply `arcsin` function to dimensionless quantities')

        # Grab the data
        selfData   = self.data

        # Propagate uncertainty if it exists
        if self._BaseImage__fullData.uncertainty is not None:
            selfUncert = self.uncertainty
            outUncert  = StdDevUncertainty(selfUncert/np.sqrt(1.0 + selfData))
        else:
            outUncert = None

        # Compute the arcsin and store the propagated uncertainty
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(
            np.arcsin(selfData),
            uncertainty=outUncert,
            unit=u.rad,
            wcs=self._BaseImage__fullData.wcs
        )

        return outImg

    def cos(self):
        '''Computes the cosine of the image data values'''
        # Check if the image has angle units
        if not self.has_angle_units:
            raise TypeError('Can only apply `cos` function to quantities with angle units')

        selfRadData = (self.data*self.unit).to(u.rad)
        selfData    = selfRadData.value

        # Propagate uncertainty if it exists
        if self._BaseImage__fullData.uncertainty is not None:
            selfRadUncert = (self.uncertainty*self.unit).to(u.rad)
            selfUncert    = selfRadUncert.value
            outUncert     = StdDevUncertainty(np.sin(selfData)*selfUncert)
        else:
            outUncert = None

        # Compute the sine and store the propagated uncertainty
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDatArray(
            np.cos(selfData),
            uncertainty=outUncert,
            unit=u.dimensionless_unscaled,
            wcs=self.wcs
        )

        return outImg

    def arccos(self):
        '''Computes the arccosine of the image data values'''
        # Check if the image is a dimensionless quantity
        if not self.has_dimensionless_units:
            raise TypeError('Can only apply `arccos` function to dimensionless quantities')

        # Grab the data
        selfData   = self.data

        # Propagate uncertainty if it exists
        if self._BaseImage__fullData.uncertainty is not None:
            selfUncert = self.uncertainty
            outUncert  = StdDevUncertainty(selfUncert/np.sqrt(1.0 + selfData))
        else:
            outUncert = None

        # Compute the arcsin and store the propagated uncertainty
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(
            np.arccos(selfData),
            uncertainty=outUncert,
            unit=u.rad,
            wcs=self._BaseImage__fullData.wcs
        )

        return outImg

    def tan(self):
        '''Computes the tangent of the image data values'''
        # Check if the image has angle units
        if not self.has_angle_units:
            raise TypeError('Can only apply `tan` function to quantities with angle units')

        selfRadData = (self.data*self.unit).to(u.rad)
        selfData    = selfRadData.value

        # Propagate uncertainty if it exists
        if self._BaseImage__fullData.uncertainty is not None:
            selfRadUncert = (self.uncertainty*self.unit).to(u.rad)
            selfUncert    = selfRadUncert.value
            outUncert     = StdDevUncertainty(selfUncert/(np.cos(selfData)**2))
        else:
            outUncert = None

        # Compute the sine and store the propagated uncertainty
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDatArray(
            np.tan(selfData),
            uncertainty=outUncert,
            unit=u.dimensionless_unscaled,
            wcs=self.wcs
        )

        return outImg

    def arctan(self):
        '''Computes the arctan of the image data values'''
        # Check if the image is a dimensionless quantity
        if not self.has_dimensionless_units:
            raise TypeError('Can only apply `arctan` function to dimensionless quantities')

        # Grab the data
        selfData   = self.data

        # Propagate uncertainty if it exists
        if self._BaseImage__fullData.uncertainty is not None:
            selfUncert = self.uncertainty
            outUncert  = StdDevUncertainty(selfUncert/(1.0 + selfData**2))
        else:
            outUncert = None

        # Compute the arcsin and store the propagated uncertainty
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(
            np.arccos(selfData),
            uncertainty=outUncert,
            unit=u.rad,
            wcs=self._BaseImage__fullData.wcs
        )

        return outImg

    def arctan2(self, other):
        '''Computes the 'smart' arctan of the ratio of two images'''
        # Grab the data if posible
        if not self.has_dimensionless_units:
            raise TypeError('Can only apply `arccos` function to dimensionless quantities')

        if issubclass(type(other), BaseImage):
            # Handle BaseImage (or subclass) instance
            if not other.has_dimensionless_units:
                raise TypeError('Can only apply `arccos` function to dimensionless quantities')

            otherData = other.data

            # Grab the uncertainty of the other image if possible
            if other._BaseImage__fullData.uncertainty is not None:
                otherUncert = other._BaseImage__fullData.uncertainty
            else:
                otherUncert = 0.0

        elif issubclass(type(other), u.Quantity):
            # Handle Quantity instance
            if not other.has_dimensionless_units:
                raise TypeError('Can only apply `arccos` function to dimensionless quantities')

            otherData   = other.value

            # Assume zero uncertainty
            otherUncert = 0.0
        elif issubclass(type(other),
            (int, np.int8, np.int16, np.int32, np.int64,
            float, np.float16, np.float32, np.float64)):
            # Add a unitless scalar quantity (if image is unitless)
            otherData   = other

            # Assume zero uncertainty
            otherUncert = 0.0
        else:
            # Incompatible types and/or units
            raise TypeError('Cannot compute arctan {0} with {1} units and {2}'.format(
                type(self).__name__, str(self.unit), type(other).__name__))

        # Grab the uncertainty of this image
        if self._BaseImage__fullData.uncertainty is not None:
            selfUncert = self._BaseImage__fullData.uncertainty
        else:
            selfUncert = 0.0

        # Compute the smart arctan2(x,y)
        outData = np.arctan2(selfData, otherData)

        # Check if the uncertainty is "zero"
        if selfUncert > 0 or otherUncert > 0:
            # Compute the propagated uncertainty
            # d/dx (arctan2(x,y)) = +x/(x^2 + y^2)
            # d/dy (arctan2(x,y)) = -y/(x^2 + y^2)
            d_arctan_dx = +selfData  / (selfData**2 + otherData**2)
            d_arctan_dy = -otherData / (selfData**2 + otherData**2)
            outUncert   = StdDevUncertainty(np.sqrt(
                (d_arctan_dx * selfUncert)**2 +
                (d_arctan_dy * otherUncert)**2
            ))
        else:
            outUncert = None

        # Copy the image and store the output
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(
            outData,
            uncertainty=outUncert,
            unit=u.rad,
            wcs=self._BaseImage__fullData.wcs
        )

        return outImg

    def sqrt(self):
        '''Computes the square root of the image data values'''
        # Grab the data
        selfData = self.data
        outData  = np.sqrt(selfData)

        # Propagate uncertainty if it exists
        if self._BaseImage__fullData.uncertainty is not None:
            selfUncert = self.uncertainty
            outUncert  = StdDevUncertainty(selfUncert/(2*outData))
        else:
            outUncert = None

        # Compute the arcsin and store the propagated uncertainty
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(
            outData,
            uncertainty=outUncert,
            unit=np.sqrt(self.unit),
            wcs=self._BaseImage__fullData.wcs
        )

        return outImg


    def exp(self):
        '''Compute the exponential of the image data values'''
        # Grab the data if posible
        if not self.has_dimensionless_units:
            raise TypeError('Can only apply `exp` function to dimensionless quantities')

        selfData = self.data
        outData  = np.exp(selfData)

        # Propagate uncertainty if it exists
        if self._BaseImage__fullData.uncertainty is not None:
            selfUncert = self.uncertainty
            outUncert  = StdDevUncertainty(selfUncert*outData)
        else:
            outUncert = None

        # Compute the arcsin and store the propagated uncertainty
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(
            outData,
            uncertainty=outUncert,
            unit=u.dimensionless_unscaled,
            wcs=self._BaseImage__fullData.wcs
        )

        return outImg

    def log(self):
        '''Computes the natural log of the image data values'''
        # Grab the data if posible
        if not self.has_dimensionless_units:
            raise TypeError('Can only apply `log` function to dimensionless quantities')

        selfData = self.data
        outData  = np.log(selfData)

        # Propagate uncertainty if it exists
        if self._BaseImage__fullData.uncertainty is not None:
            selfUncert = self.uncertainty
            outUncert  = StdDevUncertainty(selfUncert/selfData)
        else:
            outUncert = None

        # Compute the arcsin and store the propagated uncertainty
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(
            outData,
            uncertainty=outUncert,
            unit=u.dimensionless_unscaled,
            wcs=self._BaseImage__fullData.wcs
        )

        return outImg

    def log10(self):
        '''Computes the base-10 log of the image data values'''
        # Grab the data if posible
        if not self.has_dimensionless_units:
            raise TypeError('Can only apply `log10` function to dimensionless quantities')

        selfData = self.data
        outData  = np.log10(selfData)

        # Propagate uncertainty if it exists
        if self._BaseImage__fullData.uncertainty is not None:
            selfUncert = self.uncertainty
            outUncert  = StdDevUncertainty(selfUncert/(selfData*np.log(10)))
        else:
            outUncert = None

        # Compute the arcsin and store the propagated uncertainty
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(
            outData,
            uncertainty=outUncert,
            unit=u.dimensionless_unscaled,
            wcs=self._BaseImage__fullData.wcs
        )

        return outImg

    ##################################
    ### END OF NUMERICAL METHODS   ###
    ##################################
