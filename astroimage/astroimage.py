import pdb
import os
import copy
import numpy as np
from scipy import signal
from scipy import ndimage
from scipy.ndimage.filters import median_filter
import scipy.optimize as opt
import scipy.stats
import matplotlib as mpl
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from astropy.io import fits

# TODO
import warnings
# Catch the "RuntimeWarning" in the magic methods
# Catch the "UserWarning" in the init procedure

from astropy.wcs import WCS
from wcsaxes import WCSAxes
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.wcs.utils import proj_plane_pixel_scales, proj_plane_pixel_area
from photutils import DAOStarFinder

# Finally import the associated "utils" python module
from . import utils

class AstroImage(object):
    """An object which stores an image array and header and provides a
    read method and an overscan correction method.
    """

    def __init__(self, filename=''):
        if len(filename) > 0:
            # Replace dot notation
            # (for some reason the leading dot was causing errors...)
            if filename[0:2] == ('.' + os.path.sep):
                filename = os.path.join(os.getcwd(), filename[2:])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    HDUlist = fits.open(filename, do_not_scale_image_data=True)
            except:
                raise FileNotFoundError('File {0} does not exist'.format(filename))

            # Read in the header and store it in an attribute.
            self.header = HDUlist[0].header.copy()

            # Test if there is any WCS present in this header
            # NOTE: IT WOULD SEEM THAT *THIS* IS WHAT IS CAUSING THE 'FK5' ERROR
            #
            # WARNING: FITSFixedWarning: RADECSYS= 'FK5 '
            # the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]
            #
            thisWCS = WCS(self.header)
            if thisWCS.has_celestial:
                self.wcs = thisWCS

            # Parse the number of bits used for each pixel
            floatFlag   = self.header['BITPIX'] < 0
            numBits     = np.abs(self.header['BITPIX'])

            # Determine the appropriate data type for the array.
            if floatFlag:
                if numBits >= 64:
                    dataType = np.float64
                else:
                    dataType = np.float32
            else:
                if numBits >= 64:
                    dataType = np.int64
                elif numBits >= 32:
                    dataType = np.int32
                else:
                    dataType = np.int16

            # Store the data as the correct data type
            self.arr = HDUlist[0].data.astype(dataType, copy = False)

            # If binning has been specified, then set it...
            if ('CRDELT1' in self.header) or ('CRDELT2' in self.header):
                # Check that binning makes sense and store it if it does
                self.binning = tuple([int(di) for di in self.header['CRDELT*'].values()])
            else:
                # No binning found, so call this (1x1) binning, and add that
                # information to the image header under the CRDELT keyword.
                self.binning = tuple(np.ones(self.arr.ndim).astype(int))
                for i, di in enumerate(self.binning):
                    self.header['CRDELT'+str(i)] = di

            # Loop through the HDUlist and check for a 'SIGMA' HDU
            for HDU in HDUlist:
                if HDU.name.upper() == 'SIGMA':
                    self.sigma = HDU.data

            # Set the file type properties
            self.filename = filename
            self.dtype    = dataType
            HDUlist.close()
            #
            # I do not seem to be able to catch the "userwarning"
            # caused by loading FITS files into Windows.
            #
            # # This the "UserWarning" error from Astropy in Windows
            # # should be ignored until further notice
            # except UserWarning:
            #     pass

    def __pos__(self):
        # Implements behavior for unary positive (e.g. +some_object)
        outImg = self.copy()
        tmpArr = self.arr.copy()
        outImg.arr = +1*tmpArr

        return outImg

    def __neg__(self):
        # Implements behavior for negation (e.g. -some_object)
        outImg = self.copy()
        tmpArr = self.arr.copy()
        outImg.arr = -1*tmpArr

        return outImg

    def __abs__(self):
        # Implements behavior for the built in abs() function.
        outImg = self.copy()
        tmpArr = self.arr.copy()
        outImg.arr = abs(tmpArr)

        return outImg

    def __add__(self, other):
        # Implements addition.
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = (isinstance(other, int) or
                         isinstance(other, np.int8) or
                         isinstance(other, np.int16) or
                         isinstance(other, np.int32) or
                         isinstance(other, np.int64))
        oneIsFloat    = (isinstance(other, float) or
                         isinstance(other, np.float16) or
                         isinstance(other, np.float32) or
                         isinstance(other, np.float64))

        if bothAreImages:
            # Check that image shapes make sense
            shape1     = self.arr.shape
            shape2     = other.arr.shape
            if shape1 == shape2:
                output     = self.copy()
                output.arr = self.arr + other.arr

                # Attempt to propagate errors
                # Check which images have sigma arrays
                selfSig  = hasattr(self, 'sigma')
                otherSig = hasattr(other, 'sigma')
                # Either add in quadrature,
                # or just give the existing sigma to the output
                if selfSig and otherSig:
                    output.sigma = np.sqrt(self.sigma**2 + other.sigma**2)
                elif selfSig and not otherSig:
                    output.sigma = self.sigma
                elif not selfSig and otherSig:
                    output.sigma = other.sigma
            else:
                print('Cannot add images with different shapes')
                return None

        elif oneIsInt or oneIsFloat:
            output     = self.copy()
            output.arr = self.arr + other

        else:
            raise ValueError('Images can only operate with integers and floats.')

        # Return the added image
        return output

    def __radd__(self, other):
        # Implements reverse addition.
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        # Implements subtraction.
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = (isinstance(other, int) or
                         isinstance(other, np.int8) or
                         isinstance(other, np.int16) or
                         isinstance(other, np.int32) or
                         isinstance(other, np.int64))
        oneIsFloat    = (isinstance(other, float) or
                         isinstance(other, np.float16) or
                         isinstance(other, np.float32) or
                         isinstance(other, np.float64))

        if bothAreImages:
            # Check that image shapes make sense
            shape1     = self.arr.shape
            shape2     = other.arr.shape
            if shape1 == shape2:
                output     = self.copy()
                output.arr = self.arr - other.arr
            else:
                print('Cannot subtract images with different shapes')
                return None

            # Attempt to propagate errors
            # Check which images have sigma arrays
            selfSig  = hasattr(self, 'sigma')
            otherSig = hasattr(other, 'sigma')
            # Either add in quadrature,
            # or just give the existing sigma to the output
            if selfSig and otherSig:
                output.sigma = np.sqrt(self.sigma**2 + other.sigma**2)
            elif selfSig and not otherSig:
                output.sigma = self.sigma
            elif not selfSig and otherSig:
                output.sigma = other.sigma

        elif oneIsInt or oneIsFloat:
            output     = self.copy()
            output.arr = self.arr - other
        else:
            raise ValueError('Images can only operate with integers and floats.')

        # Return the subtracted image
        return output

    def __rsub__(self, other):
        # Implements reverse subtraction.
        if other == 0:
            return self
        else:
            output = self.__sub__(other)
            output.arr = -output.arr
            return output

    def __mul__(self, other):
        # Implements multiplication.
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = (isinstance(other, int) or
                         isinstance(other, np.int8) or
                         isinstance(other, np.int16) or
                         isinstance(other, np.int32) or
                         isinstance(other, np.int64))
        oneIsFloat    = (isinstance(other, float) or
                         isinstance(other, np.float16) or
                         isinstance(other, np.float32) or
                         isinstance(other, np.float64))
        otherIsArray  = isinstance(other, np.ndarray)

        if bothAreImages:
            # Check that image shapes make sense
            shapeSelf  = self.arr.shape
            shapeOther = other.arr.shape
            if shapeSelf == shapeOther:
                output     = self.copy()
                output.arr = self.arr * other.arr
            else:
                raise ValueError('Cannot multiply images with different shapes')

            # Attempt to propagate errors
            # Check which images have sigma arrays
            selfSig  = hasattr(self, 'sigma')
            otherSig = hasattr(other, 'sigma')
            # Either add in quadrature,
            # or just give the existing sigma to the output
            if selfSig and otherSig:
                output.sigma = np.sqrt((other.arr*self.sigma)**2 +
                                       (self.arr*other.sigma)**2)
            elif selfSig and not otherSig:
                output.sigma = self.sigma
            elif not selfSig and otherSig:
                output.sigma = other.sigma

        elif oneIsInt or oneIsFloat:
            output     = self.copy()
            output.arr = self.arr * other

            if hasattr(self, 'sigma'):
                output.sigma = np.abs(output.sigma * other)

        elif otherIsArray:
            output = self.copy()
            output.arr = self.arr * other
            return output

        else:
            raise ValueError('Cannot multiply AstroImage by {0}'.format(type(other)))

        # Retun the multiplied image
        return output

    def __rmul__(self, other):
        # Implements reverse multiplication.
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = (isinstance(other, int) or
                         isinstance(other, np.int8) or
                         isinstance(other, np.int16) or
                         isinstance(other, np.int32) or
                         isinstance(other, np.int64))
        oneIsFloat    = (isinstance(other, float) or
                         isinstance(other, np.float16) or
                         isinstance(other, np.float32) or
                         isinstance(other, np.float64))
        otherIsArray  = isinstance(other, np.ndarray)

        if bothAreImages or oneIsInt or oneIsFloat:
            return self.__mul__(other)

        elif otherIsArray:
            # THIS SECTION OF CODE IS NEVER EXECUTED BECAUSE IF THE LEFT-HAND
            # OBJECT IS AN numpy.ndarray THEN THE numpy.ndarray__mul__() METHOD
            # IS INVOKED. THE PROBLEM WITH THAT METHOD IS THAT IT SEEMS TO LOOP
            # OVER EACH ELEMENT OF THE ARRAY AND ATTEMPTS TO STORE GENERATE A
            # WHOLE ARRAY OF AstroImage OBJECTS. NEEDLESS TO SAY, THAT EATS UP
            # MEMORY **** CRAZY FAST ****
            outImg = self.copy()
            outImg.arr = self.arr * other
            return outImg

        else:
            raise ValueError('Cannot multiply AstroImage by {0}'.format(type(other)))

    def __div__(self, other):
        # Implements division using the / operator.
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = (isinstance(other, int) or
                         isinstance(other, np.int8) or
                         isinstance(other, np.int16) or
                         isinstance(other, np.int32) or
                         isinstance(other, np.int64))
        oneIsFloat    = (isinstance(other, float) or
                         isinstance(other, np.float16) or
                         isinstance(other, np.float32) or
                         isinstance(other, np.float64))
        oneIsArray    = isinstance(other, np.ndarray)

        # TODO
        # I should include the possibility of operating with numpy array
        # NOTE
        # IT WOULD SEEM THAT IT IS IMPOSSIBLE TO ACCURATELY HANDLE THE CASE
        #
        # numpy.ndarray * AstroImage
        #
        # because the numpy.ndarray methods control the multiplication, and they
        # loop through each element of the array while performing
        # multiplication, so from Python's perspective, you're performing
        # multiple instances of
        #
        # int * AstroImage
        #
        # And storing the results in the numpy.ndarray. Obviously, this will
        # occupy a tremendous amount of space, so attempting to use this case
        # should be avoided at all costs!!!
        #
        # It is unclear if it is possible to actually detect when the user
        # attempts to compute
        #
        # numpy.ndarray * AstroImage

        if bothAreImages:
            # Initalize some variables to check if zeros need to be handled
            catchSelfZeros  = False
            catchOtherZeros = False

            # Check that image shapes make sense
            shape1     = self.arr.shape
            shape2     = other.arr.shape
            if shape1 == shape2:
                output     = self.copy()

                # Catch division by zero
                other0Inds = np.where(other.arr == 0)
                if len(other0Inds[0]) > 0:
                    other0s = other.arr[other0Inds]
                    other.arr[other0Inds] = 1.0
                    catchOtherZeros = True

                # Do the division
                output.arr = self.arr / other.arr

            else:
                raise ValueError('Cannot divide images with different shapes')

            # Attempt to propagate errors
            # Check which images have sigma arrays
            selfSig  = hasattr(self, 'sigma')
            otherSig = hasattr(other, 'sigma')
            # Either add in quadrature,
            # or just give the existing sigma to the output
            if selfSig and otherSig:
                # Catch division by zero
                self0Inds = np.where(self.arr == 0)
                if len(self0Inds) > 0:
                    self0s = self.arr[self0Inds]
                    self.arr[self0Inds] = 1.0
                    catchSelfZeros = True

                output.sigma = np.abs(output.arr *
                               np.sqrt((self.sigma/self.arr)**2 +
                                       (other.sigma/other.arr)**2))
            elif selfSig and not otherSig:
                output.sigma = self.sigma
            elif not selfSig and otherSig:
                output.sigma = other.sigma

            # Replace zeros
            if catchSelfZeros > 0:
                self.arr[self0Inds] = self0s
            if catchOtherZeros > 0:
                other.arr[other0Inds] = other0s
                output.arr[other0Inds] = np.nan

        elif oneIsInt or oneIsFloat:
            output     = self.copy()
            output.arr = self.arr / other

            if hasattr(self, 'sigma'):
                output.sigma = self.sigma / other

        else:
            raise ValueError('Images can only operate with integers and floats.')

        # Return the divided image
        return output

    def __rdiv__(self, other):
        # Implements reverse multiplication.
        if other == 0:
            output = self.copy()
            output.arr = np.zeros(self.arr.shape)
            return output
        else:
            output = self.copy()
            output.arr = np.ones(self.arr.shape)
            output = output.__mul__(other)
            output = output.__div__(self)
            return output

    def __truediv__(self, other):
        # Implements true division (converting to float).
        # TODO catch division by zero (how to handle it?)
        return self.__div__(other)

    def __rtruediv__(self, other):
        # Implements reverse true division (converting to float).
        #TODO catch division by zero (how to handle it?)
        if other == 0:
            output = self.copy()
            output.arr = np.zeros(self.arr.shape)
            return output
        else:
            return self.__rdiv__(other)

    def __mod__(self, other):
        # Compute the mod of two images or an image and a number
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = (isinstance(other, int) or
                         isinstance(other, np.int8) or
                         isinstance(other, np.int16) or
                         isinstance(other, np.int32) or
                         isinstance(other, np.int64))
        oneIsFloat    = (isinstance(other, float) or
                         isinstance(other, np.float16) or
                         isinstance(other, np.float32) or
                         isinstance(other, np.float64))

        # Copy the self image for manipulation
        output = self.copy()

        # Check for stupid values
        badVals = np.logical_not(np.isfinite(self.arr))
        badInds = np.where(badVals)

        # Replace stpud values with zeros
        if len(badInds[0]) > 0:
            output.arr[badInds] = 0.0

        if bothAreImages:
            shape1 = self.arr.shape
            shape2 = other.arr.shape
            if shape1 == shape2:
                # Make a copy of the other image for manipulation
                modulo = other.copy()

                # Check for stupid values in the OTHER image
                bad_O_Vals = np.logical_not(np.isfinite(modulo.arr))
                bad_O_Inds = np.where(bad_O_Vals)

                # Replace stpud values with zeros
                if len(bad_O_Inds[0]) > 0:
                    modulo.arr[badInds] = 0.0

                output.arr = output.arr % modulo.arr
            else:
                print('Cannot modulate images with different shapes')
                return None
        elif oneIsInt or oneIsFloat:
            # Perform the actual modulation
            output.arr = output.arr % other

        # Return the bad values to nans (since that's really what they are)
        if len(badInds[0]) > 0:
            output.arr[badInds] = np.nan

        # Return the modulated image to the user
        return output


    def __pow__(self, other):
        # Implements raising the image to some arbitrary power
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = (isinstance(other, int) or
                         isinstance(other, np.int8) or
                         isinstance(other, np.int16) or
                         isinstance(other, np.int32) or
                         isinstance(other, np.int64))
        oneIsFloat    = (isinstance(other, float) or
                         isinstance(other, np.float16) or
                         isinstance(other, np.float32) or
                         isinstance(other, np.float64))

        if bothAreImages:
            # Check that image shapes make sense
            shape1     = self.arr.shape
            shape2     = other.arr.shape
            if shape1 == shape2:
                output     = self.copy()
                output.arr = (self.arr)**(other.arr)
            else:
                print('Cannot exponentiate images with different shapes')
                return None

            # Attempt to propagate errors
            # Check which images have sigma arrays
            selfSig  = hasattr(self, 'sigma')
            otherSig = hasattr(other, 'sigma')
            # Either add in quadrature,
            # or just give the existing sigma to the output
            if selfSig and otherSig:
                output.sigma = np.abs(output.arr) * np.sqrt(
                    (other.arr*(self.sigma/self.arr))**2 +
                    (np.log(self.arr)*other.sigma)**2)
            elif selfSig and not otherSig:
                # The line below explodes when self.arr == 0
                # output.sigma = np.abs(output.arr*other.arr*self.sigma/self.arr)
                output.sigma = np.abs((self.arr**(other.arr - 1.0))*self.sigma)
            elif not selfSig and otherSig:
                output.sigma = np.abs(other.arr)

        elif oneIsInt or oneIsFloat:
            output = self.copy()
            output.arr = self.arr**other

            # propagate errors asuming ints and floats have no errors
            if hasattr(self, 'sigma'):
                # The line below explodes when self.arr == 0
                # output.sigma = np.abs(output.arr*other*self.sigma/self.arr)
                output.sigma = np.abs(other*(self.arr**(other - 1.0))*self.sigma)
        else:
            print('Unexpected value in raising image to a power')

        # Finall return the exponentiated image
        return output

    # def __rpow__(self, other):
    #     # Implements raising the image to some arbitrary power
    #     bothAreImages = isinstance(other, self.__class__)
    #     oneIsInt      = (isinstance(other, int) or
    #                      isinstance(other, np.int8) or
    #                      isinstance(other, np.int16) or
    #                      isinstance(other, np.int32) or
    #                      isinstance(other, np.int64))
    #     oneIsFloat    = (isinstance(other, float) or
    #                      isinstance(other, np.float16) or
    #                      isinstance(other, np.float32) or
    #                      isinstance(other, np.float64))
    #
    #     if bothAreImages:
    #         # Check that image shapes make sense
    #         shape1     = self.arr.shape
    #         shape2     = other.arr.shape
    #         if shape1 == shape2:
    #             output     = self.copy()
    #             output.arr = (self.arr)**(other.arr)
    #         else:
    #             print('Cannot exponentiate images with different shapes')
    #             return None
    #
    #         # Attempt to propagate errors
    #         # Check which images have sigma arrays
    #         selfSig  = hasattr(self, 'sigma')
    #         otherSig = hasattr(other, 'sigma')
    #         # Either add in quadrature,
    #         # or just give the existing sigma to the output
    #         if selfSig and otherSig:
    #             output.sigma = np.abs(output.arr) * np.sqrt(
    #                 (other.arr*(self.sigma/self.arr))**2 +
    #                 (np.log(self.arr)*other.sigma)**2)
    #         elif selfSig and not otherSig:
    #             # The line below explodes when self.arr == 0
    #             # output.sigma = np.abs(output.arr*other.arr*self.sigma/self.arr)
    #             output.sigma = np.abs((self.arr**(other.arr - 1.0))*self.sigma)
    #         elif not selfSig and otherSig:
    #             output.sigma = np.abs(other.arr)
    #
    #     elif oneIsInt or oneIsFloat:
    #         output = self.copy()
    #         output.arr = self.arr**other
    #
    #         # propagate errors asuming ints and floats have no errors
    #         if hasattr(self, 'sigma'):
    #             # The line below explodes when self.arr == 0
    #             # output.sigma = np.abs(output.arr*other*self.sigma/self.arr)
    #             output.sigma = np.abs(other*(self.arr**(other - 1.0))*self.sigma)
    #     else:
    #         print('Unexpected value in raising image to a power')
    #
    #     # Finall return the exponentiated image
    #     return output

    ##################################
    ### END OF MAGIC METHODS       ###
    ### BEGIN OTHER COMMON METHODS ###
    ##################################
    def rad2deg(self):
        '''A simple method for converting the image array values from radians
        to degrees.
        '''

        outImg = self.copy()
        outImg.arr = np.rad2deg(self.arr)

        if hasattr(self, 'sigma'):
            outImg.sigma = np.abs(np.rad2deg(self.sigma))

        return outImg

    def deg2rad(self):
        '''A simple method for converting the image array values from degrees
        to radians.
        '''
        outImg = self.copy()
        outImg.arr = np.deg2rad(self.arr)

        if hasattr(self, 'sigma'):
            outImg.sigma = np.deg2rad(self.sigma)

        return outImg

    def sin(self):
        '''A simple method for computing the sine of the image and propagating
        its uncertainty (if a sigma array has been defined)
        '''
        outImg = self.copy()
        outImg.arr = np.sin(self.arr)

        if hasattr(self, 'sigma'):
            outImg.sigma = np.abs(np.cos(self.arr)*self.sigma)

        return outImg

    def arcsin(self):
        '''A simple method for computing the arcsin of the image and
        propagating its uncertainty (if a sigma array has been defined)
        '''
        outImg = self.copy()
        outImg.arr = np.arcsin(self.arr)

        if hasattr(self, 'sigma'):
            outImg.sigma = np.abs(self.sigma/(np.sqrt(1.0 - self.arr)))

        return outImg

    def cos(self):
        '''A simple method for computing the cosine of the image and propagating
        its uncertainty (if a sigma array has been defined)
        '''
        outImg = self.copy()
        outImg.arr = np.cos(self.arr)

        if hasattr(self, 'sigma'):
            outImg.sigma = np.abs(np.sin(self.arr)*self.sigma)

        return outImg

    def arccos(self):
        '''A simple method for computing the arccos of the image and
        propagating its uncertainty (if a sigma array has been defined)
        '''
        outImg = self.copy()
        outImg.arr = np.arccos(self.arr)

        if hasattr(self, 'sigma'):
            outImg.sigma = np.abs(self.sigma/(np.sqrt(1.0 - self.arr)))

        return outImg

    def tan(self):
        '''A simple method for computing the tangent of the image and
        propagating its uncertainty (if a sigma array has been defined)
        '''
        outImg = self.copy()
        outImg.arr = np.tan(self.arr)

        if hasattr(self, 'sigma'):
            outImg.sigma = np.abs(self.sigma*(np.cos(self.arr)**(-2)))

        return outImg

    def arctan(self):
        '''A simple method for computing the arctan of the image and
        propagating its uncertainty (if a sigma array has been defined)
        '''
        outImg = self.copy()
        outImg.arr = np.arctan(self.arr)

        if hasattr(self, 'sigma'):
            outImg.sigma = np.abs(self.sigma/(1.0 - self.arr))

        return outImg

    def arctan2(self, other):
        '''A simple method for computing the unambiguous arctan of the image
        and propagating its uncertainty (if a sigma array has been defined).
        The "self" instance is treated as the y value. Another image, array, or
        scalar value must be passed as an argument to the method. This will be
        treated as the x value.
        '''
        # Check what type of variable has been passed as the x argument.
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = (isinstance(other, int) or
                         isinstance(other, np.int8) or
                         isinstance(other, np.int16) or
                         isinstance(other, np.int32) or
                         isinstance(other, np.int64))
        oneIsFloat    = (isinstance(other, float) or
                         isinstance(other, np.float16) or
                         isinstance(other, np.float32) or
                         isinstance(other, np.float64))

        if bothAreImages:
            # Check that image shapes make sense
            shape1     = self.arr.shape
            shape2     = other.arr.shape
            if shape1 == shape2:
                outImg     = self.copy()
                outImg.arr = np.arctan2(self.arr, other.arr)

                # Perform error propagation
                selfSig  = hasattr(self, 'sigma')
                otherSig = hasattr(other, 'sigma')
                if selfSig and otherSig:
                    outImg.sigma = (np.sqrt((self.arr*other.sigma)**2 +
                                            (other.arr*self.sigma)**2) /
                                    (other.arr**2 + self.arr**2))
                elif selfSig and not xSig:
                    outImg.sigma = ((other.arr*self.sigma) /
                                    (other.arr**2 + self.arr**2))
                elif not selfSig and xSig:
                    outImg.sigma = ((self.arr*other.sigma) /
                                    (other.arr**2 + self.arr**2))

            else:
                print('Cannot arctan2 images with different shapes')
                return None

        elif oneIsInt or oneIsFloat:
            outImg     = self.copy()
            outImg.arr = np.arctan2(self.arr, other)

            # Perform error propagation
            if hasattr(self, 'sigma'):
                outImg.sigma = ((other.arr*self.sigma) /
                                (other.arr**2 + self.arr**2))
        else:
            raise ValueError('Images can only operate with integers and floats.')

        return outImg

    def sqrt(self):
        '''A simple method for computing the square root of the image and
        propagating its uncertainty (if a sigma array has been defined)
        '''

        return self**(0.5)

    def exp(self):
        '''A simple method for computing the expontial of the image and
        propagating its uncertainty (if a sigma array has been defined).
        '''
        outImg = self.copy()
        outImg.arr = np.exp(self.arr)

        if hasattr(self, 'sigma'):
            outImg.sigma = np.abs(self.sigma*self.arr)

        return outImg

    def log(self):
        '''A simple method for computing the natural log of the image and
        propagating its uncertainty (if a sigma array has been defined).
        '''
        outImg = self.copy()
        outImg.arr = np.log(self.arr)

        if hasattr(self, 'sigma'):
            outImg.sigma = np.abs(self.sigma/self.arr)

        return outImg

    def log10(self):
        '''A simple method for computing the base-10 log of the image and
        propagating its uncertainty (if a sigma array has been defined).
        '''
        outImg = self.copy()
        outImg.arr = np.log10(self.arr)

        if hasattr(self, 'sigma'):
            outImg.sigma = np.abs(self.sigma/(np.log(10)*self.arr))

        return outImg

    def copy(self):
        # Make a copy of the image and return it to the user
        output = copy.deepcopy(self)

        return output

    def write(self, filename = '', dtype = None, clobber=True):
        # Test if a filename was provided and default to current filename
        if len(filename) == 0:
            filename = self.filename

        # Compute booleans for which output data is present
        hasArr  = hasattr(self, 'arr')
        hasHead = hasattr(self, 'header')
        hasSig  = hasattr(self, 'sigma')

        # Make copies of the output data
        if hasArr:
            outArr = self.arr.copy()
        if hasHead:
            outHead = self.header.copy()
        if hasSig:
            outSig = self.sigma.copy()


        # If a data type was specified, recast output data into that format
        if dtype is not None:
            # First convert the array data
            try:
                outArr = outArr.astype(dtype)
            except:
                raise ValueError('dtype not recognized')

            # Next convert the sigma data if it exists
            if hasSig:
                outSig = outSig.astype(dtype)

            # Now update the header to include that information
            # First parse which bitpix value is required
            # define BYTE_IMG      8  /*  8-bit unsigned integers */
            # define SHORT_IMG    16  /* 16-bit   signed integers */
            # define LONG_IMG     32  /* 32-bit   signed integers */
            # define LONGLONG_IMG 64  /* 64-bit   signed integers */
            # define FLOAT_IMG   -32  /* 32-bit single precision floating point */
            # define DOUBLE_IMG  -64  /* 64-bit double precision floating point */
            isByte    = ((dtype is np.byte) or (dtype is np.int8))
            isInt16   = (dtype is np.int16)
            isInt32   = (dtype is np.int32)
            isInt64   = ((dtype is int) or (dtype is np.int64))
            isFloat32 = (dtype is np.float32)
            isFloat64 = ((dtype is float) or (dtype is np.float64))
            if isByte:
                bitpix = 8
            elif isInt16:
                bitpix = 16
            elif isInt32:
                bitpix = 32
            elif isInt64:
                bitpix = 64
            elif isFloat32:
                bitpix = -32
            elif isFloat64:
                bitpix = -64

            # Now update the header itself
            outHead['BITPIX'] = bitpix

        # Initalize an empty variable to make it possible to ignore the whole
        # shebang when there is NO array data and NO header data, etc...

        if hasArr:
            # Build a new HDU object to store the data
            arrHDU = fits.PrimaryHDU(data = outArr,
                                     do_not_scale_image_data=True)
        else:
            raise ValueError('There was no data in this AstroImage object.')

        if hasHead:
            # Replace the original header (since some cards may have been stripped)
            arrHDU.header = self.header

        # If there is a sigma attribute,
        # then include it in the list of HDUs
        if hasSig:
            # Bulid a secondary HDU
            sigmaHDU = fits.ImageHDU(data = outSig,
                                     name = 'sigma',
                                     do_not_scale_image_data=True)
            HDUs = [arrHDU, sigmaHDU]
        else:
            HDUs = [arrHDU]

        if HDUs is not None:
            # Build the final output HDUlist
            HDUlist = fits.HDUList(HDUs)

            # Write file to disk
            HDUlist.writeto(filename, clobber=clobber)
        else:
            raise ValueError('There was no data in this AstroImage object.')

    def get_plate_scales(self):
        """This is a convenience method for computing the plate scales along
        each axis, which can, in principle, be differente from eachother
        """
        return proj_plane_pixel_scales(self.wcs)

    def get_rotation(self):
        """This method will check if the image header has a celestial coordinate
        system and returns the rotation angle between the image pixels and the
        celestial coordinate system
        """
        # Check if it has a celestial coordinate system
        if hasattr(self, 'wcs'):
            # Grab the cd matrix
            cd = self.wcs.wcs.cd
            # Check if the frames have non-zero rotation
            if cd[0,0] != 0 or cd[1,1] != 0:
                # If a non-zero rotation was found, then compute rotation angles
                det  = cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0]
                sgn  = np.int(np.round(det/np.abs(det)))
                rot1 = np.rad2deg(np.arctan2(sgn*cd[0,1], sgn*cd[0,0]))
                rot2 = np.rad2deg(np.arctan2(-cd[1,0], cd[1,1]))

                # Check that rotations are within 2 degrees of eachother
                if np.abs(rot1 - rot2) < 2.0:
                    # Take an average rotation value
                    rotAng = 0.5*(rot1 + rot2)
                elif np.abs(rot1 - rot2 - 360.0) < 2.0:
                    rotAng = 0.5*(rot1 + rot2 - 360.0)
                elif np.abs(rot1 - rot2 + 360.0) < 2.0:
                    rotAng = 0.5*(rot1 + rot2 + 360.0)
                else:
                    raise ValueError('Rotation angles do not agree!')

                # Check if the longitude pole is located where expected
                if self.wcs.wcs.lonpole != 180.0:
                    rotAng += (180.0 - self.wcs.wcs.lonpole)

                # Now return the computed rotation angle
                return rotAng
            else:
                # No rotation was found, so just return a zero
                return 0.0
        else:
            return None

    def get_sources(self, satLimit=16000, crowdThresh=0, edgeThresh=0):
        """This method simply uses the daofind algorithm to extract source
        positions. It will also test for saturation using a default value of
        16000 ADU. It will also optionally test for crowded stars and omit them.
        """
        # Double check that edge-stars will be rejected before checking for
        # crowding...
        if edgeThresh < crowdThresh:
            edgeThresh = crowdThresh + 1

        # Ensure there are no problematic values
        self1   = self.copy()
        badPix  = np.logical_not(np.isfinite(self1.arr))
        if np.sum(badPix) > 0:
            badInds  = np.where(badPix)
            self1.arr[badInds] = np.nanmin(self1.arr)

        # Grab the image sky statistics
        mean, median, std = sigma_clipped_stats(self1.arr, sigma=3.0, iters=5)

        # Start by instantiating a DAOStarFinder object
        daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)

        # Use that object to find the stars in the image
        sources = daofind(np.nan_to_num(self1.arr) - median)

        # Grab the image shape for later use
        ny, nx = self1.arr.shape

        # Cut out edge stars if requested
        if edgeThresh > 0:
            nonEdgeStars = sources['xcentroid'] > edgeThresh
            nonEdgeStars = np.logical_and(nonEdgeStars,
                sources['xcentroid'] < nx - edgeThresh - 1)
            nonEdgeStars = np.logical_and(nonEdgeStars,
                sources['ycentroid'] > edgeThresh)
            nonEdgeStars = np.logical_and(nonEdgeStars,
                sources['ycentroid'] < ny - edgeThresh - 1)

            # Cull the sources list to only include non-edge stars
            if np.sum(nonEdgeStars) > 0:
                nonEdgeInds = np.where(nonEdgeStars)
                sources = sources[nonEdgeInds]
            else:
                raise IndexError('There are no non-edge stars')

        # Generate a map of pixel positions
        yy, xx = np.mgrid[0:ny, 0: nx]

        # Perform the saturation test
        notSaturated = []
        for source in sources:
            # Extract the position for this source
            xs, ys = source['xcentroid'], source['ycentroid']

            # Compute the distance from this source
            dists = np.sqrt((xx - xs)**2 + (yy - ys)**2)

            # Grab the values within 15 pixels of this source, and test if the
            # source is saturated
            nearInds = np.where(dists < 15.0)
            notSaturated.append(self1.arr[nearInds].max() < satLimit)

        # Cull the sources list to ONLY include non-saturated sources
        if np.sum(notSaturated) > 0:
            notSaturatedInds = np.where(notSaturated)
            sources = sources[notSaturatedInds]
        else:
            raise IndexError('No sources passed the saturation test')

        # Perform the crowding test
        isolatedSource = []
        if crowdThresh > 0:
            # Generate pixel positions for the patch_data
            yy, xx = np.mgrid[0:np.int(crowdThresh), 0:np.int(crowdThresh)]

            # Loop through the sources and test if they're crowded
            for source in sources:
                # Extract the posiition for this source
                xs, ys = source['xcentroid'], source['ycentroid']

                # Compute the distance between other sources and this source
                dists = np.sqrt((sources['xcentroid'] - xs)**2 +
                                (sources['ycentroid'] - ys)**2)

                # Test if there are any OTHER stars within crowdThresh
                isolatedBool1 = np.sum(dists < crowdThresh) < 2

                # Do a double check to see if there are any EXTRA sources
                # Start by cutting out the patch surrounding this star
                # Establish the cutout bondaries
                btCut = np.int(np.round(ys - np.floor(0.5*crowdThresh)))
                tpCut = np.int(np.round(btCut + crowdThresh))
                lfCut = np.int(np.round(xs - np.floor(0.5*crowdThresh)))
                rtCut = np.int(np.round(lfCut + crowdThresh))

                # Cut out that data and subtract the floor.
                patch_data  = self1.arr[btCut:tpCut,lfCut:rtCut]
                patch_data -= patch_data.min()

                # Null out data beyond the crowdThresh from the center
                xs1, ys1 = xs - lfCut, ys - btCut
                pixDist  = np.sqrt((xx - xs1)**2 + (yy - ys1)**2)
                nullInds = np.where(pixDist > crowdThresh)

                # Null those pixels
                patch_data[nullInds] = 0

                with warnings.catch_warnings():
                    # Ignore model linearity warning from the fitter
                    warnings.simplefilter('ignore')

                    # Start by instantiating a DAOStarFinder object
                    daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)

                    # Use that object to check for other sources in this patch
                    sources1 = daofind(patch_data)

                # Test if more than one source was found
                isolatedBool2 = len(sources1) < 2

                # Check if there are other sources nearby
                if isolatedBool1 and isolatedBool2:
                    isolatedSource.append(True)
                else:
                    isolatedSource.append(False)


            # Cull the sources list to ONLY include non-crowded sources
            if np.sum(isolatedSource) > 0:
                isolatedInds = np.where(isolatedSource)
                sources = sources[isolatedInds]
            else:
                raise IndexError('No sources passed the crowding test')

        # Grab the x, y positions of the sources and return as arrays
        xs, ys = sources['xcentroid'].data, sources['ycentroid'].data

        return xs, ys

    def get_psf(self):
        """This method analyses the stars in the image and returns the average
        PSF properties of the image. The default mode fits 2D-gaussians to the
        brightest, isolated stars in the image. Future versions could use there
        2MASS sersic profile, etc...
        """
        # Set the patch size to use in fitting gaussians (e.g. 20x20 pixels)
        starPatch = 20

        # Find the isolated sources for gaussian fitting
        crowdThresh = np.sqrt(2)*0.5*starPatch
        xsrcs, ysrcs = self.get_sources(
            crowdThresh = crowdThresh,
            edgeThresh = starPatch + 1)

        # Build a gaussian + 2Dpolynomial(1st degree) model to fit to patches
        # Build a gaussian model for fitting stars
        gauss_init = models.Gaussian2D(
            amplitude=1000.0,
            x_mean=10.0,
            y_mean=10.0,
            x_stddev=3.0,
            y_stddev=3.0,
            theta=0.0)
        bkg_init   = models.Polynomial2D(1)
        patch_init = gauss_init + bkg_init
        fitter     = fitting.LevMarLSQFitter()

        # Setup the pixel coordinates for the star patch
        yy, xx = np.mgrid[0:starPatch, 0:starPatch]

        # Loop through the sources and fit gaussians to each patch.
        # Store the background subtracted patches for a final average img.
        sxList    = []
        syList    = []
        patchList = []
        for xs, ys in zip(xsrcs, ysrcs):
            # Start by cutting out the patch surrounding this star
            # Establish the cutout bondaries
            btCut = np.int(np.round(ys - np.floor(0.5*starPatch)))
            tpCut = np.int(np.round(btCut + starPatch))
            lfCut = np.int(np.round(xs - np.floor(0.5*starPatch)))
            rtCut = np.int(np.round(lfCut + starPatch))

            # Cut out the star patch
            patch_data = self.arr[btCut:tpCut, lfCut:rtCut].copy()

            # Ignore model warning from the fitter
            with warnings.catch_warnings():
                # Fit the model to the patch
                warnings.simplefilter('ignore')
                patch_model = fitter(patch_init, xx, yy, patch_data)

            # Store the gaussian component eigen-values
            sxList.append(patch_model.x_stddev_0.value)
            syList.append(patch_model.y_stddev_0.value)

            # Build a 2D polynomial background to subtract
            bkg_model = models.Polynomial2D(1)

            # Transfer the background portion of the patch_model to the
            # polynomial plane model.
            bkg_model.c0_0 = patch_model.c0_0_1
            bkg_model.c1_0 = patch_model.c1_0_1
            bkg_model.c0_1 = patch_model.c0_1_1

            # Subtract and renormalize the star patch
            patch_data1  = patch_data - bkg_model(xx, yy)
            patch_data1 -= patch_data1.min()
            patch_data1 /= patch_data1.max()

            # Store the patch in the patchList
            patchList.append(patch_data1)

        # Convert the sxList, syList, and patchList into an arrays
        sxArr    = np.array(sxList)
        syArr    = np.array(syList)
        patchArr = np.array(patchList)

        # Test for good sx and sy eigen-values
        meanSX, medianSX, stdSX = sigma_clipped_stats(sxArr)
        meanSY, medianSY, stdSY = sigma_clipped_stats(syArr)

        # Find the indices of the good sx and sy values (well behaved gaussians)
        goodSX   = np.abs(sxArr - medianSX)/stdSX < 2.5
        goodSY   = np.abs(syArr - medianSY)/stdSY < 2.5
        goodSXSY = np.logical_and(goodSX, goodSY)

        # Cut out any patches with bad sx or bad sy values
        if np.sum(goodSXSY) > 0:
            goodInds = (np.where(goodSXSY))[0]
            patchArr = patchArr[goodInds, :, :]
        else:
            raise IndexError('There are no well fitted stars')

        # Compute an "median patch"
        patch_data = np.median(patchArr, axis=0)

        # Finllay, re-fit a gaussian to this median patch
        # Ignore model warning from the fitter
        with warnings.catch_warnings():
            # Fit the model to the patch
            warnings.simplefilter('ignore')
            patch_model = fitter(patch_init, xx, yy, patch_data)

        # Modulate the fitted theta value into a reasonable range
        goodTheta = (patch_model.theta_0.value % (2*np.pi))
        patch_model.theta_0 = goodTheta

        # Build a 2D polynomial background to subtract
        bkg_model = models.Polynomial2D(1)

        # Transfer the background portion of the patch_model to the
        # polynomial plane model.
        bkg_model.c0_0 = patch_model.c0_0_1
        bkg_model.c1_0 = patch_model.c1_0_1
        bkg_model.c0_1 = patch_model.c0_1_1

        # Subtract and renormalize the star patch
        patch_data1  = patch_data - bkg_model(xx, yy)
        patch_data1 -= patch_data1.min()
        patch_data1 /= np.sum(patch_data1)

        # Return the fitted PSF values
        sx, sy, theta = (patch_model.x_stddev_0.value,
                         patch_model.y_stddev_0.value,
                         patch_model.theta_0.value)

        return ({'sx':sx, 'sy':sy, 'theta':theta}, patch_data1)

    # def get_PSF_old(self, shape='gaussian'):
    #     """This method analyses the stars in the image and returns the PSF
    #     properties of the image. The default mode fits 2D-gaussians to the
    #     brightest, isolated stars in the image. Future versions could use there
    #     2MASS sersic profile, etc...
    #     """
    #     # Grab the image sky statistics
    #     mean, median, std = sigma_clipped_stats(self.arr, sigma=3.0, iters=5)
    #
    #     # Start by finding all the stars in the image
    #     sources = daofind(self.arr - median, fwhm=3.0, threshold=15.0*std)
    #
    #     # Eliminate stars near the image edge
    #     ny, nx = self.arr.shape
    #     xStars, yStars = sources['xcentroid'].data, sources['ycentroid'].data
    #     badXstars = np.logical_or(xStars < 50, xStars > nx - 50)
    #     badYstars = np.logical_or(yStars < 50,  yStars > ny - 50)
    #     edgeStars = np.logical_or(badXstars, badYstars)
    #     if np.sum(edgeStars) > 0:
    #         sources = sources[np.where(np.logical_not(edgeStars))]
    #
    #     # Eliminate any stars with neighbors within 30 pixels
    #     keepFlags = np.ones_like(sources['flux'].data, dtype='bool')
    #     for i, star in enumerate(sources):
    #         # Compute the distance between this star and other stars
    #         xs, ys = star['xcentroid'], star['ycentroid']
    #         dist = np.sqrt((sources['xcentroid'].data - xs)**2 +
    #                        (sources['ycentroid'].data - ys)**2)
    #
    #         # If there is another star within 20 pixels, then don't keep this
    #         if np.sum(dist < 30) > 1:
    #             keepFlags[i] = False
    #
    #     # Cull the list of sources to only include "isolated" stars
    #     if np.sum(keepFlags) > 0:
    #         sources = sources[np.where(keepFlags)]
    #     else:
    #         print('No sources survided the neighbor test')
    #         pdb.set_trace()
    #
    #     # Sort the stars by brightness
    #     sortInds = (sources['flux'].argsort())[::-1]
    #     sources = sources[sortInds]
    #
    #     def gauss2d(xy, base, height, center_x, center_y, width_x, width_y, rotation):
    #         """Returns a gaussian function with the given parameters"""
    #         # Parse the xy vector
    #         x, y     = xy
    #
    #         # Ensure the parameters are floats
    #         width_x = float(width_x)
    #         width_y = float(width_y)
    #
    #         xp = x - center_x
    #         yp = y - center_y
    #
    #         # Convert rotation to radians and apply the rotation matrix
    #         # to center coordinates
    #         rotation = np.deg2rad(rotation)
    #         # center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    #         # center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)
    #
    #         # Rotate the xy coordinates
    #         xp1 = xp * np.cos(rotation) - yp * np.sin(rotation)
    #         yp1 = xp * np.sin(rotation) + yp * np.cos(rotation)
    #
    #         # Compute the gaussian values
    #         g = base + height*np.exp(-((xp1/width_x)**2 + (yp1/width_y)**2)/2.)
    #         return g
    #
    #     # Fit 2D-gaussians to the 10 brightest, isolated stars
    #     yy, xx = np.mgrid[0:41, 0:41]
    #     sxList  = list()
    #     syList  = list()
    #     rotList = list()
    #     for i, star in enumerate(sources):
    #         # Fit a 2D gaussian to each star
    #         # First cut out a square array centered on the star
    #         xs, ys = star['xcentroid'], star['ycentroid']
    #         lf = np.int(xs.round()) - 20
    #         rt = lf + 41
    #         bt = np.int(ys.round()) - 20
    #         tp = bt + 41
    #         starArr = self.arr[bt:tp,lf:rt]
    #
    #         # Package xy, zobs for fitting
    #         xy   = np.array([xx.ravel(), yy.ravel()])
    #         zobs = starArr.ravel()
    #
    #         # Guess the initial parameters and perform the fit
    #         #              base, amp,   xc,   yc,   sx,  sy,  rot
    #         arrMax      = np.max(zobs)
    #         base1       = np.median(zobs)
    #         guessParams = [base1,    arrMax,   21.0,   21.0,   2.0,  2.0,   0.0]
    #         boundParams = ((-np.inf, 0.0,    -100.0, -100.0,   0.1,  0.1,   0.0),
    #                        (+np.inf, np.inf, +100.0, +100.0,  10.0, 10.0, 360.0))
    #         try:
    #             fitParams, uncert_cov = opt.curve_fit(gauss2d, xy, zobs,
    #                 p0=guessParams, bounds=boundParams)
    #         except:
    #             print("Star {0} could not be fit".format(i))
    #
    #         # Test goodness of fitCenter
    #         fitX, fitY = fitParams[2:4]
    #         fitCenter  = np.sqrt((21 - fitParams[2])**2 + (21 - fitParams[3])**2)
    #         goodStar   = ((fitCenter < 3.0) and
    #                       (fitParams[4] > 0.5) and (fitParams[4] < 3.0) and
    #                       (fitParams[5] > 0.5) and (fitParams[5]) < 3.0)
    #
    #         if goodStar:
    #             # Store the relevant parameters
    #             sxList.append(fitParams[4])
    #             syList.append(fitParams[5])
    #             rotList.append(fitParams[6])
    #
    #     # Compute an average gaussian shape and return to user
    #     PSFparams = (np.median(sxList),
    #                  np.median(syList),
    #                  np.arctan2(np.median(np.sin(rotList)),
    #                             np.median(np.cos(rotList))))
    #
    #     return PSFparams

    def overscan_correction(self, overscanPos, sciencePos,
                            overscanPolyOrder = 3):
        """Fits a polynomial to the overscan column and subtracts and extended
        polynomial from the entire image array.
        Note: The prescan region is not an accurate representation of the bias,
        so it is completely ignored.
        """
        # Check the binning
#        binning = np.unique((self.header['CRDELT1'], self.header['CRDELT2']))

        # Compute the rebinned locations of the pre-scan and post-scan regions.
        overscanPos1 = overscanPos/self.binning
        sciencePos1  = sciencePos/self.binning
#        overscanRegion = self.arr[overscanPos1[0][1]:overscanPos1[1][1], \
#                                  overscanPos1[0][0]:overscanPos1[1][0]]

        # Grab the pre-scan and post-scan regions of the array
        overscanRegion  = self.arr[:, overscanPos1[0][0]:overscanPos1[1][0]]

        # Make sure I know what is "PRESCAN" (right?) and "POSTSCAN" (left?)
        # and which one corresponds to the "overscan region"

        # Grab the shape of the array for future use
        ny, nx = self.arr.shape

        # Fit the correct order polynomial to the overscan columns
        overscanRowValues  = np.mean(overscanRegion, axis = 1)
        rowInds            = range(ny)
        overscanPolyCoeffs = np.polyfit(rowInds, overscanRowValues, overscanPolyOrder)
        overscanPolyValues = np.polyval(overscanPolyCoeffs, rowInds)

        # Expand the overscan along the horizontal axis and subtract.
        overscan  = (np.tile(overscanPolyValues, (self.arr.shape[1],1)).
                     astype(np.float32)).T
        self.arr  = self.arr.astype(np.float32)
        self.arr -= overscan

        # Trim the array to include only the science data
        self.arr  = self.arr[sciencePos1[0][1]:sciencePos1[1][1],
                             sciencePos1[0][0]:sciencePos1[1][0]]

    def scale(self, quantity='flux', copy=False):
        """Scales the data in the arr attribute using the BSCALE and BZERO
        values from the header. If no such values exist, then return original
        array.
        """
        # Scale the array
        scaledArr = self.arr.copy()
        if 'BSCALE' in self.header.keys():
            if quantity.upper() == 'FLUX':
                scaleConst1 = self.header['BSCALE']

                # Check for uncertainty in BSCALE
                if 'SBSCALE' in self.header.keys():
                    sig_scaleConst1 = self.header['SBSCALE']

            elif quantity.upper() == 'INTENSITY':
                pixArea     = proj_plane_pixel_area(self.wcs)*(3600**2)
                scaleConst1 = self.header['BSCALE']/pixArea

                # Check for uncertainty in BSCALE
                if 'SBSCALE' in self.header.keys():
                    sig_scaleConst1 = self.header['SBSCALE']/pixArea
        else:
            scaleConst1 = 1

        if 'BZERO' in self.header.keys():
            scaleConst0 = self.header['BZERO']
        else:
            scaleConst0 = 0

        # Perform the actual scaling!
        scaledArr = scaleConst1*self.arr.copy() + scaleConst0

        # Apply scaling uncertainty if available
        if hasattr(self, 'sigma'):
            # If there is an uncertainty in the scaling factor, then propagate
            # that into the uncertainty
            if 'SBSCALE' in self.header.keys():
                # Include the uncertainty in the scaling...
                sigArr = np.abs(scaledArr)*np.sqrt((self.sigma/self.arr)**2
                    + (sig_scaleConst1/scaleConst1)**2)
            else:
                # Otherwise just scale up the uncertainty...
                sigArr  = self.sigma.copy()
                sigArr *= scaleConst1

        # Check if a copy of the image was requested
        if copy:
            outImg = self.copy()
            outImg.arr = scaledArr

            # Try to store the sigArr array in the output image
            try:
                outImg.sigma = sigArr
            except:
                pass

            return outImg
        else:
            self.arr = scaledArr

            # Try to store the sigArr array in the original image
            try:
                self.sigma = sigArr
            except:
                pass

    def frebin(self, nx1, ny1, copy=False, total=False):
        """Rebins the image using a flux conservative method. If 'copy' is True,
        then the method will return a new copy of the image with its array
        rebinned. Otherwise, the image will be rebinned in place.
        """

        # First test for the trivial case
        ny, nx = self.arr.shape
        if (nx == nx1) and (ny == ny1):
            if copy:
                return self.copy()
            else:
                return

        # Compute the pixel ratios of upsampling and down sampling
        xratio, yratio = np.float(nx1)/np.float(nx), np.float(ny1)/np.float(ny)
        pixRatio       = np.float(xratio*yratio)
        aspect         = yratio/xratio         #Measures change in aspect ratio.

        if ((nx % nx1) == 0) and ((ny % ny1) == 0):
            # Handle integer downsampling
            # Get the new shape for the array and compute the rebinning shape
            sh = (ny1, ny//ny1,
                  nx1, nx//nx1)

            # Make a copy of the array before any manipulation
            tmpArr = (self.arr.copy()).astype(np.float)

            # Perform the actual rebinning
            rebinArr = tmpArr.reshape(sh).mean(-1).mean(1)

            # Check if total flux conservation was requested
            if total:
                # Re-normalize by pixel area ratio
                rebinArr /= pixRatio

        elif ((nx1 % nx) == 0) and ((ny1 % ny) == 0):
            # Handle integer upsampling
            # Make a copy of the array before any manipulation
            tmpArr = (self.arr.copy()).astype(np.float)

            # Perform the actual rebinning
            rebinArr   = np.kron(tmpArr, np.ones((ny1//ny, nx1//nx)))

            # Check if total flux conservation was requested
            if total:
                # Re-normalize by pixel area ratio
                rebinArr /= pixRatio

        else:
            # Handle the cases of non-integer rebinning
            # Make a copy of the array before any manipulation
            tmpArr = np.empty((ny1, nx), dtype=np.float)

            # Loop along the y-axis
            ybox, xbox = np.float(ny)/np.float(ny1), np.float(nx)/np.float(nx1)
            for i in range(ny1):
                # Define the boundaries of this box
                rstart = i*ybox
                istart = np.int(rstart)
                rstop  = rstart + ybox
                istop  = np.int(rstop) if (np.int(rstop) < (ny - 1)) else (ny - 1)
                frac1  = rstart - istart
                frac2  = 1.0 - (rstop - istop)

                # Compute the values in each box
                if istart == istop:
                    tmpArr[i,:] = (1.0 - frac1 - frac2)*self.arr[istart, :]
                else:
                    tmpArr[i,:] = (np.sum(self.arr[istart:istop+1, :], axis=0)
                                   - frac1*self.arr[istart, :]
                                   - frac2*self.arr[istop, :])

            # Transpose tmpArr and prepare to loop along other axis
            tmpArr = tmpArr.T
            result = np.empty((nx1, ny1))

            # Loop along the x-axis
            for i in range(nx1):
                # Define the boundaries of this box
                rstart = i*xbox
                istart = np.int(rstart)
                rstop  = rstart + xbox
                istop  = np.int(rstop) if (np.int(rstop) < (nx - 1)) else (nx - 1)
                frac1  = rstart - istart
                frac2  = 1.0 - (rstop - istop)

                # Compute the values in each box
                if istart == istop:
                    result[i,:] = (1.0 - frac1 - frac2)*tmpArr[istart, :]
                else:
                    result[i,:] = (np.sum(tmpArr[istart:istop+1, :], axis=0)
                                   - frac1*tmpArr[istart, :]
                                   - frac2*tmpArr[istop, :])

            # Transpose the array back to its proper numpy style shape
            rebinArr = result.T

            # Check if total flux conservation was requested
            if not total:
                rebinArr *= pixRatio

        # Check if there is a header needing modification
        if hasattr(self, 'header'):
            outHead = self.header.copy()

            # Update the NAXIS values
            outHead['NAXIS1'] = nx1
            outHead['NAXIS2'] = ny1

            # Update the CRPIX values
            outHead['CRPIX1'] = (self.header['CRPIX1'] + 0.5)*xratio - 0.5
            outHead['CRPIX2'] = (self.header['CRPIX2'] + 0.5)*yratio - 0.5
            if self.wcs.wcs.has_cd():
                # Attempt to use CD matrix corrections, first
                # Apply updates to CD valus
                thisCD = self.wcs.wcs.cd
                # TODO set CDELT value properly in the "astrometry" step
                outHead['CD1_1'] = thisCD[0,0]/xratio
                outHead['CD1_2'] = thisCD[0,1]/yratio
                outHead['CD2_1'] = thisCD[1,0]/xratio
                outHead['CD2_2'] = thisCD[1,1]/yratio
            elif self.wcs.wcs.has_pc():
                # Apply updates to CDELT valus
                outHead['CDELT1'] = outHead['CDELT1']/xratio
                outHead['CDELT2'] = outHead['CDELT2']/yratio

                # Adjust the PC matrix if non-equal plate scales.
                # See equation 187 in Calabretta & Greisen (2002)
                if aspect != 1.0:
                    outHead['PC1_1'] = outHead['PC1_1']
                    outHead['PC2_2'] = outHead['PC2_2']
                    outHead['PC1_2'] = outHead['PC1_2']/aspect
                    outHead['PC2_1'] = outHead['PC2_1']*aspect
        else:
            # If no header exists, then buil a basic one
            keywords = ['NAXIS2', 'NAXIS1']
            values   = (ny1, nx1)
            headDict = dict(zip(keywords, values))
            outHead  = fits.Header(headDict)

        if copy:
            # If a copy was requesty, then return a copy of the original image
            # with a newly rebinned array
            outImg         = self.copy()
            outImg.arr     = rebinArr
            outImg.header  = outHead
            # TODO make binning a (dx, dy) tuple
            outImg.binning = xratio * outImg.binning
            return outImg
        else:
            # Otherwise place the rebinned array directly into the Image object
            self.arr     = rebinArr
            self.header  = outHead
            # TODO make binning a (dx, dy) tuple
            self.binning = xratio * self.binning

    def rebin(self, nx1, ny1, copy=False, total=False):
        """Rebins the image using sigma attribute to produce a weighted average.
        The new image shape must be integer multiples of fractions of the
        original shape. Default behavior is to use inverse variance weighting
        for the average if a "sigma" attribute is present. Otherwise, simply
        do straight averaging or summing.

        copy  - [True, False] if set to true, then returns a copy of the image
                with a rebinned array. Otherwise, the image will be rebinned in
                place.
        total - [True, False] if set to true, then returned array is total of
                the binned pixels rather than the average.
        """
        # Grab the shape of the initial array
        ny, nx = self.arr.shape

        # Test for improper result shape
        goodX = ((nx % nx1) == 0) or ((nx1 % nx) == 0)
        goodY = ((ny % ny1) == 0) or ((ny1 % ny) == 0)
        if not (goodX and goodY):
            raise ValueError('Result dimensions must be integer factor of original dimensions')

        # First test for the trivial case
        if (nx == nx1) and (ny == ny1):
            if copy:
                return self.copy()
            else:
                return None

        # Compute the pixel ratios of upsampling and down sampling
        xratio, yratio = np.float(nx1)/np.float(nx), np.float(ny1)/np.float(ny)
        pixRatio       = np.float(xratio*yratio)
        aspect         = yratio/xratio         #Measures change in aspect ratio.

        if ((nx % nx1) == 0) and ((ny % ny1) == 0):
            # Handle integer downsampling
            # Get the new shape for the array and compute the rebinning shape
            sh = (ny1, ny//ny1,
                  nx1, nx//nx1)

            # Build the appropriate weights for the averaging procedure
            if hasattr(self, 'sigma'):
                # Catch the zeros uncertainty points and null them out.
                tmpSig = self.sigma.copy()
                zeroInds = np.where(self.sigma == 0)
                if len(zeroInds[0]) > 0:
                    tmpSig[zeroInds] = 1.0

                # Now actually compute the weights
                wts    = tmpSig**(-2)

                # Finally replace "zero-uncertainty" values with zero weight.
                if len(zeroInds[0]) > 0:
                    wts[zeroInds] = 0.0

            else:
                wts = np.ones_like(self.arr)

            # Build the weighted array
            tmpArr = wts*self.arr

            # Perform the actual rebinning
            # rebinWts1 = wts.reshape(sh).mean(-1).mean(1)
            rebinWts = wts.reshape(sh).sum(-1).sum(1)

            # Catch division by zero
            zeroInds   = np.where(rebinWts == 0)
            noZeroInds = np.where(
                np.logical_and(
                (rebinWts != 0),
                np.isfinite(rebinWts)))

            # Computed weighted rebinning
            rebinArr = (tmpArr.reshape(sh).sum(-1).sum(1))
            rebinArr[noZeroInds] /= rebinWts[noZeroInds]

            # Compute uncertainyt in weighted rebinning
            rebinSig = np.zeros(rebinArr.shape) + np.NaN
            rebinSig[noZeroInds] = np.sqrt(1.0/rebinWts[noZeroInds])

            # Check if total flux conservation was requested
            if total:
                # Re-normalize by pixel area ratio
                rebinArr /= pixRatio

                # Apply the same re-normalizing to the sigma array
                if hasattr(self, 'sigma'):
                    rebinSig /= pixRatio

        elif ((nx1 % nx) == 0) and ((ny1 % ny) == 0):
            # Handle integer upsampling
            rebinArr = np.kron(self.arr, np.ones((ny1//ny, nx1//nx)))
            if hasattr(self, 'sigma'):
                rebinSig  = np.kron(self.sigma, np.ones((ny1//ny, nx1//nx)))

            # Check if total flux conservation was requested
            if total:
                # Re-normalize by pixel area ratio
                rebinArr /= pixRatio

                if hasattr(self, 'sigma'):
                    rebinSig /= pixRatio

        # Check if there is a header needing modification
        if hasattr(self, 'header'):
            outHead = self.header.copy()

            # Update the NAXIS values
            outHead['NAXIS1'] = nx1
            outHead['NAXIS2'] = ny1

            # Update the CRPIX values
            outHead['CRPIX1'] = (self.header['CRPIX1'] + 0.5)*xratio + 0.5
            outHead['CRPIX2'] = (self.header['CRPIX2'] + 0.5)*yratio + 0.5
            if self.wcs.wcs.has_cd():
                # Attempt to use CD matrix corrections, first
                # Apply updates to CD valus
                thisCD = self.wcs.wcs.cd
                # TODO set CDELT value properly in the "astrometry" step
                outHead['CD1_1'] = thisCD[0,0]/xratio
                outHead['CD1_2'] = thisCD[0,1]/yratio
                outHead['CD2_1'] = thisCD[1,0]/xratio
                outHead['CD2_2'] = thisCD[1,1]/yratio
            elif self.wcs.wcs.has_pc():
                # Apply updates to CDELT valus
                outHead['CDELT1'] = outHead['CDELT1']/xratio
                outHead['CDELT2'] = outHead['CDELT2']/yratio

                # Adjust the PC matrix if non-equal plate scales.
                # See equation 187 in Calabretta & Greisen (2002)
                if aspect != 1.0:
                    outHead['PC1_1'] = outHead['PC1_1']
                    outHead['PC2_2'] = outHead['PC2_2']
                    outHead['PC1_2'] = outHead['PC1_2']/aspect
                    outHead['PC2_1'] = outHead['PC2_1']*aspect

            # Adjust BZERO and BSCALE for new pixel size, unless these values
            # are used to define unsigned integer data types.
            # TODO handle special cases of unsigned integers, where BSCALE may
            # be used to define integer data types.
            if not total:
                if 'BSCALE' in self.header.keys():
                    bscale = self.header['BSCALE']
                    # If BSCALE has been set to something reasonable, then adjust it
                    if (bscale != 0) and (bscale != 1):
                        outHead['BSCALE'] = (bscale/pixRatio,
                            'Calibration Factor')

                if 'BZERO' in self.header.keys():
                    bzero  = self.header['BZERO']
                    # If BZERO has been set to something reasonable, then adjust it
                    if (bzero != 0):
                        outHead['BZERO'] = (bzero/pixRatio,
                            'Additive Constant for Calibration')

        else:
            # If no header exists, then buil a basic one
            keywords = ['NAXIS2', 'NAXIS1']
            values   = (ny1, nx1)
            headDict = dict(zip(keywords, values))
            outHead  = fits.Header(headDict)

        if copy:
            # If a copy was requesty, then return a copy of the original image
            # with a newly rebinned array
            outImg         = self.copy()
            outImg.arr     = rebinArr

            # Update the uncertainty attribute
            # This may be a newly computed "uncertainty of the mean"
            outImg.sigma = rebinSig

            # Update the header if it exists
            if hasattr(self, 'header'):
                outImg.header  = outHead

            # TODO make binning a (dx, dy) tuple
            outImg.binning = (outImg.binning[0]/xratio,
                              outImg.binning[1]/yratio)

            # Return the updated image object
            return outImg
        else:
            # Otherwise place the rebinned array directly into the Image object
            self.arr     = rebinArr

            if hasattr(self, 'sigma'):
                self.sigma   = rebinSig

            if hasattr(self, 'header'):
                self.header  = outHead

            # TODO make binning a (dx, dy) tuple
            self.binning = (self.binning[0]/xratio,
                            self.binning[1]/yratio)

    def pad(self, pad_width, mode='constant', **kwargs):
        '''A method for padding the arr and sigma attributes and also updating
        the astrometry information in the header.
        '''
        # Pad the primary array
        tmpArr   = np.pad(self.arr, pad_width, mode=mode, **kwargs)
        self.arr = tmpArr

        # Pad the sigma array if one is defined
        if hasattr(self, 'sigma'):
            tmpSig     = np.pad(self.sigma, pad_width, mode=mode, **kwargs)
            self.sigma = tmpSig

        # Update the header information if possible
        if hasattr(self, 'header'):
            tmpHeader = self.header.copy()

            # Check if the header has good WCS
            if self.wcs.has_celestial:
                # Parse the pad_width parameter
                if len(pad_width) > 1:
                    # If separate x and y paddings were specified, check them
                    yPad, xPad = pad_width
                    # Grab only theh left-padding values
                    if len(xPad) > 1: xPad = xPad[0]
                    if len(yPad) > 1: yPad = yPad[0]
                else:
                    xPad, yPad = pad_width, pad_width

                # Now apply the actual updates to the header
                tmpHeader['CRPIX1'] = self.header['CRPIX1'] + xPad
                tmpHeader['CRPIX2'] = self.header['CRPIX2'] + yPad
                tmpHeader['NAXIS1'] = self.arr.shape[1]
                tmpHeader['NAXIS2'] = self.arr.shape[0]

                # And store those updateds in the self object
                self.header = tmpHeader

    def crop(self, x1, x2, y1, y2, copy=False):
        """This method crops the image array to the locations specified by the
        arguments and updates the header to match the new array.
        """

        # Check that the crop values are reasonable
        ny, nx = self.arr.shape
        if ((x1 < 0) or (x2 > (nx - 1)) or
            (y1 < 0) or (y2 > (ny - 1)) or
            (x2 < x1) or (y2 < y1)):
            raise ValueError('Bad crop values.')
        else:
            # Force crop points to be integers
            x1 = int(np.round(x1))
            x2 = int(np.round(x2))
            y1 = int(np.round(y1))
            y2 = int(np.round(y2))

        # Make a copy of the array and header
        cropArr = self.arr.copy()

        # Perform the actual croping
        cropArr = cropArr[y1:y2, x1:x2]

        # Repeat the process for the sigma array if it exists
        if hasattr(self, 'sigma'):
            cropSig = self.sigma.copy()
            cropSig = cropSig[y1:y2, x1:x2]

        if hasattr(self, 'header'):
            outHead = self.header.copy()
            # Update the header keywords
            # First update the NAXIS keywords
            outHead['NAXIS1'] = y2 - y1
            outHead['NAXIS2'] = x2 - x1

            # Next update the CRPIX keywords
            outHead['CRPIX1'] = outHead['CRPIX1'] - x1
            outHead['CRPIX2'] = outHead['CRPIX2'] - y1

        if copy:
            # If a copy was requesty, then return a copy of the original image
            # with a newly cropped array
            outImg        = self.copy()
            outImg.arr    = cropArr
            outImg.header = outHead

            # Handle sigma array if it exists
            if hasattr(self, 'sigma'):
                outImg.sigma  = cropSig

            return outImg
        else:
            # Otherwise place the rebinned array directly into the Image object
            self.arr    = cropArr
            self.header = outHead

            # Handle sigma array if it exists
            if hasattr(self, 'sigma'):
                self.sigma = cropSig

    def shift(self, dx, dy, conserve_flux = True, padding=-1e6):
        """A method to shift the image dx pixels to the right and dy pixels up.
        This method will NOT conserve flux, but it will attempt to handle
        uncertainties.

        parameters:
        dx -- number of pixels to shift right (negative is left)
        dy -- number of pixels to shift up (negative is down)
        """

        # If there is no sigma, then inverse-variance weighting is impossible
        if not hasattr(self, 'sigma'):
            conserve_flux = True
            sigmaExists = False
        else:
            sigmaExists = True

        # Store the original shape of the image array
        ny, nx = self.arr.shape

        # Check if the X shift is an within 1 billionth of an integer value
        if round(float(dx), 12).is_integer():
            # Force the shift to an integer value
            dx = np.int(round(dx))

            # Make a copy and apply the shift.
            shiftArr  = np.roll(self.arr, dx, axis = 1)

            # Apply the same shifts to the sigma array if it exists
            if sigmaExists:
                shiftSig = np.roll(self.sigma, dx, axis = 1)

        else:
            # The x-shift is non-integer...
            # Compute the two integer shiftings needed
            dxRt = np.int(np.ceil(dx))
            dxLf = dxRt - 1

            # Produce the shifted arrays
            arrRt = np.roll(self.arr, dxRt, axis = 1)
            arrLf = np.roll(self.arr, dxLf, axis = 1)

            # Compute the fractional contributions of each array
            fracRt = np.abs(dx - dxLf)
            fracLf = np.abs(dx - dxRt)

            if conserve_flux:
                # Compute the shifted array
                shiftArr = fracRt*arrRt + fracLf*arrLf

                if sigmaExists:
                    sigRt = np.roll(self.sigma, dxRt, axis = 1)
                    sigLf = np.roll(self.sigma, dxLf, axis = 1)

                    # Compute the shifted array
                    shiftSig = np.sqrt((fracRt*sigRt)**2 + (fracLf*sigLf)**2)
            else:
                # Apply the same shifts to the sigma array if it exists
                sigRt = np.roll(self.sigma, dxRt, axis = 0)
                sigLf = np.roll(self.sigma, dxLf, axis = 0)

                # Compute the weights for the array
                wtRt = fracRt/sigRt**2
                wtLf = fracLf/sigLf**2

                # Compute the weighted array values
                shiftArr  = wtRt*arrRt + wtLf*arrLf
                shiftNorm = wtRt + wtLf
                shiftArr /= shiftNorm

                # Compute the shifted array
                shiftSig = np.sqrt(shiftNorm)

        # Now fill in the shifted arrays and save them in the attributes
        fillX = np.int(np.abs(np.ceil(dx)))
        if dx > 0:
            shiftArr[:,0:fillX] = padding
        elif dx < 0:
            shiftArr[:,(nx-fillX-1):nx] = padding

        # Place the final result in the arr attribute
        self.arr = shiftArr

        if sigmaExists:
            # Now fill in the shifted arrays
            if dx > 0:
                shiftSig[:,0:fillX] = np.abs(padding)
            elif dx < 0:
                shiftSig[:,(nx-fillX-1):nx] = np.abs(padding)

            # Place the shifted array in the sigma attribute
            self.sigma = shiftSig

        # Check if the Y shift is an within 1 billianth of an integer value
        if round(float(dy), 12).is_integer():
            # Force the shift to an integer value
            dy = np.int(round(dy))

            # Make a copy and apply the shift.
            shiftArr  = np.roll(self.arr, dy, axis = 0)

            # Apply the same shifts to the sigma array if it exists
            if sigmaExists:
                shiftSig = np.roll(self.sigma, dy, axis = 0)
        else:
            # The y-shift is non-integer...
            # Compute the two integer shiftings needed
            dyTop = np.int(np.ceil(dy))
            dyBot = dyTop - 1

            # Produce the shifted arrays
            arrTop = np.roll(self.arr, dyTop, axis = 0)
            arrBot = np.roll(self.arr, dyBot, axis = 0)

            # Compute the fractional contributions of each array
            fracTop = np.abs(dy - dyBot)
            fracBot = np.abs(dy - dyTop)

            if conserve_flux:
                # Compute the shifted array
                shiftArr = fracTop*arrTop + fracBot*arrBot

                # Apply the same shifts to the sigma array if it exists
                if sigmaExists:
                    sigTop = np.roll(self.sigma, dyTop, axis = 0)
                    sigBot = np.roll(self.sigma, dyBot, axis = 0)

                    # Compute the shifted array
                    shiftSig = np.sqrt((fracTop*sigTop)**2 + (fracBot*sigBot)**2)
            else:
                # Apply the same shifts to the sigma array if it exists
                sigTop = np.roll(self.sigma, dyTop, axis = 0)
                sigBot = np.roll(self.sigma, dyBot, axis = 0)

                # Compute the weights for the array
                wtTop = fracTop/sigTop**2
                wtBot = fracBot/sigBot**2

                # Compute the weighted array values
                shiftArr  = wtTop*arrTop + wtBot*arrBot
                shiftNorm = wtTop + wtBot
                shiftArr /= shiftNorm

                # Compute the shifted array
                shiftSig = np.sqrt(shiftNorm)

        # Filll in the emptied pixels
        fillY = np.int(np.abs(np.ceil(dy)))
        if dy > 0:
            shiftArr[0:fillY,:] = padding
        elif dy < 0:
            shiftArr[(ny-fillY-1):ny,:] = padding

        # Place the shifted array in the arr attribute
        self.arr = shiftArr

        if sigmaExists:
            # Now fill in the shifted arrays
            if dy > 0:
                shiftSig[0:fillY,:] = np.abs(padding)
            elif dy < 0:
                shiftSig[(ny-fillY-1):ny,:] = np.abs(padding)

            # Place the shifted array in the sigma attribute
            self.sigma = shiftSig

        # Finally, update the header information
        if hasattr(self, 'header'):
            # Check if the header contains celestial WCS coords
            if self.wcs.has_celestial:
                # Update the header astrometry
                self.header['CRPIX1'] = self.header['CRPIX1'] + dx
                self.header['CRPIX2'] = self.header['CRPIX2'] + dy

    def in_image(self, coords, edge=0):
        """A method to test which (RA, Dec) coordinates lie within the image.

        returns:
        Array of boolean values. The array will contain "True" for those
         coordinates which lie within the image footprint and False for those
         coordinates which lie outside the image footprint.

        parameters:
        coords -- the (RA, Dec) pairs to check (must be an instance of the
                  astropy.coordinates.SkyCoord class)
        edge   -- Specifies the amount of image to ignore (arcsec). If the
                  specified point is within the image but is less than "edge"
                  arcsec from the edge of the image, then a False value is
                  returned.
        """
        # Transform coordinates to pixel positions
        x, y = coords.to_pixel(self.wcs)

        # Make sure that x and y are at least one dimension long
        if x.size == 1:
            x = np.array([x]).flatten()
        if y.size == 1:
            y = np.array([y]).flatten()

        # Replace NANs with reasonable but sure to fail values
        badX = np.logical_not(np.isfinite(x))
        badY = np.logical_not(np.isfinite(y))
        badInd = np.where(np.logical_or(badX, badY))
        if len(badInd[0]) > 0:
            x[badInd] = -1
            y[badInd] = -1

        # Grab the array size
        ny, nx = self.arr.shape

        # Check which coordinates fall within the image
        xGood = np.logical_and(x > edge, x < (nx - edge - 1))
        yGood = np.logical_and(y > edge, y < (ny - edge - 1))
        allGood = np.logical_and(xGood, yGood)

        return allGood

    def clear_astrometry(self):
        """Delete the header values pertaining to the astrometry.
        """
        if hasattr(self, 'header'):
            # Double make sure that there is no other WCS data in the header
            if 'WCSAXES' in self.header.keys():
                del self.header['WCSAXES']

            if len(self.header['CDELT*']) > 0:
                del self.header['CDELT*']

            if len(self.header['CUNIT*']) > 0:
                del self.header['CUNIT*']

            if len(self.header['*POLE']) > 0:
                del self.header['*POLE']

            if len(self.header['CD*_*']) > 0:
                del self.header['CD*_*']

            if len(self.header['PC*_*']) > 0:
                del self.header['PC*_*']

            if len(self.header['CRPIX*']) > 0:
                del self.header['CRPIX*']

            if len(self.header['CRVAL*']) > 0:
                del self.header['CRVAL*']

            if len(self.header['CTYPE*']) > 0:
                del self.header['CTYPE*']

    def get_img_offsets(self, img, subPixel=False, mode='wcs'):
        """A method to compute the displacement offsets between two the "self"
        AstroImage and the "img" AstroImage.
        """
        ########################################################################
        #################### HANDLE WCS MODE IMAGE OFFSETS #####################
        ########################################################################
        if mode.lower() == 'wcs':
            # Grab self image WCS and pixel center
            try:
                wcs1 = self.wcs.copy()
                wcs2 = img.wcs.copy()
            except:
                raise RuntimeError('One of the images does not have a builtin WCS')

            # Compute the basic pointing of the two images
            x1 = np.mean([wcs1.wcs.crpix[0], wcs2.wcs.crpix[0]])
            y1 = np.mean([wcs1.wcs.crpix[1], wcs2.wcs.crpix[1]])

            # Convert pixels to sky coordinates
            RA1, Dec1 = wcs1.all_pix2world(x1, y1, 0)

            # Grab the WCS of the alignment image and convert back to pixels
            x2, y2 = wcs2.all_world2pix(RA1, Dec1, 0)
            x2, y2 = float(x2), float(y2)

            # Compute the image possition offset vector
            dx = x2 - x1
            dy = y2 - y1

            if subPixel:
                # If a fractional shift is requested, then simply store these
                # values for later use
                pass
            else:
                # Otherwise round shifts to nearest integer
                dx = np.int(np.round(dx))
                dy = np.int(np.round(dy))
        ########################################################################
        ############# HANDLE CROSS-CORRELATION MODE IMAGE OFFSETS ##############
        ########################################################################
        elif mode == 'cross_correlate':
            # n. b. This method appears to produce results accurate to better
            # than 0.1 pixel as determined by simply copying an image, shifting
            # it an arbitrary amount, and attempting to recover that shift.
            newSelf = self.copy()
            newImg  = img.copy()

            # Just to be sure everything works out properly, let's test to see if
            # these two images are the same shape.
            if newImg.arr.shape != newSelf.arr.shape:
                # Pad the arrays to make sure they are the same size
                ny1, nx1 = self.arr.shape
                ny2, nx2 = img.arr.shape
                if (nx1 > nx2):
                    padX    = nx1 - nx2
                    newImg.pad(((0,0), (0,padX)), mode='constant')
                    del padX
                if (nx1 < nx2):
                    padX    = nx2 - nx1
                    newSelf.pad(((0,0), (0,padX)), mode='constant')
                    del padX
                if (ny1 > ny2):
                    padY    = ny1 - ny2
                    newImg.pad(((0,padY),(0,0)), mode='constant')
                    del padY
                if (ny1 < ny2):
                    padY    = ny2 - ny1
                    newSelf.pad(((0,padY),(0,0)), mode='constant')
                    del padY

            # Use this value to test for pixel saturation
            satLimit = 300000

            # Define kernal shape for an median filtering needed
            binX, binY = self.binning
            medianKernShape = np.int(np.ceil(9.0/binX)), np.int(np.ceil(9.0/binY))

            # Make temporary copies for the "fftconvolve" alignment
            self1 = newSelf.copy()
            img1  = newImg.copy()

            # Replace negative values with image median, for now.
            selfNegPix  = np.nan_to_num(self1.arr) < 0
            if np.sum(selfNegPix) > 0:
                badInds  = np.where(selfNegPix)
                goodInds = np.where(np.logical_not(selfNegPix))
                self1.arr[badInds] = np.median(np.nan_to_num(self1.arr[goodInds]))

            # Replace NaN values with local median, for now.
            selfNaNpix  = np.logical_not(np.isfinite(self1.arr))
            if np.sum(selfNaNpix) > 0:
                badInds  = np.where(selfNaNpix)
                goodInds = np.where(np.logical_not(selfNaNpix))
                self1.arr[badInds] = np.median(self1.arr[goodInds])
                selfMed = median_filter(self1.arr, size = medianKernShape)
                self1.arr[badInds] = selfMed[badInds]

            # Replace negative values with image median, for now.
            imgNegPix = np.nan_to_num(img1.arr) < 0
            if np.sum(imgNegPix) > 0:
                badInds  = np.where(imgNegPix)
                goodInds = np.where(np.logical_not(imgNegPix))
                img1.arr[badInds] = np.median(np.nan_to_num(img1.arr[goodInds]))

            # Replace NaN values with local median, for now.
            imgNaNpix  = np.logical_not(np.isfinite(img1.arr))
            if np.sum(imgNaNpix) > 0:
                badInds  = np.where(imgNaNpix)
                goodInds = np.where(np.logical_not(imgNaNpix))
                img1.arr[badInds] = np.median(img1.arr[goodInds])
                imgMed = median_filter(img1.arr, size = medianKernShape)
                img1.arr[badInds] = imgMed[badInds]

            # Do an array flipped convolution, which is a correlation.
            corr = signal.fftconvolve(img1.arr, self1.arr[::-1, ::-1], mode='same')

            # Do a little post-processing to block out bad points in corr image
            # First filter with the median
            medCorr = median_filter(corr, size = medianKernShape)

            # Compute sigma_clipped_stats of the correlation image
            mean, median, stddev = sigma_clipped_stats(corr)

            # Then check for significant deviations from median.
            deviations = (np.abs(corr - medCorr) > 2.0*stddev)

            # Count the number of masked neighbors for each pixel
            neighborCount = np.zeros_like(corr, dtype=int)
            for dx1 in range(-1,2,1):
                for dy1 in range(-1,2,1):
                    neighborCount += np.roll(np.roll(deviations, dy1, axis=0),
                                             dx1, axis=1).astype(int)

            # Find isolated deviant pixels (these are no good!)
            deviations = np.logical_and(deviations, neighborCount <= 4)

            if np.sum(deviations > 0):
                # Inpaint those deviations
                tmp1 = utils.inpaint_nans(corr, mask = deviations)
                corr = tmp1

            # Check for the maximum of the cross-correlation function
            peak1  = np.unravel_index(corr.argmax(), corr.shape)
            dy, dx = np.array(peak1) - np.array(corr.shape)//2

            # If integer pixel shifts are ok, then just use the peak of the
            # correlation image to determine the total image offsets
            if subPixel:
                # Otherwise cut out subarrays around the brightest 25 stars, and
                # figure out what fractional shift value produces the best
                # correlation image for those subarrays.

                # Start by inpainting any local deviations (e.g. saturation)
                # First filter with the median
                medSelf1 = median_filter(self1.arr, size = medianKernShape)
                # Compute sigma_clipped_stats of the self image
                mean, median, stddev = sigma_clipped_stats(self1.arr)
                # Then check for significant deviations from median.
                deviations = (np.abs(self1.arr - medSelf1) > 2.0*stddev)
                # Count the number of masked neighbors for each pixel
                neighborCount = np.zeros(self1.arr.shape, dtype=int)
                for dx1 in range(-1,2,1):
                    for dy1 in range(-1,2,1):
                        neighborCount += np.roll(np.roll(deviations, dy1, axis=0),
                                                 dx1, axis=1).astype(int)

                # Apply the initial (integer) shifts to the (non-nan) images
                img1.shift(-dx, -dy)

                # Combine images to find the brightest 25 (or 16, or 9 stars in the image)
                comboImg = self1 + img1

                # Find nans and low numbers
                badPix = np.logical_or(
                    comboImg.arr < -1e5,
                    np.logical_not(np.isfinite(comboImg.arr)))
                if np.sum(badPix) > 0:
                    badInds = np.where(badPix)
                    comboImg.arr[badInds] = 0

                # Get the image statistics
                mean, median, std = sigma_clipped_stats(
                    comboImg.arr, sigma=3.0, iters=5)

                # Start by instantiating a DAOStarFinder object
                daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)

                # Use that object to find the "stars" in the images
                sources = daofind(comboImg.arr - median)

                # Sort the sources lists by brightness
                sortInds = np.argsort(sources['mag'])
                sources  = sources[sortInds]

                # Only keep high quality detected sources.
                # (1) - delete edge stars
                # Remove detections within 40 pixels of the image edge
                # (This guarantees that the star-cutout process will succeed)
                ny, nx   = comboImg.arr.shape
                goodX    = np.logical_and(sources['xcentroid'] > 40,
                                         sources['xcentroid'] < (nx - 40))
                goodY    = np.logical_and(sources['ycentroid'] > 40,
                                         sources['ycentroid'] < (ny - 40))
                goodInds = np.where(np.logical_and(goodX, goodY))
                sources  = sources[goodInds]

                # (2) - delete "crowded" stars and non-gaussian stars
                # Build a gaussian model for fitting stars
                gauss_init = models.Gaussian2D(
                    amplitude=1000.0,
                    x_mean=10.0,
                    y_mean=10.0,
                    x_stddev=3.0,
                    y_stddev=3.0,
                    theta=0.0)
                bkg_init   = models.Polynomial2D(1)
                patch_init = gauss_init + bkg_init
                fitter     = fitting.LevMarLSQFitter()

                # Remove any detections with neighbors within sqrt(2)*10 pixels,
                # or if a guassian model cannot be fit to that star.
                starCutout    = 20
                nearestDist   = np.sqrt(2)*0.5*starCutout
                yy, xx        = np.mgrid[0:starCutout,0:starCutout]
                patchList     = []
                keepStarList  = []
                keepStarCount = 0
                for source in sources:
                    # If we already have the maximum number of acceptable stars,
                    # then simply BREAK out of the Loop.
                    if keepStarCount >= 25: break

                    # Save THIS star position
                    xStar, yStar = source['xcentroid'], source['ycentroid']

                    # Compute the distances to the other stars
                    dists = np.sqrt((sources['xcentroid'].data - xStar)**2 +
                                    (sources['ycentroid'].data - yStar)**2)

                    # Test for stars *CLOSER* than star patch and *FARTHER* than 0 pixels
                    nearBool = np.logical_and(dists > 0, dists < nearestDist)

                    # Check if the nearest star is within this star's "patch"
                    numNearStars = np.sum(nearBool)
                    crowdBool = numNearStars > 1

                    # Do a double check to see if there are any EXTRA sources
                    # Start by cutting out the patch surrounding this star
                    # Establish the cutout bondaries
                    btCut = np.int(np.round(yStar - np.floor(0.5*starCutout)))
                    tpCut = np.int(np.round(btCut + starCutout))
                    lfCut = np.int(np.round(xStar - np.floor(0.5*starCutout)))
                    rtCut = np.int(np.round(lfCut + starCutout))
                    patch_data = np.nan_to_num(comboImg.arr[btCut:tpCut,lfCut:rtCut])

                    # Start by instantiating a DAOStarFinder object
                    daofind = DAOStarFinder(fwhm=3.0, threshold=3.0*std)

                    # Use that object to find the sources in the patch
                    srcTest = daofind(patch_data - patch_data.min())

                    # Count the length of the source table found
                    crowdBool = np.logical_or(crowdBool, len(srcTest) > 1)

                    # If there is a nearby star, then mark for deletion
                    if crowdBool == True:
                        keepStarList.append(False)
                        continue

                    # Next, check if any part of this image is saturated...
                    satBool = patch_data.max() > satLimit
                    if satBool == True:
                        keepStarList.append(False)
                        continue

                    # Next, try to fit a gaussian to this image.
                    # By this point, we should be dealing with an *isolated*,
                    # *non-sturated* star, so a guassian fit *SHOULD* work if
                    # the source really is a star
                    with warnings.catch_warnings():
                        # Ignore model linearity warning from the fitter
                        warnings.simplefilter('ignore')
                        patch_model = fitter(patch_init, xx, yy, patch_data)

                    # Now test the "roundness" of the fit and skip the non-round
                    roundStat = patch_model.x_stddev_0/patch_model.y_stddev_0
                    if roundStat < 1.0: roundStat = 1.0/roundStat
                    if (roundStat > 1.5) or roundStat < 0:
                        keepStarList.append(False)
                        continue

                    # Finally, test if the star position is where expected.
                    fitDist = np.sqrt((patch_model.x_mean_0 - 0.5*starCutout)**2 +
                                      (patch_model.y_mean_0 - 0.5*starCutout)**2)
                    if fitDist > 3.0:
                        keepStarList.append(False)
                        continue

                    # At this point, if the star has passed all the above tests,
                    # then it should be stored as a positive "keepStar"
                    # Grab the patches from the respective images
                    patch_data1 = img1.arr[btCut:tpCut, lfCut:rtCut]
                    patch_data2 = self1.arr[btCut:tpCut, lfCut:rtCut]

                    # Fit each image with its own polynomial + gaussian model
                    with warnings.catch_warnings():
                        # Ignore model linearity warning from the fitter
                        warnings.simplefilter('ignore')
                        patch_model1 = fitter(patch_init, xx, yy, patch_data1)
                        patch_model2 = fitter(patch_init, xx, yy, patch_data2)

                    # First compute background subtracted data
                    # Begin by constructing a 2D polynomial model (plane)
                    bkg_model1 = models.Polynomial2D(1)
                    bkg_model2 = models.Polynomial2D(1)

                    # Transfer the background portion of the patch_model to the
                    # polynomial plane model.
                    bkg_model1.c0_0 = patch_model1.c0_0_1
                    bkg_model1.c1_0 = patch_model1.c1_0_1
                    bkg_model1.c0_1 = patch_model1.c0_1_1
                    bkg_model2.c0_0 = patch_model2.c0_0_1
                    bkg_model2.c1_0 = patch_model2.c1_0_1
                    bkg_model2.c0_1 = patch_model2.c0_1_1

                    # Compute the normalized, background subtracted data
                    patch_data1  = patch_data1 - bkg_model1(xx, yy)
                    patch_data1 -= patch_data1.min()
                    patch_data1 /= patch_data1.max()
                    patch_data2  = patch_data2 - bkg_model2(xx, yy)
                    patch_data2 -= patch_data2.min()
                    patch_data2 /= patch_data2.max()

                    # Store that in a list of subarrays
                    patchList.append((patch_data1, patch_data2))

                    # Finally, increment the number of stars kept, and mark this
                    # index as a "good" keepStar source.
                    keepStarCount += 1
                    keepStarList.append(True)

                # Now cull the source list to only include good stars
                if keepStarCount >= 9:
                    keepInds = np.where(keepStarList)
                    sources  = sources[keepInds]
                else:
                    raise RuntimeError('Fewer than 9 stars found: cannot compute subpixel offsets')

                # Cull the list to the brightest square number of stars
                if keepStarCount > 25:
                    keepStarCount = 25
                elif keepStarCount > 16:
                    keepStarCount = 16
                elif keepStarCount > 9:
                    keepStarCount = 9
                else:
                    raise RuntimeError('Fewer than 9 stars found: cannot compute subpixel offsets')

                # Perform the actual data cut
                sources = sources[0:keepStarCount]

                # Chop out the sections around each star,
                # and build a "starImage"
                numZoneSide = np.int(np.round(np.sqrt(keepStarCount)))
                starImgSide = starCutout*numZoneSide
                starImg1 = np.zeros((starImgSide, starImgSide))
                starImg2 = np.zeros((starImgSide, starImgSide))
                # Loop through each star to be cut out
                iStar = 0
                for xZone in range(numZoneSide):
                    for yZone in range(numZoneSide):
                        # Establish the pasting boundaries
                        btPaste = np.int(np.round(starCutout*yZone))
                        tpPaste = np.int(np.round(starCutout*(yZone + 1)))
                        lfPaste = np.int(np.round(starCutout*xZone))
                        rtPaste = np.int(np.round(starCutout*(xZone + 1)))

                        # Past each of the previously selected "patches" into
                        # their respective "starImg" for cross-correlation
                        patch1, patch2 = patchList[iStar]
                        starImg1[btPaste:tpPaste, lfPaste:rtPaste] = patch1
                        starImg2[btPaste:tpPaste, lfPaste:rtPaste] = patch2

                        # Increment the star counter
                        iStar += 1

                # Do an array flipped convolution, which is a correlation.
                corr  = signal.fftconvolve(starImg1, starImg2[::-1, ::-1],
                    mode='same')
                corr  = 100*corr/np.max(corr)

                # Check for the maximum of the cross-correlation function
                peak2 = np.unravel_index(corr.argmax(), corr.shape)

                # Chop out the central peak
                peakSz = 6
                btCorr = peak2[0] - peakSz
                tpCorr = btCorr + 2*peakSz + 1
                lfCorr = peak2[1] - peakSz
                rtCorr = lfCorr + 2*peakSz + 1
                corr1  = corr[btCorr:tpCorr, lfCorr:rtCorr]

                # Get the gradient of the cross-correlation function
                tmp     = AstroImage()
                tmp.arr = corr1
                Gx, Gy  = tmp.gradient()

                # Grab the index of the peak
                yPeak, xPeak = np.unravel_index(corr1.argmax(), corr1.shape)

                # Chop out the central zone and grab the minimum of the gradient
                cenSz = 3
                bot   = yPeak - cenSz//2
                top   = bot + cenSz
                lf    = xPeak - cenSz//2
                rt    = lf + cenSz

                # Grab the region near the minima
                yy, xx   = np.mgrid[bot:top, lf:rt]
                Gx_plane = Gx[bot:top, lf:rt]
                Gy_plane = Gy[bot:top, lf:rt]

                # Fit planes to the x and y gradients...Gx
                px_init = models.Polynomial2D(degree=1)
                py_init = models.Polynomial2D(degree=1)
                fit_p   = fitting.LinearLSQFitter()
                px      = fit_p(px_init, xx, yy, Gx_plane)
                py      = fit_p(py_init, xx, yy, Gy_plane)

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

                # Finally convert back into an absolute image offset
                dx1 = lfCorr + (x_soln.item(0) - (numZoneSide*starCutout)//2)
                dy1 = btCorr + (x_soln.item(1) - (numZoneSide*starCutout)//2)

                # Do a final check to see if these values are logical.
                # Sub-pixel perterbations greater than 2-pixels are rejected.
                if (dx1 > 2) or (dy1 > 2):
                    raise RuntimeError('Illogical sub-pixel offsets.')

                # Accumulate the fractional shift on top of the integer shift
                # computed earlier
                dx += dx1
                dy += dy1

        # At this point all image offsets (including possible sub-pixel
        # corrections) have been computed and will be returned as a tuple
        return (dx, dy)

    def align(self, img, subPixel=False, mode='wcs', offsets=None):
        """A method to align the self image with an other image
        using the astrometry from each header to shift an INTEGER
        number of pixels.

        parameters:
        img      -- the image with which self will be aligned
        subPixel -- if True, then images are shifted to be aligned with
                    sub-pixel precision
        mode     -- ['wcs' | 'cross_correlate']
                    defines the method used to align the two images
        """
        # Check if a list of offsets was supplied
        if offsets is None:
            # If no offsets were supplid, then retrieve them
            offsets = self.get_img_offsets(img, subPixel=subPixel, mode=mode)
        elif hasattr(offsets, '__iter__') and len(offsets) == 2:
            # This is just a check to make sure that the offsets are the
            # appropriate type and size. The actual alignment occurs after the
            # typechecking.
            pass
        else:
            raise ValueError('offsets keyword must be an iterable, 2 element offset')

        # Align self image with img image
        newSelf = self.copy()
        newImg  = img.copy()

        # Pad the arrays to make sure they are the same size
        ny1, nx1 = self.arr.shape
        ny2, nx2 = img.arr.shape
        if (nx1 > nx2):
            padX    = nx1 - nx2
            newImg.pad(((0,0), (0,padX)), mode='constant')
            del padX
        if (nx1 < nx2):
            padX    = nx2 - nx1
            newSelf.pad(((0,0), (0,padX)), mode='constant')
            del padX
        if (ny1 > ny2):
            padY    = ny1 - ny2
            newImg.pad(((0,padY),(0,0)), mode='constant')
            del padY
        if (ny1 < ny2):
            padY    = ny2 - ny1
            newSelf.pad(((0,padY),(0,0)), mode='constant')
            del padY

        # newImage and newSelf should the same shape and size now...
        # let's double check that's true
        if newImg.arr.shape != newSelf.arr.shape:
            raise IndexError('AstroImage should be padded to the same shape')

        # Unpack the offsets into their (dx, dy) vector format
        dx, dy = offsets

        # Compute number of padding pixels required for each side of the image to
        # accomodate this displacement vector (dx, dy)
        padX = np.int(np.ceil(np.abs(dx)/2))
        padY = np.int(np.ceil(np.abs(dy)/2))

        # Construct the before-after padding combinations
        selfPadX = (padX, padX)
        selfPadY = (padY, padY)

        # Reverse the padding ammounts to be applied to the secondary image
        imgPadX = selfPadX[::-1]
        imgPadY = selfPadY[::-1]

        # Compute the shifting amount
        if subPixel:
            selfShiftX = +0.5*dx
            imgShiftX  = selfShiftX - dx
            selfShiftY = +0.5*dy
            imgShiftY  = selfShiftY - dy
        else:
            selfShiftX = +np.int(np.round(0.5*dx))
            imgShiftX  = -np.int(np.round(dx - selfShiftX))
            selfShiftY = +np.int(np.round(0.5*dy))
            imgShiftY  = -np.int(np.round(dy - selfShiftY))

        # Define the padding widths
        # (recall axis ordering is 0=z, 1=y, 2=x, etc...)
        selfPadWidth = np.array((selfPadY, selfPadX), dtype=np.int)
        imgPadWidth  = np.array((imgPadY,  imgPadX), dtype=np.int)

        # Compute the padding to be added and pad the arr and sigma arrays
        newSelf.pad(selfPadWidth, mode='constant')
        newImg.pad(imgPadWidth, mode='constant')

        # Update header info
        # New header may already be correct, but no harm in double checking.
        newSelf.header['NAXIS1'] = newSelf.arr.shape[1]
        newSelf.header['NAXIS2'] = newSelf.arr.shape[0]
        newImg.header['NAXIS1']  = newImg.arr.shape[1]
        newImg.header['NAXIS2']  = newImg.arr.shape[0]

        # Shift the images
        newSelf.shift(selfShiftX, selfShiftY)
        newImg.shift(imgShiftX, imgShiftY)

        # Retun the aligned Images (not the same size as the input images)
        return (newSelf, newImg)

    def fix_astrometry(self):
        """This ensures that the CDELT values and PC matrix are properly set.
        """

        # Check if there is a header in this image
        if hasattr(self, 'header'):
            pix_scales = self.get_plate_scales()
            if len(self.header['CDELT*']) > 0:
                # If there are CDELT values,
                if ((pix_scales[0] != self.header['CDELT1']) or
                    (pix_scales[1] != self.header['CDELT2'])):
                    # and if they are not ACTUALLY set to the plate scales,
                    # then update the astrometry keyword values
                    CDELT1p = pix_scales[0]
                    CDELT2p = pix_scales[1]

                    # Update the header values
                    self.header['CDELT1'] = CDELT1p
                    self.header['CDELT2'] = CDELT2p
                    self.header['PC1_1']  = self.header['PC1_1']/CDELT1p
                    self.header['PC1_2']  = self.header['PC1_2']/CDELT1p
                    self.header['PC2_1']  = self.header['PC2_1']/CDELT2p
                    self.header['PC2_2']  = self.header['PC2_2']/CDELT2p
        else:
            raise ValueError('No header in this imagae')

    def gradient(self, kernel='sobel'):
        """Computes and returns the gradient (Gx, Gy) of the image using
        either the Sobel or Prewitt opperators.
        """
        if kernel.upper() == 'SOBEL':
            Gx = ndimage.sobel(self.arr, axis=1)
            Gy = ndimage.sobel(self.arr, axis=0)
        elif kernel.upper() == 'PREWITT':
            Gx = ndimage.prewitt(self.arr, axis=1)
            Gy = ndimage.prewitt(self.arr, axis=0)
        else:
            raise ValueError('kernel keyword value must be "SOBEL" or "PREWITT"')

        return (Gx, Gy)

    def get_ticks(self):
        """Uses info from the array shape and and image header to assess the
        spacing between RA and Dec ticks, major tick formatting, and the minor
        tick frequency.
        """
        # Grab the WCS from the header.
        if hasattr(self, 'wcs'):
            # First compute the image dimensions in arcsec
            ps_x, ps_y    = self.get_plate_scales()
            height, width = (
                np.array(self.arr.shape)
                *np.array([ps_y, ps_x])*3600*u.arcsec)

            # Setup a range of viable major tick spacing options
            spacingOptions = np.array([
                0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 20, 30, 1*60, 2*60, 5*60,
                10*60, 30*60, 1*60*60, 2*60*60, 5*60*60])*u.arcsec

            # Setup corresponding RA and Dec tick label format strings
            RAformatters = np.array([
                'hh:mm:ss.s', 'hh:mm:ss.s', 'hh:mm:ss.s',
                'hh:mm:ss', 'hh:mm:ss', 'hh:mm:ss', 'hh:mm:ss',
                'hh:mm:ss', 'hh:mm:ss', 'hh:mm:ss', 'hh:mm',
                'hh:mm',    'hh:mm',    'hh:mm',    'hh:mm',
                'hh',       'hh',       'hh'])
            DecFormatters = np.array([
                'dd:mm:ss.s', 'dd:mm:ss.s', 'dd:mm:ss.s',
                'dd:mm:ss', 'dd:mm:ss', 'dd:mm:ss', 'dd:mm:ss',
                'dd:mm:ss', 'dd:mm:ss', 'dd:mm:ss', 'dd:mm',
                'dd:mm',    'dd:mm',    'dd:mm',    'dd:mm',
                'dd',       'dd',       'dd'])

            # Define a set of minor tick frequencies associated with each
            # major tick spacing
            minorTicksFreqs = np.array([
                10, 4, 5, 10, 4, 5, 10, 3, 10, 6, 10, 4, 5, 10, 6, 10, 4, 5])

            # Figure out which major tick spacing provides the FEWEST ticks
            # but greater than 3
            y_cen, x_cen = 0.5*np.array(self.arr.shape)
            RA_cen, Dec_cen = self.wcs.wcs_pix2world([x_cen], [y_cen], 0)

            # Find the index of the proper tick spacing for each axis
            RAspacingInd  = np.max(np.where(np.floor(
                (width/(15*spacingOptions*np.cos(np.deg2rad(Dec_cen)))).value) >= 2))
            DecSpacingInd = np.max(np.where(np.floor(
                (height/spacingOptions).value) >= 3))

            # Select the actual RA and Dec spacing
            RAspacing  = 15*spacingOptions[RAspacingInd]
            DecSpacing = spacingOptions[DecSpacingInd]

            # Select the specific formatting for this tick interval
            RAformatter  = RAformatters[RAspacingInd]
            DecFormatter = DecFormatters[DecSpacingInd]

            # And now select the minor tick frequency
            RAminorTicksFreq  = minorTicksFreqs[RAspacingInd]
            DecMinorTicksFreq = minorTicksFreqs[DecSpacingInd]

        # TODO NEST THIS RETURN WITHIN THE 'IF' TO GUARANTEE THAT IT ONLY HAPPENS
        # IF A WCS IS PRESENT
        return ((RAspacing, DecSpacing),
            (RAformatter, DecFormatter),
            (RAminorTicksFreq, DecMinorTicksFreq))

    def show(self, axes=None, origin='lower', noShow=False,
             scale='linear', vmin=None, vmax=None,
             ticks=True, cmap='viridis', **kwargs):
        """Displays the image to the user for interaction (including clicking?)
        This method includes all the same keyword arguments as the "imshow()"
        method from matplotlip.pyplot. This allows the user to control how the
        image is displayed.

        Additional keyword arguments include
        scale -- ['linear' | 'log' | 'asinh']
                 allows the user to specify if if the image stretch should be
                 linear or log space
        """
        # First, determine if the current state is "interactive"
        isInteractive = mpl.is_interactive()

        # Set the scaling for the image
        if scale == 'linear':
            # Compute a display range in terms of the image noise level
            mean, median, stddev = sigma_clipped_stats(self.arr.flatten())
            if vmin == None: vmin = median - 2*stddev
            if vmax == None: vmax = median + 10*stddev

            # Steup the array to display
            showArr    = self.arr
            normalizer = mcol.Normalize(vmin = vmin, vmax = vmax)
        elif scale == 'log':
            # Compute a display range in terms of the image noise level
            mean, median, stddev = sigma_clipped_stats(self.arr.flatten())
            if vmin == None:
                testMin = median - 0.8*stddev
                vmin = testMin if testMin > 0 else 0.01
            if vmax == None: vmax = median + 80*stddev

            # Clip the array (so there are no logs of negative values)
            showArr    = np.clip(self.arr, vmin, vmax)
            normalizer = mcol.LogNorm(vmin = vmin, vmax = vmax)
        else:
            raise ValueError('The provided "scale" keyword is not recognized')

        # Create the figure and axes for displaying the image
        if axes is None:
            # Create a new figure and axes
            fig  = plt.figure(figsize = (8,8))
            if hasattr(self, 'wcs'):
                axes = fig.add_subplot(1, 1, 1, projection=self.wcs)

                # Set the axes linewidth
                axes.coords.frame.set_linewidth(2)

                # Label the axes establish minor ticks.
                RA_ax  = axes.coords[0]
                Dec_ax = axes.coords[1]
                RA_ax.set_axislabel('RA [J2000]',
                                    fontsize=12, fontweight='bold')
                Dec_ax.set_axislabel('Dec [J2000]',
                                     fontsize=12, fontweight='bold', minpad=-0.4)

                # Retrieve the apropriate spacing and formats
                spacing, formatter, minorTicksFreq = self.get_ticks()

                # Set the tick width and length
                RA_ax.set_ticks(spacing=spacing[0], size=12, width=2)
                Dec_ax.set_ticks(spacing=spacing[1], size=12, width=2)

                # Set tick label formatters
                RA_ax.set_major_formatter(formatter[0])
                Dec_ax.set_major_formatter(formatter[1])

                # Set the other tick label format
                RA_ax.set_ticklabel(fontsize=12, fontweight='demibold')
                Dec_ax.set_ticklabel(fontsize=12, fontweight='demibold')

                # Turn on minor ticks and set number of minor ticks
                RA_ax.display_minor_ticks(True)
                Dec_ax.display_minor_ticks(True)
                RA_ax.set_minor_frequency(minorTicksFreq[0])
                Dec_ax.set_minor_frequency(minorTicksFreq[1])
            else:
                axes = fig.add_subplot(1,1,1)
            # Set the axes line properties
            for axis in ['top','bottom','left','right']:
                axes.spines[axis].set_linewidth(4)

            # Put the image in its place
            axIm = axes.imshow(showArr, origin=origin, norm = normalizer,
                               cmap=cmap, **kwargs)
        else:
            # Use the provided axes
            fig  = axes.figure
            axIm = axes.imshow(showArr, origin=origin, norm = normalizer,
                               cmap=cmap, **kwargs)

        # Display the image to the user, if requested
        if not noShow:
            plt.ion()
            fig.show()

            # Reset the original "interactive" state
            if not isInteractive:
                plt.ioff()

        # Return the graphics objects to the user
        return (fig, axes, axIm)

    def oplot_sources(self, satLimit=16000, crowdThresh=0.0,
        s=100, edgecolor='red', facecolor='none', **kwargs):
        """A method to overplot sources using the pyplot.scatter() method and its
        associated keywords
        """
        # Grab the sources using the get_sources() method
        xs, ys = self.get_sources(satLimit=satLimit, crowdThresh=crowdThresh)

        # The following line makes it so that the zoom level no longer changes,
        # otherwise Matplotlib has a tendency to zoom out when adding overlays.
        ax = plt.gca()
        ax.set_autoscale_on(False)
        ax.scatter(xs, ys, s=s, edgecolor=edgecolor, facecolor=facecolor, **kwargs)

        # Force a redraw of the canvas
        plt.draw()

class Bias(AstroImage):
    """A subclass of the "Image" class: stores bias images and provides some
    methods for bias type operations.
    """

    def __init__(self, filename = ''):
        super(Bias, self).__init__(filename)

    def average(self):
        return np.mean(self.arr)

    def master_bias(biasList, clipSigma = 3.0):
        return utils.stacked_average(biasList, clipSigma = clipSigma)

    def overscan_polynomial(biasList, overscanPos):
        ## Loop through biasList and build a stacked average of the
        ## overscanRegion
        numBias      = len(biasList)
#        overscanList = biasList.copy()
        binning      = np.unique([(biasList[i].header['CRDELT1'],
                                   biasList[i].header['CRDELT2'])
                                   for i in range(numBias)])
        if len(binning) > 1:
            raise ValueError('Binning is different for each axis')
        overscanPos1   = overscanPos/binning

        ## Trim the overscan list to contain ONLY the overscan.
        overscanList = []
        for i in range(len(biasList)):
            overscanList.append(biasList[i].copy())
            overscanList[i].arr = \
              biasList[i].arr[overscanPos1[0][1]:overscanPos1[1][1], \
                                  overscanPos1[0][0]:overscanPos1[1][0]]
        masterOverscan = utils.stacked_average(overscanList)

        ## Average across each row.
        overscanRowValues = np.mean(masterOverscan, axis = 1)
        rowInds           = range(len(overscanRowValues))

        ## Compute statistics for a zeroth order polynomial.
        polyCoeffs = np.polyfit(rowInds, overscanRowValues, 0)
        polyValues = np.polyval(polyCoeffs, rowInds)
        chi2_m     = np.sum((overscanRowValues - polyValues)**2)

        ## Loop through polynomial degrees and store Ftest result
#        alpha  = 0.05   #Use a 5% "random probability" as a cutoff
        sigma  = 7.0    #Use a 7 sigma requirement for so much data
        alpha  = (1 - scipy.stats.norm.cdf(sigma))
        Ftests = []
        coeffs = []
        for deg in range(1,10):
            dof        = len(overscanRowValues) - deg - 1
            polyCoeffs = np.polyfit(rowInds, overscanRowValues, deg)
            coeffs.append(polyCoeffs)
            polyValues = np.polyval(polyCoeffs, rowInds)
            chi2_m1    = np.sum((overscanRowValues - polyValues)**2)
#            print('reduced Chi2 = ' + str(chi2_m1/dof))
            Fchi       = (chi2_m - chi2_m1)/(chi2_m1/dof)
            prob       = 1 - scipy.stats.f.cdf(Fchi, 1, dof)
            Ftests.append(prob < alpha)

            ## Store chi2_m1 in chi2 for use in next iteration
            #***** NOTE HERE *****
            # there are so many degrees of freedom (v2 ~ 20000 pixels!)
            # that Fchi(v1=1, v2) = Fchi(v1=1, v2=Inf), so almost all
            # polynomials will pass the test (until we get to Fchi < 3.84
            # for alpha = 0.05). We can decrease alpha (stricter test), or
            # we can rebin along the column
            chi2_m     = chi2_m1


        ## Find the LOWEST ORDER FAILED F-test and return the degree
        ## of the best fitting polynomial
        bestDegree = np.min(np.where([not test for test in Ftests]))

#        # Plot all of the fits...
#        for plotNum in range(len(coeffs)):
#            cf = coeffs[plotNum]
#            plt.subplot(3, 3, plotNum + 1)
#            plt.plot(rowInds, overscanRowValues)
#            polyValues = np.polyval(cf, rowInds)
#            plt.plot(rowInds, polyValues, linewidth = 4, color = 'r')
#            plt.text(1250, 1029, str(Ftests[plotNum]))
#        plt.show()
        return bestDegree

#    @property
#    def read_noise(self):
#        return np.std(self.arr)

class Flat(AstroImage):
    """A subclass of the "Image" class: stores flat frames and provides some
    methods for flat type operations.
    """

    def __init__(self, filename = ''):
        super(Flat, self).__init__(filename)

    def master_flat(flatList, clipSigma = 3.0):
        return utils.stacked_average(flatList, clipSigma = clipSigma)

class Dark(AstroImage):
    """A subclass of the "Image" class: stores dark frames and provides some
    methods for dark type operations.
    """

    def __init__(self, filename = ''):
        super(Dark, self).__init__(filename)

    def dark_time(darkList):

        # Build a list of all the exposure times
        expTimeList = [dark.header['EXPTIME'] for dark in darkList]

        # Check if all the exposure times are the same...
        expTimes = np.unique(expTimeList)
        if expTimes.size > 1:
            raise ValueError('More than one exposure time found in list')
        else:
            return expTimes[0]

    def dark_current(darkList, clipSigma = 3.0):
        avgImg   = utils.stacked_average(darkList, clipSigma = clipSigma)
        darkTime = Dark.dark_time(darkList)
        return avgImg/darkTime
