import pdb
import os
import copy
import numpy as np
from scipy import signal
from scipy import ndimage
from scipy.ndimage.filters import median_filter
import scipy.optimize as opt
import scipy.stats

import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats
from astropy.io import fits

# TODO
import warnings
# Catch the "RuntimeWarning" in the magic methods
# Catch the "UserWarning" in the init procedure

#from astropy.wcs import WCS
from wcsaxes import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils import daofind

# Finally import the associated "image_tools" python module
import image_tools

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
                HDUlist     = fits.open(filename, do_not_scale_image_data=True)
            except:
                raise FileNotFoundError('File {0} does not exist'.format(filename))

            # If the file loaded properly, then proceed as usual
            self.header = HDUlist[0].header.copy()
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
            try:
                # Check that binning makes sense and store it if it does
                self.binning = tuple([int(di) for di in self.header['CRDELT*'].values()])
            except:
                # No binning found, so call this (1x1) binning.
                self.binning = tuple(np.ones(self.arr.ndim).astype(int))
                for i, di in enumerate(self.binning):
                    self.header['CRDELT'+str(i)] = di

            # Loop through the HDUlist and check for a 'SIGMA' HDU
            for HDU in HDUlist:
                if HDU.name == 'SIGMA':
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
        oneIsInt      = isinstance(other, int)
        oneIsFloat    = isinstance(other, float)

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
            print('Attempting to add image and something weird...')
            return None

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
        oneIsInt      = isinstance(other, int)
        oneIsFloat    = isinstance(other, float)

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
            print('Attempting to subtract image and something weird...')
            return None

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
        oneIsInt      = isinstance(other, int)
        oneIsFloat    = isinstance(other, float)
        if bothAreImages:
            # Check that image shapes make sense
            shape1     = self.arr.shape
            shape2     = other.arr.shape
            if shape1 == shape2:
                output     = self.copy()
                output.arr = self.arr * other.arr
            else:
                print('Cannot multiply images with different shapes')
                return None

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
        else:
            print('Attempting to multiply image by something weird...')
            pdb.set_trace()
            return None

        # Retun the multiplied image
        return output

    def __rmul__(self, other):
        # Implements reverse multiplication.
        if other == 0:
            output = self.copy()
            output.arr = np.zeros(self.arr.shape)
            return output
        else:
            return self.__mul__(other)

    def __div__(self, other):
        # Implements division using the / operator.
        bothAreImages = isinstance(other, self.__class__)
        oneIsInt      = (isinstance(other, int) or
                        isinstance(other, np.int8) or
                        isinstance(other, np.int16) or
                        isinstance(other, np.int32) or
                        isinstance(other, np.int64))
        oneIsFloat    = (isinstance(other, float) or
                        isinstance(other, np.float32) or
                        isinstance(other, np.float64))
        oneIsArray    = isinstance(other, np.ndarray)

        # TODO
        # I should include the possibility of operating with numpy array

        if bothAreImages:
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

                # Do the division
                output.arr = self.arr / other.arr

            else:
                print('Cannot divide images with different shapes')
                return None

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

                output.sigma = np.abs(output.arr *
                               np.sqrt((self.sigma/self.arr)**2 +
                                       (other.sigma/other.arr)**2))
            elif selfSig and not otherSig:
                output.sigma = self.sigma
            elif not selfSig and otherSig:
                output.sigma = other.sigma

            # Replace zeros
            if len(self0Inds[0]) > 0:
                self.arr[self0Inds] = self0s
            if len(other0Inds[0]) > 0:
                other.arr[other0Inds] = other0s
                output.arr[other0Inds] = np.nan

        elif oneIsInt or oneIsFloat:
            output     = self.copy()
            output.arr = self.arr / other

            if hasattr(self, 'sigma'):
                output.sigma = self.sigma / other

        else:
            print('Attempting to divide image by something weird...')
            return None

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
        oneIsInt      = isinstance(other, int)
        oneIsFloat    = isinstance(other, float)

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
        oneIsInt      = isinstance(other, int)
        oneIsFloat    = isinstance(other, float)

        if bothAreImages:
            # Check that image shapes make sense
            shape1     = self.arr.shape
            shape2     = other.arr.shape
            if shape1 == shape2:
                output     = self.copy()
                output.arr = (self.arr)**(other.arr)
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
                output.sigma = np.abs(output.arr) * np.sqrt(
                    (other.arr/self.arr*self.sigma)**2 +
                    (np.log(self.arr)*other.sigma)**2)
            elif selfSig and not otherSig:
                # The line below explodes when self.arr == 0
                # output.sigma = np.abs(output.arr*other.arr*self.sigma/self.arr)
                output.sigma = np.abs((self.arr**(other.arr - 1.0))*self.sigma)
            elif not selfSig and otherSig:
                print('const ** AstroImage not yet defined... not sure how?')

        elif oneIsInt or oneIsFloat:
            output = self.copy()
            output.arr = self.arr**other

            # propagate errors asuming ints and floats have no errors
            if hasattr(self, 'sigma'):
                # The line below explodes when self.arr == 0
                # output.sigma = np.abs(output.arr*other*self.sigma/self.arr)
                output.sigma = np.abs((self.arr**(other - 1.0))*self.sigma)
        else:
            print('Unexpected value in raising image to a power')

        # Finall return the exponentiated image
        return output

    ##################################
    ### END OF MAGIC METHODS       ###
    ### BEGIN OTHER COMMON METHODS ###
    ##################################

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

    def arctan2(self, x):
        '''A simple method for computing the unambiguous arctan of the image
        and propagating its uncertainty (if a sigma array has been defined).
        The "self" instance is treated as the y value. Another image, array, or
        scalar value must be passed as an argument to the method. This will be
        treated as the x value.
        '''
        # Check what type of variable has been passed as the x argument.
        bothAreImages = isinstance(x, self.__class__)
        oneIsInt      = isinstance(x, int)
        oneIsFloat    = isinstance(x, float)
        if bothAreImages:
            # Check that image shapes make sense
            shape1     = self.arr.shape
            shape2     = x.arr.shape
            if shape1 == shape2:
                outImg     = self.copy()
                outImg.arr = np.arctan2(self.arr, x.arr)

                # Perform error propagation
                selfSig = hasattr(self, 'sigma')
                xSig    = hasattr(x, 'sigma')
                if selfSig and xSig:
                    outImg.sigma = (np.sqrt((self.arr*x.sigma)**2 +
                                            (x.arr*self.sigma)**2) /
                                    (x.arr**2 + self.arr**2))
                elif selfSig and not xSig:
                    outImg.sigma = ((x.arr*self.sigma) /
                                    (x.arr**2 + self.arr**2))
                elif not selfSig and xSig:
                    outImg.sigma = ((self.arr*x.sigma) /
                                    (x.arr**2 + self.arr**2))

            else:
                print('Cannot subtract images with different shapes')
                return None

        elif oneIsInt or oneIsFloat:
            outImg     = self.copy()
            outImg.arr = np.arctan2(self.arr, x)

            # Perform error propagation
            if hasattr(self, 'sigma'):
                outImg.sigma = ((x.arr*self.sigma) /
                                (x.arr**2 + self.arr**2))
        else:
            print('Attempting to arctan with something weird')
            return None

        return outImg

    def sqrt(self):
        '''A simple method for computing the square root of the image and
        propagating its uncertainty (if a sigma array has been defined)
        '''

        return self**(0.5)

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

    def copy(self):
        # Make a copy of the image and return it to the user
        output = copy.deepcopy(self)

        return output

    def write(self, filename = '', dtype = None):
        # Test if a filename was provided and default to current filename
        if len(filename) == 0:
            filename = self.filename


        # Make copies of the output data
        outArr = self.arr.copy()
        outHead = self.header.copy()
        if hasattr(self, 'sigma'):
            outSig = self.sigma.copy()


        # If a data type was specified, recast output data into that format
        if dtype is not None:
            # First convert the array data
            try:
                outArr = outArr.astype(dtype)
            except:
                raise ValueError('dtype not recognized')

            # Next convert the sigma data if it exists
            if hasattr(self, 'sigma'):
                outSig = outSig.astype(dtype)

            # Now update the header to include that information
            # First parse which bitpix value is required
            #define BYTE_IMG      8  /*  8-bit unsigned integers */
            #define SHORT_IMG    16  /* 16-bit   signed integers */
            #define LONG_IMG     32  /* 32-bit   signed integers */
            #define LONGLONG_IMG 64  /* 64-bit   signed integers */
            #define FLOAT_IMG   -32  /* 32-bit single precision floating point */
            #define DOUBLE_IMG  -64  /* 64-bit double precision floating point */
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

        # Build a new HDU object to store the data
        arrHDU = fits.PrimaryHDU(data = outArr,
                                 header = outHead,
                                 do_not_scale_image_data=True)

        # Replace the original header (since some cards may have been stripped)
        arrHDU.header = self.header

        # If there is a sigma attribute,
        # then include it in the list of HDUs
        try:
            # Bulid a secondary HDU
            sigmaHDU = fits.ImageHDU(data = outSig,
                                     name = 'sigma',
                                     header = outHead,
                                     do_not_scale_image_data=True)
            HDUs = [arrHDU, sigmaHDU]
        except:
            HDUs = [arrHDU]

        # Build the final output HDUlist
        HDUlist = fits.HDUList(HDUs)

        # Write file to disk
        HDUlist.writeto(filename, clobber=True)

    def get_PSF(self, shape='gaussian'):
        """This method analyses the stars in the image and returns the PSF
        properties of the image. The default mode fits 2D-gaussians to the
        brightest, isolated stars in the image. Future versions could use there
        2MASS sersic profile, etc...
        """
        # Grab the image sky statistics
        mean, median, std = sigma_clipped_stats(self.arr, sigma=3.0, iters=5)

        # Start by finding all the stars in the image
        sources = daofind(self.arr - median, fwhm=3.0, threshold=15.0*std)

        # Eliminate stars near the image edge
        ny, nx = self.arr.shape
        xStars, yStars = sources['xcentroid'].data, sources['ycentroid'].data
        badXstars = np.logical_or(xStars < 50, xStars > nx - 50)
        badYstars = np.logical_or(yStars < 50,  yStars > ny - 50)
        edgeStars = np.logical_or(badXstars, badYstars)
        if np.sum(edgeStars) > 0:
            sources = sources[np.where(np.logical_not(edgeStars))]

        # Eliminate any stars with neighbors within 30 pixels
        keepFlags = np.ones_like(sources['flux'].data, dtype='bool')
        for i, star in enumerate(sources):
            # Compute the distance between this star and other stars
            xs, ys = star['xcentroid'], star['ycentroid']
            dist = np.sqrt((sources['xcentroid'].data - xs)**2 +
                           (sources['ycentroid'].data - ys)**2)

            # If there is another star within 20 pixels, then don't keep this
            if np.sum(dist < 30) > 1:
                keepFlags[i] = False

        # Cull the list of sources to only include "isolated" stars
        if np.sum(keepFlags) > 0:
            sources = sources[np.where(keepFlags)]
        else:
            print('No sources survided the neighbor test')
            pdb.set_trace()

        # Sort the stars by brightness
        sortInds = (sources['flux'].argsort())[::-1]
        sources = sources[sortInds]

        # Prep a local 2D gaussian function for fitting purposes
        # def gauss2d(xy, amp, x0, y0, a, b, c):
        #     x, y = xy
        #     inner = a * (x - x0)**2
        #     inner += 2 * b * (x - x0)**2 * (y - y0)**2
        #     inner += c * (y - y0)**2
        #     return amp * np.exp(-inner)

        def gauss2d(xy, base, height, center_x, center_y, width_x, width_y, rotation):
            """Returns a gaussian function with the given parameters"""
            # Parse the xy vector
            x, y     = xy

            # Ensure the parameters are floats
            width_x = float(width_x)
            width_y = float(width_y)

            xp = x - center_x
            yp = y - center_y

            # Convert rotation to radians and apply the rotation matrix
            # to center coordinates
            rotation = np.deg2rad(rotation)
            # center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
            # center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

            # Rotate the xy coordinates
            xp1 = xp * np.cos(rotation) - yp * np.sin(rotation)
            yp1 = xp * np.sin(rotation) + yp * np.cos(rotation)

            # Compute the gaussian values
            g = base + height*np.exp(-((xp1/width_x)**2 + (yp1/width_y)**2)/2.)
            return g

        # Fit 2D-gaussians to the 10 brightest, isolated stars
        yy, xx = np.mgrid[0:41, 0:41]
        sxList  = list()
        syList  = list()
        rotList = list()
        for i, star in enumerate(sources):
            # Fit a 2D gaussian to each star
            # First cut out a square array centered on the star
            xs, ys = star['xcentroid'], star['ycentroid']
            lf = np.int(xs.round()) - 20
            rt = lf + 41
            bt = np.int(ys.round()) - 20
            tp = bt + 41
            starArr = self.arr[bt:tp,lf:rt]

            # Package xy, zobs for fitting
            xy   = np.array([xx.ravel(), yy.ravel()])
            zobs = starArr.ravel()

            # Guess the initial parameters and perform the fit
            #              base, amp,   xc,   yc,   sx,  sy,  rot
            arrMax      = np.max(zobs)
            base1       = np.median(zobs)
            guessParams = [base1,    arrMax,   21.0,   21.0,   2.0,  2.0,   0.0]
            boundParams = ((-np.inf, 0.0,    -100.0, -100.0,   0.1,  0.1,   0.0),
                           (+np.inf, np.inf, +100.0, +100.0,  10.0, 10.0, 360.0))
            try:
                fitParams, uncert_cov = opt.curve_fit(gauss2d, xy, zobs,
                    p0=guessParams, bounds=boundParams)
            except:
                print("Star {0} could not be fit".format(i))

            # Test goodness of fitCenter
            fitX, fitY = fitParams[2:4]
            fitCenter  = np.sqrt((21 - fitParams[2])**2 + (21 - fitParams[3])**2)
            goodStar   = ((fitCenter < 3.0) and
                          (fitParams[4] > 0.5) and (fitParams[4] < 3.0) and
                          (fitParams[5] > 0.5) and (fitParams[5]) < 3.0)

            if goodStar:
                # Store the relevant parameters
                sxList.append(fitParams[4])
                syList.append(fitParams[5])
                rotList.append(fitParams[6])
            # else:
            #     # Check output
            #     print('Star {0} has bad fit'.format(i))
            #     plt.ion()
            #     plt.figure()
            #     plt.imshow(starArr)
            #
            #     test = gauss2d(xy, *fitParams)
            #     test.shape = xx.shape
            #     plt.figure()
            #     plt.imshow(test)
            #     pdb.set_trace()
            #     plt.close('all')

        # Compute an average gaussian shape and return to user
        PSFparams = (np.median(sxList), np.median(syList), np.median(rotList))

        return PSFparams

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



    def scale(self, copy=False):
        """Scales the data in the arr attribute using the BSCALE and BZERO
        values from the header. If no such values exist, then return original
        array.
        """
        # Scale the array
        scaledArr = self.arr.copy()
        keys = self.header.keys()
        if 'BSCALE' in keys:
            scaledArr = self.header['BSCALE']*scaledArr
        if 'BZERO' in keys:
            scaledArr = scaledArr + self.header['BZERO']

        # Check if a copy of the image was requested
        if copy:
            outImg = self.copy()
            outImg.arr = scaledArr
            return outImg
        else:
            self.arr = scaledArr

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
            wcs     = WCS(self.header)

            # Update the NAXIS values
            outHead['NAXIS1'] = nx1
            outHead['NAXIS2'] = ny1

            # Update the CRPIX values
            outHead['CRPIX1'] = (self.header['CRPIX1'] + 0.5)*xratio - 0.5
            outHead['CRPIX2'] = (self.header['CRPIX2'] + 0.5)*yratio - 0.5
            if wcs.wcs.has_cd():
                # Attempt to use CD matrix corrections, first
                # Apply updates to CD valus
                thisCD = wcs.wcs.cd
                # TODO set CDELT value properly in the "astrometry" step
                outHead['CD1_1'] = thisCD[0,0]/xratio
                outHead['CD1_2'] = thisCD[0,1]/yratio
                outHead['CD2_1'] = thisCD[1,0]/xratio
                outHead['CD2_2'] = thisCD[1,1]/yratio
            elif wcs.wcs.has_pc():
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
        original shape. If 'copy' is True, then the method will return a new
        copy of the image with its array rebinned. Otherwise, the image will be
        rebinned in place.
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
            rebinWts = wts.reshape(sh).mean(-1).mean(1)
            rebinArr = (tmpArr.reshape(sh).mean(-1).mean(1))/rebinWts
            rebinSig = np.sqrt(1.0/rebinWts)

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
            wcs     = WCS(self.header)

            # Update the NAXIS values
            outHead['NAXIS1'] = nx1
            outHead['NAXIS2'] = ny1

            # Update the CRPIX values
            outHead['CRPIX1'] = (self.header['CRPIX1'] + 0.5)*xratio - 0.5
            outHead['CRPIX2'] = (self.header['CRPIX2'] + 0.5)*yratio - 0.5
            if wcs.wcs.has_cd():
                # Attempt to use CD matrix corrections, first
                # Apply updates to CD valus
                thisCD = wcs.wcs.cd
                # TODO set CDELT value properly in the "astrometry" step
                outHead['CD1_1'] = thisCD[0,0]/xratio
                outHead['CD1_2'] = thisCD[0,1]/yratio
                outHead['CD2_1'] = thisCD[1,0]/xratio
                outHead['CD2_2'] = thisCD[1,1]/yratio
            elif wcs.wcs.has_pc():
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

    def pad(self, pad_width, mode=None, **kwargs):
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
            if 'WCSAXES' in tmpHeader.keys():
                # Parse the pad_width parameter
                if len(pad_width) > 1:
                    # If separate x and y paddings were specified, check them
                    xPad, yPad = pad_width
                    # Grab only theh left-padding values
                    if len(xPad) > 1: xPad = xPad[0]
                    if len(yPad) > 1: yPad = yPad[0]
                else:
                    xPad, yPad = pad_width, pad_width

                # Now apply the actual updates
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
            print('bad crop values')
            return None

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

        # Check if the X shift is an within 1 billianth of an integer value
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
        # Grab the WCS from the header
        thisWCS = WCS(self.header)

        # Transform coordinates to pixel positions
        x, y = coords.to_pixel(thisWCS)

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

    def align(self, img, fractionalShift=False, mode='wcs',
              offsets=False):
        """A method to align the self image with an other image
        using the astrometry from each header to shift an INTEGER
        number of pixels.

        parameters:
        img             -- the image with which self will be aligned
        fractionalShift -- if True, then images are shifted
                           to be aligned with sub-pixel precision
        mode            -- ['wcs' | 'cross_correlate']
                           defines the method used to align the two images
        """
        #**********************************************************************
        # It is unclear if this routine can handle images of different size.
        # It definitely assumes an identical plate scale...
        # Perhaps I needto be adjusting for each of these???
        #**********************************************************************

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
        if newImg.arr.shape != newSelf.arr.shape: pdb.set_trace()

        if mode == 'wcs':
            # Grab self image WCS and pixel center
            wcs1   = WCS(self.header)
            wcs2   = WCS(img.header)
            x1     = np.mean([wcs1.wcs.crpix[0], wcs2.wcs.crpix[0]])
            y1     = np.mean([wcs1.wcs.crpix[1], wcs2.wcs.crpix[1]])

            # Convert pixels to sky coordinates
            RA1, Dec1 = wcs1.all_pix2world(x1, y1, 0)

            # Grab the WCS of the alignment image and convert back to pixels
            x2, y2 = wcs2.all_world2pix(RA1, Dec1, 0)
            x2, y2 = float(x2), float(y2)

            # Compute the image possition offset vector
            dx = x2 - x1
            dy = y2 - y1

            # Compute the padding amounts. This is a non-trivial process due to
            # the way that odd vs. even is determined in python 3.x.
            if (np.int(np.round(dx)) % 2) == 1:
                padX = np.int(np.round(np.abs(dx))/2 + 1)
            else:
                padX = np.int(np.round(np.abs(dx))/2)

            if (np.int(np.round(dy)) % 2) == 1:
                padY = np.int(np.round(np.abs(dy))/2 + 1)
            else:
                padY = np.int(np.round(np.abs(dy))/2)

            # Construct the before-after padding combinations
            if dx > 0:
                selfDX = (np.int(np.round(np.abs(dx)-padX)), padX)
            else:
                selfDX = (padX, np.int(np.round(np.abs(dx)-padX)))

            if dy > 0:
                selfDY = (np.int(np.round(np.abs(dy)-padY)), padY)
            else:
                selfDY = (padY, np.int(np.round(np.abs(dy)-padY)))

            imgDX = selfDX[::-1]
            imgDY = selfDY[::-1]

            # Compute the shifting amount
            if fractionalShift:
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
            selfPadWidth = np.array((selfDY, selfDX), dtype=np.int)
            imgPadWidth  = np.array((imgDY,  imgDX), dtype=np.int)

            # Create copy images
            newSelf = self.copy()
            newImg  = img.copy()

            # Compute the padding to be added and pad the arr and sigma arrays
            newSelf.pad(selfPadWidth, mode='constant')
            newImg.pad(imgPadWidth, mode='constant')

            # Account for un-balanced padding with an initial shift left or down
            initialXshift = selfPadWidth[1,0] - imgPadWidth[1,0]
            if initialXshift > 0:
                newSelf.shift(-initialXshift, 0)
            elif initialXshift < 0:
                newImg.shift(-initialXshift, 0)

            initialYshift = selfPadWidth[0,0] - imgPadWidth[0,0]
            if initialYshift > 0:
                newSelf.shift(0, -initialYshift)
            elif initialYshift < 0:
                newImg.shift(0, -initialYshift)

            # Update header info
            # New header may already be correct, but no harm in double checking.
            newSelf.header['NAXIS1'] = newSelf.arr.shape[1]
            newSelf.header['NAXIS2'] = newSelf.arr.shape[0]
            newImg.header['NAXIS1']  = newImg.arr.shape[1]
            newImg.header['NAXIS2']  = newImg.arr.shape[0]

            # Shift the images
            newSelf.shift(selfShiftX, selfShiftY)
            newImg.shift(imgShiftX, imgShiftY)

            # Save the total offsets
            # TODO check that this is correct
            dx_tot, dy_tot = selfShiftX - imgShiftX, selfShiftY - imgShiftY

        elif mode == 'cross_correlate':
            """
            n. b. This method appears to produce results accurate to better
            than 0.1 pixel as determined by simply copying an image, shifting
            it an arbitrary amount, and attempting to recover that shift.
            """
            # Do an array flipped convolution, which is a correlation.
            corr = signal.fftconvolve(newSelf.arr, newImg.arr[::-1, ::-1], mode='same')

            # Do a little post-processing to block out bad points in corr image
            # First filter with the median
            binX, binY = self.binning
            medianKernShape = np.int(np.ceil(9.0/binX)), np.int(np.ceil(9.0/binY))
            medCorr = median_filter(corr, size = medianKernShape)

            # Compute sigma_clipped_stats of the correlation image
            mean, median, stddev = sigma_clipped_stats(corr)

            # Then check for significant deviations from median.
            deviations = (np.abs(corr - medCorr) > 2.0*stddev)

            # Count the number of masked neighbors for each pixel
            neighborCount = np.zeros_like(corr, dtype=int)
            for dx in range(-1,2,1):
                for dy in range(-1,2,1):
                    neighborCount += np.roll(np.roll(deviations, dy, axis=0),
                                             dx, axis=1).astype(int)

            # Find isolated deviant pixels (these are no good!)
            deviations = np.logical_and(deviations, neighborCount <= 4)

            # Inpaint those deviations
            tmp  = AstroImage()
            tmp.arr = corr
            tmp.binning = (1,1)
            tmp1 = image_tools.inpaint_nans(tmp.arr, mask = deviations)
            corr = tmp1

            # Check for the maximum of the cross-correlation function
            peak1  = np.unravel_index(corr.argmax(), corr.shape)
            dy, dx = np.array(peak1) - np.array(corr.shape)//2

            # Apply the initial (integer) shifts to the images
            img1 = newImg.copy()
            img1.shift(dx, dy)

            # Combine images to find the brightest 25 (or 16, or 9 stars in the image)
            comboImg = newSelf + img1

            # Get the image statistics
            mean, median, std = sigma_clipped_stats(comboImg.arr, sigma=3.0, iters=5)

            # Find the stars in the images
            sources = daofind(comboImg.arr - median, fwhm=3.0, threshold=5.*std)

            # Sort the sources lists by brightness
            sortInds = np.argsort(sources['mag'])
            sources  = sources[sortInds]

            # Remove detections within 20 pixels of the image edge
            # (This guarantees that the star-cutout process will succeed)
            ny, nx   = comboImg.arr.shape
            goodX    = np.logical_and(sources['xcentroid'] > 20,
                                     sources['xcentroid'] < (nx - 20))
            goodY    = np.logical_and(sources['ycentroid'] > 20,
                                     sources['ycentroid'] < (ny - 20))
            goodInds = np.where(np.logical_and(goodX, goodY))
            sources  = sources[goodInds]

            # Cull the saturated stars from the list
            numStars = len(sources)
            yy, xx   = np.mgrid[0:ny,0:nx]
            badStars = []

            for iStar in range(numStars):
                # Grab pixels less than 10 from the star
                xStar, yStar = sources[iStar]['xcentroid'], sources[iStar]['ycentroid']
                starDist  = np.sqrt((xx - xStar)**2 + (yy - yStar)**2)
                starPatch = comboImg.arr[np.where(starDist < 10)]

                # Test if there are bad pixels within 10 from the star
                # numBadPix = np.sum(np.logical_or(starPatch > 12e3, starPatch < -100))
                numBadPix = np.sum(starPatch < -100)

                # Append the test result to the "badStars" list
                badStars.append(numBadPix > 0)

            sources = sources[np.where(np.logical_not(badStars))]

            # Cull the list to the brightest few stars
            numStars = len(sources)
            if numStars > 25:
                numStars = 25
            elif numStars > 16:
                numStars = 16
            elif numStars > 9:
                numStars = 9
            else:
                print('There are not very many stars. Is something wrong?')
                pdb.set_trace()

            sources = sources[0:numStars]

            # Chop out the sections around each star,
            # and build a "starImage"
            starCutout  = 20
            numZoneSide = np.int(np.round(np.sqrt(numStars)))
            starImgSide = starCutout*numZoneSide
            starImg1 = np.zeros((starImgSide, starImgSide))
            starImg2 = np.zeros((starImgSide, starImgSide))
            # Loop through each star to be cut out
            iStar = 0
            for xZone in range(numZoneSide):
                for yZone in range(numZoneSide):
                    # Grab the star positions
                    xStar = np.round(sources['xcentroid'][iStar])
                    yStar = np.round(sources['ycentroid'][iStar])

                    # Establish the cutout bondaries
                    btCut = np.int(np.round(yStar - np.floor(0.5*starCutout)))
                    tpCut = np.int(np.round(btCut + starCutout))
                    lfCut = np.int(np.round(xStar - np.floor(0.5*starCutout)))
                    rtCut = np.int(np.round(lfCut + starCutout))

                    # Establish the pasting boundaries
                    btPaste = np.int(np.round(starCutout*yZone))
                    tpPaste = np.int(np.round(starCutout*(yZone + 1)))
                    lfPaste = np.int(np.round(starCutout*xZone))
                    rtPaste = np.int(np.round(starCutout*(xZone + 1)))

                    # Chop out the star and place it in the starImg
                    #    (sqrt-scale cutouts (~SNR per pixel) to emphasize alignment
                    #    of ALL stars not just bright stars).
                    # Apply accurate flooring of values at 0 (rather than simply using np.abs)
                    starImg1[btPaste:tpPaste, lfPaste:rtPaste] = np.sqrt(np.abs(newSelf.arr[btCut:tpCut, lfCut:rtCut]))
                    starImg2[btPaste:tpPaste, lfPaste:rtPaste] = np.sqrt(np.abs(img1.arr[btCut:tpCut, lfCut:rtCut]))

                    # Increment the star counter
                    iStar += 1

            # Do an array flipped convolution, which is a correlation.
            corr  = signal.fftconvolve(starImg1, starImg2[::-1, ::-1], mode='same')
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
            grad    = np.sqrt(Gx**2 + Gy**2)

            # Fill in edges to remove artifacts
            gradMax      = np.max(grad)
            grad[0:3, :] = gradMax
            grad[:, 0:3] = gradMax
            grad[grad.shape[0]-3:grad.shape[0], :] = gradMax
            grad[:, grad.shape[1]-3:grad.shape[1]] = gradMax

            # Grab the index of the minimum
            yMin, xMin = np.unravel_index(grad.argmin(), grad.shape)

            # Chop out the central zone and grab the minimum of the gradient
            cenSz = 3
            bot   = yMin - cenSz//2
            top   = bot + cenSz
            lf    = xMin - cenSz//2
            rt    = lf + cenSz

            # Grab the region near the minima
            yy, xx   = np.mgrid[bot:top, lf:rt]
            Gx_plane = Gx[bot:top, lf:rt]
            Gy_plane = Gy[bot:top, lf:rt]

            # Fit planes to the x and y gradients...Gx
            px_init = models.Polynomial2D(degree=1)
            py_init = models.Polynomial2D(degree=1)
            #fit_p   = fitting.LevMarLSQFitter()
            fit_p   = fitting.LinearLSQFitter()
            px      = fit_p(px_init, xx, yy, Gx_plane)
            py      = fit_p(py_init, xx, yy, Gy_plane)

            # Solve these equations using NUMPY
            #0 = px.c0_0 + px.c1_0*xx_plane + px.c0_1*yy_plane
            #0 = py.c0_0 + py.c1_0*xx_plane + py.c0_1*yy_plane
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
            dx_tot = dx + dx1
            dy_tot = dy + dy1
        else:
            print('mode rot recognized')
            pdb.set_trace()

        # The image offsets have been computed, so return the requested data
        if offsets:
            # Return the image offsets
            return [dx_tot, dy_tot]
        else:
            #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            # TODO
            # IN THE FUTURE THERE SHOULD BE SOME PADDING ADDED TO PREVENT DATA
            # LOSS
            #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            # Apply the final shifts
            newSelf.shift(-0.5*dx_tot, -0.5*dy_tot)
            newImg.shift(+0.5*dx_tot, +0.5*dy_tot)

            # Retun the aligned Images (not the same size as the input images)
            return [newSelf, newImg]

    def fix_astrometry(self):
        """This ensures that the CDELT values and PC matrix are properly set.
        """

        # Check if there is a header in this image
        if hasattr(self, 'header'):
            wcs        = WCS(self.header)
            pix_scales = proj_plane_pixel_scales(wcs)
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
            print('kernel value not recognized')
            pdb.set_trace()

        return (Gx, Gy)

    def show(self, axes=None, origin='lower', noShow=False,
             scale='linear', vmin=None, vmax=None,
             ticks=True, **kwargs):
        """Displays the image to the user for interaction (including clicking?)
        This method includes all the same keyword arguments as the "imshow()"
        method from matplotlip.pyplot. This allows the user to control how the
        image is displayed.

        Additional keyword arguments include
        scale -- ['linear' | 'log' | 'asinh']
                 allows the user to specify if if the image stretch should be
                 linear or log space
        """

        # Set the scaling for the image
        if scale == 'linear':
            # Compute a display range in terms of the image noise level
            showArr = self.arr
            mean, median, stddev = sigma_clipped_stats(self.arr.flatten())
            if vmin == None: vmin = median - 2*stddev
            if vmax == None: vmax = median + 10*stddev
        elif scale == 'log':
            showArr = np.log10(self.arr)
            if vmin == None:
                vmin = -6
            else:
                vmin = np.log10(vmin)
            if vmax == None:
                vmax = np.max(showArr)
            else:
                vmax = np.log10(vmax)

            print(vmin)
            print(vmax)
            print(np.nanmin(showArr))
            print(np.nanmax(showArr))
        elif scale == 'asinh':
            showArr = np.arcsinh(self.arr)
            if vmin == None:
                vmin = np.min(showArr)
            else:
                vmin = np.arcsinh(vmin)
            if vmax == None:
                vmax = np.max(showArr)
            else:
                vmax = np.arcsinh(vmax)
        else:
            print('The provided "scale" keyword is not recognized')
            pdb.set_trace()

        # Create the figure and axes for displaying the image
        if axes is None:
            # TODO add a check for WCS values

            # Create a new figure and axes
            wcs  = WCS(self.header)
            fig  = plt.figure(figsize = (8,8))
            if wcs.has_celestial:
                axes = fig.add_subplot(1,1,1, projection=wcs)

                # Set the axes linewidth
                axes.coords.frame.set_linewidth(2)

                # Label the axes establish minor ticks.
                RA_ax  = axes.coords[0]
                Dec_ax = axes.coords[1]
                RA_ax.set_axislabel('RA [J2000]',
                                    fontsize=12, fontweight='bold')
                Dec_ax.set_axislabel('Dec [J2000]',
                                     fontsize=12, fontweight='bold', minpad=-0.4)

                # Set tick labels
                RA_ax.set_major_formatter('hh:mm')
                Dec_ax.set_major_formatter('dd:mm')

                # Set the tick width and length
                RA_ax.set_ticks(size=12, width=2)
                Dec_ax.set_ticks(size=12, width=2)

                # Set the other tick label format
                RA_ax.set_ticklabel(fontsize=12, fontweight='demibold')
                Dec_ax.set_ticklabel(fontsize=12, fontweight='demibold')

                # Turn on minor ticks
                RA_ax.display_minor_ticks(True)
                RA_ax.set_minor_frequency(6)
                Dec_ax.display_minor_ticks(True)
                RA_ax.set_minor_frequency(6)
            else:
                axes = fig.add_subplot(1,1,1)
            # Set the axes line properties
            for axis in ['top','bottom','left','right']:
                axes.spines[axis].set_linewidth(4)

            # # Label the axes establish minor ticks.
            # RA_ax  = axes.coords[0]
            # Dec_ax = axes.coords[1]
            # RA_ax.set_axislabel('RA [J2000]',
            #                     fontsize=12, fontweight='bold')
            # Dec_ax.set_axislabel('Dec [J2000]',
            #                      fontsize=12, fontweight='bold', minpad=-0.4)
            #
            # # Set tick labels
            # RA_ax.set_major_formatter('hh:mm')
            # Dec_ax.set_major_formatter('dd:mm')
            #
            # # Set the tick width and length
            # RA_ax.set_ticks(size=12, width=2)
            # Dec_ax.set_ticks(size=12, width=2)
            #
            # # Set the other tick label format
            # RA_ax.set_ticklabel(fontsize=12, fontweight='demibold')
            # Dec_ax.set_ticklabel(fontsize=12, fontweight='demibold')
            #
            # # Turn on minor ticks
            # RA_ax.display_minor_ticks(True)
            # RA_ax.set_minor_frequency(6)
            # Dec_ax.display_minor_ticks(True)
            # RA_ax.set_minor_frequency(6)

            # Put the image in its place
            axIm = axes.imshow(showArr, origin=origin, vmin=vmin, vmax=vmax,
                               **kwargs)
        else:
            # Use the provided axes
            fig  = axes.figure
            axIm = axes.imshow(showArr, origin=origin, vmin=vmin, vmax=vmax,
                               **kwargs)

        # Display the image to the user, if requested
        if not noShow:
            plt.ion()
            fig.show()
            plt.ioff()

        # Return the graphics objects to the user
        return (fig, axes, axIm)

class Bias(AstroImage):
    """A subclass of the "Image" class: stores bias images and provides some
    methods for bias type operations.
    """

    def __init__(self, filename = ''):
        super(Bias, self).__init__(filename)

    def average(self):
        return np.mean(self.arr)

    def master_bias(biasList, clipSigma = 3.0):
        return image_tools.stacked_average(biasList, clipSigma = clipSigma)

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
        masterOverscan = image_tools.stacked_average(overscanList)

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
        return image_tools.stacked_average(flatList, clipSigma = clipSigma)

class Dark(AstroImage):
    """A subclass of the "Image" class: stores dark frames and provides some
    methods for dark type operations.
    """

    def __init(self, filename = ''):
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
        avgImg   = image_tools.stacked_average(darkList, clipSigma = clipSigma)
        darkTime = Dark.dark_time(darkList)
        return avgImg/darkTime
