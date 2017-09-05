# Scipy imports
import numpy as np

# Astropy imports
from astropy.wcs import WCS
from astropy.nddata import NDDataArray, StdDevUncertainty

__all__ = ['ResizingMixin']

class ResizingMixin(object):
    """
    A mixin class to handle common numerical methods for ReducedScience class
    """

    def pad(self, pad_width, mode, **kwargs):
        """
        Pads the image arrays and updates the header and astrometry.

        Parameters
        ----------
        pad_width: sequence, array_like, int
            Number of values padded to the edges of each axis.
            ((before_1, after_1), ... (before_N, after_N)) unique pad widths for
            each axis. ((before, after),) yields same before and after pad for
            each axis. (pad,) or int is a shortcut for before = after = pad
            width for all axes. The `pad_width` value in this method is
            identical to the `pad_width` value in the numpy.pad function.

        mode: str or function
            Sets the method by which the edges of the image are padded. This
            argument is directly passed along to the numpy.pad function, so
            see numpy.pad documentation for more information.

        Other parameters
        ----------------
        All keywords allowed for numpy.pad are also permitted for this method.
        See the numpy.pad documentation for a complete listing of keyword
        arguments and their permitted values.

        Returns
        -------
        outImg: `ReducedScience`
            Padded image with shape increased according to pad_width.
        """
        # AstroImages are ALWAYS 2D (at most!)
        if len(pad_width) > 2:
            raise ValueError('Cannot use a`pad_width` value with more than 2-dimensions.')

        # Make a copy of the image to return to the user
        outImg = self.copy()

        # Pad the primary array
        outData = np.pad(self.data, pad_width, mode, **kwargs)

        if self._BaseImage__fullData.uncertainty is not None:
            outUncert = np.pad(self.uncertainty, pad_width, mode, **kwargs)
            outUncert = StdDevUncertainty(outUncert)
        else:
            outUncert = None

        # Update the header information if possible
        outHeader = self.header.copy()

        # Parse the pad_width parameter
        if len(pad_width) > 1:
            # If separate x and y paddings were specified, check them
            yPad, xPad = pad_width

            # Grab only theh left-padding values
            if len(xPad) > 1: xPad = xPad[0]
            if len(yPad) > 1: yPad = yPad[0]
        else:
            xPad, yPad = pad_width, pad_width

        # Update image size
        outHeader['NAXIS1'] = self.shape[1]
        outHeader['NAXIS2'] = self.shape[0]

        # If the header has a valid WCS, then update that info, too.
        if self.has_wcs:
            if self.wcs.has_celestial:
                # Now apply the actual updates to the header
                outHeader['CRPIX1'] = self.header['CRPIX1'] + xPad
                outHeader['CRPIX2'] = self.header['CRPIX2'] + yPad

                # Retrieve the new WCS from the updated header
                outWCS = WCS(outHeader)
        else:
            outWCS = None

        # And store the updated header in the self object
        outImg._BaseImage__header = outHeader

        # Finally replace the _BaseImage__fullData attribute
        outImg._BaseImage__fullData = NDDataArray(
            outData,
            uncertainty=outUncert,
            unit=self.unit,
            wcs=outWCS
        )

        return outImg

    def crop(self, lowPix, highPix):
        # TODO use the self.wcs.wcs.sub() method to recompute the right wcs
        # for a cropped image.
        """
        Crops the image to the specified pixel locations.

        Parameters
        ----------
        lowPix : tuple
            The starting point of the crop along each axis.

        highPix : tuple
            The stopping point of the crop along each axis.

        Returns
        -------
        outImg: `ReducedScience`
            A copy of the image cropped to the specified locations with updated
            header and astrometry.
        """
        # Decompose the crop endpoints
        bt, lf = lowPix
        tp, rt = highPix

        for p in (bt, tp, lf, rt):
            if not issubclass(type(p), (int, np.int16, np.int32, np.int64)):
                TypeError('All arguments must be integer values')

        # Check that the crop values are reasonable
        ny, nx = self.shape
        if ((lf < 0) or (rt > nx) or
            (bt < 0) or (tp > ny) or
            (rt < lf) or (tp < bt)):
            raise ValueError('The requested crop values are outside the image.')

        # Make a copy of the array and header
        outData = self.data.copy()

        # Perform the actual croping
        outData = outData[bt:tp, lf:rt]

        # Repeat the process for the sigma array if it exists
        if self.uncertainty is not None:
            outUncert = self.uncertainty[bt:tp, lf:rt]
            outUncert = StdDevUncertainty(outUncert)
        else:
            outUncert = None

        # Copy the image header
        outHead = self.header.copy()

        # Update the header keywords
        # First update the NAXIS keywords
        outHead['NAXIS1'] = tp - bt
        outHead['NAXIS2'] = rt - lf

        # Next update the CRPIX keywords
        if 'CRPIX1' in outHead:
            outHead['CRPIX1'] = outHead['CRPIX1'] - lf
        if 'CRPIX2' in outHead:
            outHead['CRPIX2'] = outHead['CRPIX2'] - bt

        # Reread the WCS from the output header if it has a wcs
        if self.has_wcs:
            if self.wcs.has_celestial:
                outWCS = WCS(outHead)
        else:
            outWCS = None

        # Copy the image and update its data
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(
            outData,
            uncertainty=outUncert,
            unit=self.unit,
            wcs=outWCS
        )

        # Update the header, too.
        outImg.header = outHead

        return outImg

    def _rebin_wcs(self, outShape):
        """
        Applies a rebinning to the WCS parameters in the header.

        Parameters
        ----------
        outShape : tuple of ints
            The new shape for the rebinned image. This must be an integer factor
            of the shape of the original image, although the integer factor does
            not need to be the same along each axis.

        Returns
        -------
        outImg : `astroimage.reduced.ReducedScience` (or subclass)
            The rebinned image instance.
        """

        # Extract the shape and rebinning properties
        ny1, nx1 = self.shape
        ny, nx   = outShape
        dxdy     = np.array([nx1/nx, ny1/ny])

        # Copy the image
        outImg = self.copy()

        # Catch the case where there is no WCS to rebin
        if not self.has_wcs:
            return outImg

        # Now treat the WCS for images which have astrometry.
        # Recompute the CRPIX and place them in the header
        CRPIX1, CRPIX2 = self.wcs.wcs.crpix/dxdy
        outImg.header['CRPIX1'] = CRPIX1
        outImg.header['CRPIX2'] = CRPIX2

        # Grab the CD matrix
        if self.wcs.wcs.has_cd():
            # Grab the cd matrix and modify it by the rebinning factor
            cd = dxdy*self.wcs.wcs.cd

        elif self.wcs.wcs.has_pc():
            # Convert the pc matrix into a cd matrix
            cd = dxdy*self.wcs.wcs.cdelt*self.wcs.wcs.pc

            # Delete the PC matrix so that it can be replaced with a CD matrix
            del outImg.header['PC*']

        else:
            raise ValueError('`wcs` does not include proper astrometry')

        # Loop through the CD values and replace them with updated values
        for i, row in enumerate(cd):
            for j, cdij in enumerate(row):
                key = 'CD' + '_'.join([str(i+1), str(j+1)])
                outImg.header[key] = cdij

        # TODO: Verify that the SIP polynomial treatment is correct
        # (This may require some trial and error)

        # Loop through all possible coefficients, starting at the 2nd order
        # values, JUST above the linear (CD matrix) relations.
        for AB in ['A', 'B']:
            ABorderKey = '_'.join([AB, 'ORDER'])
            # Check if there is a distortion polynomial to handle.
            if ABorderKey in outImg.header:
                highestOrder = outImg.header[ABorderKey]
                # Loop through each order (2nd, 3rd, 4th, etc...)
                for o in range(2,highestOrder+1):
                    # Loop through each of the horizontal axis order values
                    for i in range(o+1):
                        # Compute the vertical axis order value for THIS order
                        j = o - i

                        # Compute the correction factor given the rebinning
                        # amount along each independent axis.
                        ABcorrFact = (dxdy[0]**i)*(dxdy[1]**j)

                        # Construct the key in which the SIP coeff is stored
                        ABkey = '_'.join([AB, str(i), str(j)])

                        # Update the SIP coefficient
                        outImg.header[ABkey] = ABcorrFact*self.header[ABkey]

            # Repeat this for the inverse transformation (AP_i_j, BP_i_j).
            APBP = AB + 'P'
            APBPorderKey = '_'.join([APBP, 'ORDER'])
            if APBPorderKey in outImg.header:
                highestOrder = outImg.header[APBPorderKey]
                # Start at FIRST order this time...
                for o in range(1, highestOrder+1):
                    for i in range(o+1):
                        j = o - i

                        # Skip the zeroth order (simply provided by CRVAL)
                        if i == 0 and j == 0: continue

                        # Compute the correction factor and apply it.
                        APBPcorrFact = (dxdy[0]**(-i))*(dxdy[1]**(-j))
                        APBPkey = '_'.join([APBP, str(i), str(j)])
                        outImg.header[APBPkey] = APBPcorrFact*self.header[APBPkey]

        # Store the updated WCS and return the image to the user
        outImg._BaseImage__fullData = NDDataArray(
            outImg.data,
            uncertainty=StdDevUncertainty(outImg.uncertainty),
            unit=outImg.unit,
            wcs=WCS(outImg.header)
        )

        return outImg

    def _rebin_bzero_bscale(self, outShape):
        """
        Applies a rebinning to the BZERO and BSCALE parameters in the header.

        Parameters
        ----------
        outShape : tuple of ints
            The new shape for the rebinned image. This must be an integer factor
            of the shape of the original image, although the integer factor does
            not need to be the same along each axis.

        Returns
        -------
        outImg : `astroimage.reduced.ReducedScience` (or subclass)
            The rebinned image instance.

        """

        # Extract the shape and rebinning properties
        ny1, nx1 = self.shape
        ny, nx   = outShape
        dxdy     = np.array([nx1/nx, ny1/ny])

        outImg = self.copy()

        bscale = self.header['BSCALE']
        if (bscale != 0) and (bscale != 1):
            outImg.header['BSCALE'] = ( bscale/pix_ratio, 'Calibration Factor')

        bzero = self.header['BZERO']
        if (bzero != 0):
            outImg.header['BZERO'] = (bzero/pix_ratio, 'Additive Constant for Calibration')

        return outImg

    def rebin(self, outShape, total=False):
        """
        Rebins the image to have a specified shape.

        Parameters
        ----------
        outShape : tuple of ints
            The new shape for the rebinned image. This must be an integer factor
            of the shape of the original image, although the integer factor does
            not need to be the same along each axis.

        total : bool, optional, default: False
            If set to true, then returned array is total of the binned pixels
            rather than the average.

        Returns
        -------
        outImg : `astroimage.reduced.ReducedScience` (or subclass)
            The rebinned image instance.
        """
        # Grab the shape of the initial array
        ny0, nx0 = self.shape
        ny,  nx  = outShape

        # TODO: Catch the case of upsampling along one axis but downsampling
        # along the other. This should not be possible!

        # Test for improper result shape
        goodX = ((nx0 % nx) == 0) or ((nx % nx0) == 0)
        goodY = ((ny0 % ny) == 0) or ((ny % ny0) == 0)
        if not (goodX and goodY):
            raise ValueError('Result dimensions must be integer factor of original dimensions')

        # Make a copy of the image to manipulate and return to the user
        outImg = self.copy()

        # First test for the trivial case
        if (nx0 == nx) and (ny0 == ny):
            return outImg

        # Compute the pixel ratios of upsampling and down sampling
        xratio, yratio = np.float(nx)/np.float(nx0), np.float(ny)/np.float(ny0)
        pixRatio       = np.float(xratio*yratio)
        aspect         = yratio/xratio         #Measures change in aspect ratio.

        if ((nx0 % nx) == 0) and ((ny0 % ny) == 0):
            # Handle integer downsampling
            # Get the new shape for the array and compute the rebinning shape
            sh = (ny, ny0//ny,
                  nx, nx0//nx)

            # Computed weighted rebinning
            nanMask   = np.isfinite(self.data).astype(float)
            rebinData = np.nansum(np.nansum(self.data.reshape(sh), axis=-1), axis=1)
            numInSum  = np.sum(np.sum(nanMask.reshape(sh), axis=-1), axis=1)

            # Only consider a rebinned pixel valid if *at least* 75% of the
            # constituent pixels were finite
            dy, dx  = ny0//ny, nx0//nx
            badPix  = (numInSum < 0.75*(dx*dy))
            badInds = np.where(badPix)
            rebinData[badInds] = np.NaN

            # Check if total flux conservation was requested.
            # If not, then multiply by the pixel size ratio.
            if not total: rebinData /= numInSum

        elif ((nx % nx0) == 0) and ((ny % ny0) == 0):
            # Handle integer upsampling
            rebinData = np.kron(
                self.data,
                np.ones((ny//ny0, nx//nx0))
            )

            # Check if total flux conservation was requested.
            # If it was, then divide by the pixel size ratio.
            if total: rebinData /= pixRatio

        # Compute the output uncertainty
        if self._BaseImage__fullData.uncertainty is not None:
            selfVariance = (self.uncertainty)**2
            if ((nx0 % nx) == 0) and ((ny0 % ny) == 0):
                # Handle integer downsampling
                rebinVariance = selfVariance.reshape(sh).sum(-1).sum(1)

                # Check if total flux conservation was requested.
                # If not, then multiply by the pixel size ratio.
                if total: pass
                if not total: rebinVariance *= (pixRatio**2)

            elif ((nx % nx0) == 0) and ((ny % ny0) == 0):
                # Handle integer upsampling
                rebinVariance = np.kron(
                    selfVariance,
                    np.ones((ny//ny0, nx//nx0))
                )

                # Check if total flux conservation was requested.
                # If not, then divide by the pixel size ratio.
                if total: rebinVariance /= pixRatio
                if not total: rebinVariance *= pixRatio

            # Convert the uncertainty into the correct class for NDDataArray
            rebinUncert = StdDevUncertainty(np.sqrt(rebinVariance))
        else:
            # Handle the no-uncertainty situation
            rebinUncert = None

        # Now apply header updates to the WCS parameters (BEFORE the current
        # image shape gets distored by replacing the __fullData value.)
        outImg = outImg._rebin_wcs(outShape)

        # Now apply the header updates to BSCALE and BZERO keywords
        if total:
            raise NotImplementedError('Need to implement "_rebin_bscale_bzero" method')
            outImg = outImg._rebin_bscale_bzero(outShape)

        # Construct the output NDDataArray
        rebinFullData = NDDataArray(
            rebinData,
            uncertainty=rebinUncert,
            unit=self._BaseImage__fullData.unit,
            wcs=outImg.wcs
        )

        # Store the rebinned FullData
        outImg._BaseImage__fullData = rebinFullData

        # Update the header values
        outHead = outImg.header.copy()
        outHead['NAXIS1'] = nx
        outHead['NAXIS2'] = ny

        # Store the header in the output image
        outImg._BaseImage__header = outHead

        # Update the binning attribute to match the new array
        outImg._BaseImage__binning = (
            outImg.binning[0]/xratio,
            outImg.binning[1]/yratio
        )

        # Return the updated image object
        return outImg

# def frebin(self, nx1, ny1, total=False):
#     """
#     Rebins the image to an arbitrary size using a flux conservative method.
#
#     Parameters
#     ----------
#     nx, ny : int
#         The number of pixels desired in the output image along the
#         horizontal axis (nx) and the vertical axis (ny).
#
#     total : bool
#         If true, then the output image will have the same number of counts
#         as the input image.
#     """
#
#     # TODO: rewrite this for the new ReducedScience user interface
#     raise NotImplementedError
#
#     # First test for the trivial case
#     ny, nx = self.shape
#     if (nx == nx1) and (ny == ny1):
#         if copy:
#             return self.copy()
#         else:
#             return
#
#     # Compute the pixel ratios of upsampling and down sampling
#     xratio, yratio = np.float(nx1)/np.float(nx), np.float(ny1)/np.float(ny)
#     pixRatio       = np.float(xratio*yratio)
#     aspect         = yratio/xratio         #Measures change in aspect ratio.
#
#     ###
#     # TODO: if dealing with integers, then simply pass to the REBIN method
#     ###
#     if ((nx % nx1) == 0) and ((ny % ny1) == 0):
#         # Handle integer downsampling
#         # Get the new shape for the array and compute the rebinning shape
#         sh = (ny1, ny//ny1,
#               nx1, nx//nx1)
#
#         # Make a copy of the array before any manipulation
#         tmpArr = (self.data.copy()).astype(np.float)
#
#         # Perform the actual rebinning
#         rebinArr = tmpArr.reshape(sh).mean(-1).mean(1)
#
#         # Check if total flux conservation was requested
#         if total:
#             # Re-normalize by pixel area ratio
#             rebinArr /= pixRatio
#
#     elif ((nx1 % nx) == 0) and ((ny1 % ny) == 0):
#         # Handle integer upsampling
#         # Make a copy of the array before any manipulation
#         tmpArr = (self.data.copy()).astype(np.float)
#
#         # Perform the actual rebinning
#         rebinArr   = np.kron(tmpArr, np.ones((ny1//ny, nx1//nx)))
#
#         # Check if total flux conservation was requested
#         if total:
#             # Re-normalize by pixel area ratio
#             rebinArr /= pixRatio
#
#     else:
#         # Handle the cases of non-integer rebinning
#         # Make a copy of the array before any manipulation
#         tmpArr = np.empty((ny1, nx), dtype=np.float)
#
#         # Loop along the y-axis
#         ybox, xbox = np.float(ny)/np.float(ny1), np.float(nx)/np.float(nx1)
#         for i in range(ny1):
#             # Define the boundaries of this box
#             rstart = i*ybox
#             istart = np.int(rstart)
#             rstop  = rstart + ybox
#             istop  = np.int(rstop) if (np.int(rstop) < (ny - 1)) else (ny - 1)
#             frac1  = rstart - istart
#             frac2  = 1.0 - (rstop - istop)
#
#             # Compute the values in each box
#             if istart == istop:
#                 tmpArr[i,:] = (1.0 - frac1 - frac2)*self.arr[istart, :]
#             else:
#                 tmpArr[i,:] = (np.sum(self.arr[istart:istop+1, :], axis=0)
#                                - frac1*self.arr[istart, :]
#                                - frac2*self.arr[istop, :])
#
#         # Transpose tmpArr and prepare to loop along other axis
#         tmpArr = tmpArr.T
#         result = np.empty((nx1, ny1))
#
#         # Loop along the x-axis
#         for i in range(nx1):
#             # Define the boundaries of this box
#             rstart = i*xbox
#             istart = np.int(rstart)
#             rstop  = rstart + xbox
#             istop  = np.int(rstop) if (np.int(rstop) < (nx - 1)) else (nx - 1)
#             frac1  = rstart - istart
#             frac2  = 1.0 - (rstop - istop)
#
#             # Compute the values in each box
#             if istart == istop:
#                 result[i,:] = (1.0 - frac1 - frac2)*tmpArr[istart, :]
#             else:
#                 result[i,:] = (np.sum(tmpArr[istart:istop+1, :], axis=0)
#                                - frac1*tmpArr[istart, :]
#                                - frac2*tmpArr[istop, :])
#
#         # Transpose the array back to its proper numpy style shape
#         rebinArr = result.T
#
#         # Check if total flux conservation was requested
#         if not total:
#             rebinArr *= pixRatio
#
#         # Check if there is a header needing modification
#         outHead = self.header.copy()
#
#         # Update the NAXIS values
#         outHead['NAXIS1'] = nx1
#         outHead['NAXIS2'] = ny1
#
#         # Update the CRPIX values
#         outHead['CRPIX1'] = (self.header['CRPIX1'] + 0.5)*xratio - 0.5
#         outHead['CRPIX2'] = (self.header['CRPIX2'] + 0.5)*yratio - 0.5
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
#     # If a copy was requested, then return a copy of the original image
#     # with a newly rebinned array
#     if outWCS.has_celestial:
#         outWCS = outWcs
#     else:
#         outWCS = None
#
#     outImg._BaseImage__fullData = NDDataArray(
#         rebinArr,
#         uncertainty=rebinUncert,
#         unit=outImg.unit
#          wcs=outWCS
#     )
#     outImg._BaseImage__header   = outHead
#     outBinning = (xratio*outImg.binning[0],
#                   yratio*outImg.binning[1])
#     outImg._dictionary_to_properties({'binning': outBinning})
#     outImg._properties_to_header()
#
#     return outImg

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

            >>> from astroimage.reduced import ReducedImage
            >>> img1 = ReducedImage(np.arange(100).reshape((10, 10)))
            >>> img1.shape
            (10, 10)
            >>> img2 = img1[1:9:2, 1:9:2]
            >>> img2.shape
            (4, 4)

        An *average* rebinning method is used, but that can be transformed to
        a total value through a simple multiplicative factor.

            >>> img1 = ReducedImage(np.arange(16).reshape((4, 4)))
            >>> img1.shape
            >>> dy, dx = 2, 2
            >>> avgImg = img1[1:9:dy, 1:9:dx]
            >>> totImg = avgImg*(dx*dy)
            >>> avgImg
            array([[  2.5,   4.5],
                   [ 11.5,  12.5]])
            >>> totImg
            array([[ 10.,  18.],
                   [ 46.,  50.]])
        """
        # Convert a singular slice into a tuple of slices
        if isinstance(key, slice):
            key1 = (key,)
        else:
            if len(key) > len(self.shape):
                raise IndexError('Too many indices for 2D image')
            key1 = key

        # Get the starting point of the slices
        startPix = [k.start if k.start is not None else 0
            for k in key]

        # Get the stopping point of the slices
        stopPix = [k.stop if k.stop is not None else nPix
            for k, nPix in zip(key, self.shape)]

        # Compute the rebinning factors along each axis
        stepPix = [k.step if k.step is not None else 1
            for k in key]

        # Test if all the slice values are integers before proceeding
        intClasses = (int, np.int8, np.int16, np.int32, np.int64)
        startsAreInts = all([isinstance(start, intClasses) for start in startPix])
        stopsAreInts  = all([isinstance(stop, intClasses) for stop in stopPix])
        stepsAreInts  = all([np.float(step).is_integer() for step in stepPix])
        stepsAreInts  = (stepsAreInts or
            all([(1.0/np.float(step)).is_integer() for step in stepPix]))
        if not (startsAreInts and stopsAreInts and stepsAreInts):
            raise ValueError('All start, stop[, step] values must be integers')

        # Get the proposed shape based on the provided slices
        cropShape = [(stop - start)
            for start, stop in zip(startPix, stopPix)]

        # Compute the number of remainder pixels along each axis at this binning
        remainderPix = [np.int(nPix % binning)
            for nPix, binning in zip(cropShape, stepPix)]

        # Recompute the crop boundaries using the rebinning factors
        cropShape = [propPix - remainPix
            for propPix, remainPix in zip(cropShape, remainderPix)]

        # Recompute the stopping point
        stopPix = [start + length
            for start, length in zip(startPix, cropShape)]

        # Crop the arrays
        outImg = self.crop(startPix, stopPix)

        # Rebin the arrays if necessary
        if any([b != 1 for b in stepPix]):
            rebinShape = tuple([
                np.int(cs/sp) for cs, sp in zip(cropShape, stepPix)
            ])
            outImg     = outImg.rebin(rebinShape)

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
