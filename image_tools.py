import numpy as np
import psutil
import warnings
import subprocess
import os
import sys
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, proj_plane_pixel_scales
from scipy.odr import *
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import median_filter, gaussian_filter
from photutils import daofind, Background, detect_sources
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord, ICRS

# Import AstroImage in order to check types...
import AstroImage

# Import pdb for debugging
import pdb

def get_img_offsets(imgList, subPixel=False, mode='wcs'):
    """A function to compute the offsets between images using either the WCS
    values contained in each image header or using cross-correlation techniques
    with an emphasis on star alignment for sub-pixel accuracy.

    parameters:
    imgList  -- the list of images to be aligned.
    subPixel -- this boolean flag determines whether to round image offsets to
                the nearest integer value.
    mode     -- ['wcs' | 'cross_correlate'] the method to be used for
                aligning the images in imgList. 'wcs' uses the astrometry
                in the header while 'cross_correlate' selects a reference
                image and computes image offsets using cross-correlation.
    """
    # Catch the case where a list of images was not passed
    if not isinstance(imgList, list):
        raise ValueError('imgList variable must be a list of images')

    # Catch the case where imgList has only one image
    if len(imgList) <= 1:
        print('Must have more than one image in the list to be aligned')
        return (0, 0)

    # Catch the case where imgList has only two images
    if len(imgList) == 2:
        return imgList[0].get_img_offsets(imgList[1],
            subPixel=subPixel, mode=mode)

    #**********************************************************************
    # Get the offsets using whatever mode was selected
    #**********************************************************************
    if mode.lower() == 'wcs':
        # Compute the relative position of each of the images in the stack
        wcs1      = WCS(imgList[0].header)
        x1, y1    = imgList[0].arr.shape[1]//2, imgList[0].arr.shape[0]//2

        # Append the first image coordinates to the list
        shapeList = [imgList[0].arr.shape]
        imgXpos   = [float(x1)]
        imgYpos   = [float(y1)]

        # Convert pixels to sky coordinates
        skyCoord1 = pixel_to_skycoord(x1, y1, wcs1,
            origin=0, mode='wcs', cls=None)

        # Loop through all the remaining images in the list
        # Grab the WCS of the alignment image and convert back to pixels
        for img in imgList[1:]:
            wcs2   = WCS(img.header)
            x2, y2 = wcs2.all_world2pix(skyCoord1.ra, skyCoord1.dec, 0)
            shapeList.append(img.arr.shape)
            imgXpos.append(float(x2))
            imgYpos.append(float(y2))

    elif mode.lower() == 'cross_correlate':
        # Begin by selecting a reference image.
        # This should be the image with the BROADEST PSF. To determine this,
        # Let's grab the PSFparams of all the images and store the geometric
        # mean of sx, sy the best-fit Gaussian eigen-values.
        PSFsize = []
        for img in imgList:
            PSFparams, _ = img.get_psf()
            PSFsize.append(np.sqrt(PSFparams['sx']*PSFparams['sy']))

        # Use the first image in the list as the "reference image"
        refInd    = np.int((np.where(PSFsize == np.max(PSFsize)))[0])
        otherInds = (np.where(PSFsize != np.max(PSFsize)))[0]
        refImg    = imgList[refInd]

        # Initalize empty lists for storing offsets and shapes
        shapeList = []
        imgXpos   = []
        imgYpos   = []

        # Loop through the rest of the images.
        # Use cross-correlation to get relative offsets,
        # and accumulate image shapes
        for img in imgList:
            if img is refImg:
                # Just append null values for the reference image
                shapeList.append(refImg.arr.shape)
                imgXpos.append(0.0)
                imgYpos.append(0.0)
            else:
                # Compute actual image offset between reference and image
                dx, dy = refImg.get_img_offsets(img,
                    mode='cross_correlate',
                    subPixel=subPixel)

                # Append cross_correlation values for non-reference image
                shapeList.append(img.arr.shape)
                imgXpos.append(dx)
                imgYpos.append(dy)
    else:
        raise ValueError('Mode not recognized')

    # Center the image offsets about the median vector
    # Compute the median pointing
    x1 = np.median(imgXpos)
    y1 = np.median(imgYpos)

    # Compute the relative pointings from the median position
    dx = x1 - np.array(imgXpos)
    dy = y1 - np.array(imgYpos)

    # Compute the each distance from the median pointing
    imgDist   = np.sqrt(dx**2.0 + dy**2.0)
    centerImg = np.where(imgDist == np.min(imgDist))[0][0]

    # Set the "reference image" to the one closest to the median pointing
    x1, y1 = imgXpos[centerImg], imgYpos[centerImg]

    # Recompute the offsets from the reference image
    # (add an 'epsilon' shift to make sure ALL images get shifted
    # at least a tiny bit... this guarantees the images all get convolved
    # by the pixel shape.)
    dx = x1 - np.array(imgXpos)
    dy = y1 - np.array(imgYpos)

    # Return the image offsets
    return (dx, dy)

def align_images(imgList, padding=0, mode='wcs', subPixel=False, offsets=None):
    """A function to align the a whole stack of images using the astrometry
    from each header or cross-correlation techniques to align all the images
    so that image addition, subtraction, etc... works out.

    N.B. (2016-06-29) This function *DOES NOT* math image PSFs.
    Perhaps this functionality will be implemented in future versions

    parameters:
    imgList  -- the list of images to be aligned.
    padding  -- the value to use for padding the edges of the aligned
                images. Common values are 0 and NaN.
    mode     -- ['wcs' | 'cross_correlate'] the method to be used for
                aligning the images in imgList. 'wcs' uses the astrometry
                in the header while 'cross_correlate' selects a reference
                image and computes image offsets using cross-correlation.
    subPixel -- this boolean flag indicates whether or not the images should be
                shifted to within a sub-pixel accuracy (generally not possible
                using WCS, but sometimes possible using 'cross_correlate')
    offsets  -- True (default) returns aligned images.
                False returns image offsets.
    """
    # Catch the case where a list of images was not passed
    if not isinstance(imgList, list):
        raise ValueError('imgList variable must be a list of images')

    # Catch the case where imgList has only one image
    if len(imgList) <= 1:
        print('Must have more than one image in the list to be aligned')
        return imgList[0]

    # Catch the case where imgList has only two images
    if len(imgList) == 2:
        return imgList[0].align(imgList[1],
            mode=mode, subPixel=subPixel, offsets=offsets)

    # Check if a list of offsets was supplied and if they make sense
    if offsets is None:
        # If no offsets were supplid, then retrieve them using above function
        offsets = get_img_offsets(imgList,
            mode=mode, subPixel=subPixel)
    # Check if there are both x and y offsets provided
    elif hasattr(offsets, '__iter__') and len(offsets) == 2:
        # Check if they make sense...
        if (len(offsets[0]) == len(offsets[1])
            and len(offsets[0]) == len(imgList)):
            pass
        else:
            raise ValueError('there must be exactly one offset pair per image')
    else:
        raise ValueError('offsets keyword must be an iterable, 2 element offset')

    # Unpack the computed (or provided) image offsets
    # imgXpos, imgYpos = offsets
    dx, dy = offsets

    if subPixel == True:
        # Make sure that ALL images get shifted at least a small amount.
        # Shifting by a non-integer amount convolves the image with the pixel
        # shape, so we need to make sure there are no integer value shifts
        epsilon = 1e-4
        dx += epsilon
        dy += epsilon

        # Check for perfect integer shifts
        for dx1, dy1 in zip(dx, dy):
            if dx1.is_integer(): pdb.set_trace()
            if dy1.is_integer(): pdb.set_trace()

    # Grab a list of image shapes and compute maximum input image dimensions
    shapeList = np.array([img.arr.shape for img in imgList])
    nyFinal   = np.max(shapeList[:,0])
    nxFinal   = np.max(shapeList[:,1])

    # Compute the total image padding necessary to fit the whole stack, and
    # check if these values are at all reasonable
    padLf     = np.int(np.ceil(np.abs(np.min(dx))))
    padRt     = np.int(np.ceil(np.max(dx)))
    padBot    = np.int(np.ceil(np.abs(np.min(dy))))
    padTop    = np.int(np.ceil(np.max(dy)))
    totalPadX = padLf  + padRt
    totalPadY = padBot + padTop

    # Test for sanity. If the shift amount is greater than the image size, then
    # these two images don't actually overlap... this is a "mosaicing" problem,
    # not an "alignment" problem.
    if ((totalPadX > nxFinal) or
        (totalPadY > nyFinal)):
        print('there is a problem with the alignment')
        pdb.set_trace()

    # Compute padding
    padX     = (padLf, padRt)
    padY     = (padBot, padTop)
    padWidth = np.array((padY,  padX), dtype=np.int)

    # Create an empty list to store the aligned images
    alignedImgList = []

    # Loop through each image and pad it accordingly
    for img, dx1, dy1 in zip(imgList, dx, dy):
        # Make a copy of the image
        newImg = img.copy()

        # Check if this image needs an initial padding to match final size, and
        # apply padding to ensure that all images START with the same shape.
        if (nyFinal, nxFinal) != img.arr.shape:
            padX1      = nxFinal - img.arr.shape[1]
            padY1      = nyFinal - img.arr.shape[0]
            initialPad = ((0, padY1), (0, padX1))
            newImg.pad(initialPad, mode='constant', constant_values=padding)

        # Apply the extra padding to prevent data loss in final shift
        newImg.pad(padWidth, mode='constant', constant_values=padding)

        # Shift the images to their final positions
        if subPixel:
            # If sub-pixel shifting was requested, then use it
            shiftX = dx1
            shiftY = dy1
        else:
            # otherwise just take the nearest integer shifting offset
            shiftX = np.int(np.round(dx1))
            shiftY = np.int(np.round(dy1))

        # Actually apply the shift (along with the error-propagation)
        newImg.shift(shiftX, shiftY, padding=padding)

        # Check that the header is already correct!
        # Update the header information
        if hasattr(newImg, 'header'):
            wcs = WCS(newImg.header)
            if wcs.has_celestial:
                # Update the CRPIX values
                newImg.header['CRPIX1'] = newImg.header['CRPIX1'] + padWidth[1][0]
                newImg.header['CRPIX2'] = newImg.header['CRPIX2'] + padWidth[0][0]

            # Update the image NAXIS values
            newImg.header['NAXIS1'] = newImg.arr.shape[1]
            newImg.header['NAXIS2'] = newImg.arr.shape[0]

        # Append the shifted image
        alignedImgList.append(newImg)

    # If offsets were also requested, return them as a tupple after image list
    if offsets == True:
        # Build the tupple of offsets
        if subPixel:
            # (Use floats if subpixels shifts were requested)
            offs = (dy, dx)
        else:
            # (Use rounded integers if integer shifts were requested)
            offs = ((dy.round()).astype(np.int), (dx.round()).astype(np.int))

        return alignedImgList, offs
    else:
        return alignedImgList

def combine_images(imgList, output = 'MEAN',
                   bkgClipSigma = 5.0, starClipSigma = 40.0, iters=5,
                   weighted_mean = False,
                   mean_bkg = 0.0, effective_gain = None, read_noise = None):
    """Compute the median filtered mean of a stack of images.
    Standard deviation is computed from the variance of the stack of
    pixels.

    parameters:
    imgList        -- a list containing Image class objects.
    output         -- [MEAN', 'SUM']
                      the desired array to be returned be the function.
    bkgClipSigma   -- the level at which to trim outliers in the relatively
                      flat background (default = 5.0)
    starClipSigma  -- the level at which to trim outliers within a bright, star
                      PSF region (default = 40.0). This number should be quite
                      large because the naturally varrying PSF will cause large
                      deviations, which should not actually be rejected.
    weighted_mean  -- this flag indicates whether to use the uncertainties
                      contained in each AstroImage instance to compute a final
                      weighted average image. If marked as True, then each image
                      in the imgList is supposed
    mean_bkg       -- this is the mean background level that was subtracted from
                      the images in imgList. This should be ADDED back into the
                      counts used to estimate pixel uncertainty using
                      "effective_gain" and "read_noise".
    effective_gain -- the conversion factor from image counts to electrons
                      (units of electrons/count). This number will be used in
                      estimating the Poisson contribution to the uncertainty
                      in each pixel flux.
    read_noise     -- the estimated read-noise contribution to the uncertainty
                      of the flux in each pixel.
    """
    # Check that a proper output has been requested
    if not (output.upper() == 'MEAN' or output.upper() == 'SUM'):
        raise ValueError('The the keyword "output" must be "SUM" or "MEAN"')

    if weighted_mean == True:
        # Check that all the images have a sigma array
        for img in imgList:
            if not hasattr(img, 'sigma'):
                raise ValueError('All images must have a sigma array')

    # Count the number of images to be combined
    numImg = len(imgList)
    print('\nEntered averaging method')
    if numImg > 1:
        # Test for the correct number of bits in each pixel
        dataType    = imgList[0].dtype
        if dataType == np.int16:
            numBits = 16
        elif (dataType == np.int32) or (dataType == np.float32):
            numBits = 32
        elif (dataType == np.int64) or (dataType == np.float64):
            numBits = 64

        # Compute the number of pixels that fit under the memory limit.
        memLimit    = (psutil.virtual_memory().available/
                      (numBits*(1024**2)))
        memLimit    = int(10*np.floor(memLimit/10.0))
        numStackPix = memLimit*(1024**2)*8/numBits
        ny, nx      = imgList[0].arr.shape
        numRows     = int(np.floor(numStackPix/(numImg*nx)))
        if numRows > ny: numRows = ny
        numSections = int(np.ceil(ny/numRows))

        # Recompute the number of rows to be evenly spaced
        numRows = int(np.ceil(ny/numSections))

        # Compute "star mask" for this stack of images
        print('\nComputing masks for bright sources')
        # Grab binning
        binX, binY = imgList[0].binning

        # Compute kernel shape
        medianKernShape = (np.ceil(9.0/binX), np.ceil(9.0/binY))

        # Initalize a final star mask
        starMask = np.zeros((ny,nx), dtype=int)

        # Loop through the images and compute individual star masks
        for imgNum, img in enumerate(imgList):
            print('Building star mask for image {0:g} of {1:g}'.format(imgNum + 1, numImg), end='\r')
            # Grab the image array
            thisArr = img.arr.copy()

            # Replace bad values with zeros
            badInds = np.where(np.logical_not(np.isfinite(thisArr)))
            thisArr[badInds] = 0

            # Filter the image
            medImg = median_filter(thisArr, size = medianKernShape)

            # get stddev of image background
            mean, median, stddev = sigma_clipped_stats(thisArr)

            # Look for deviates from the filter (positive values only)
            starMask1 = np.logical_and(np.abs(thisArr - medImg) > 2.0*stddev,
                                       thisArr > 0)

            # Count the number of masked neighbors for each pixel
            neighborCount = np.zeros_like(thisArr, dtype=int)
            for dx in range(-1,2,1):
                for dy in range(-1,2,1):
                    neighborCount += np.roll(np.roll(starMask1, dy, axis=0),
                                             dx, axis=1).astype(int)

            # Find pixels with more than two masked neighbor (including self)
            starMask1 = np.logical_and(starMask1, neighborCount > 2)

            # Accumulate these pixels into the final star mask
            starMask += starMask1

        # Cleanup temporary variables
        del thisArr, badInds, medImg

        # Compute final star mask based on which pixels were masked more than
        # 10% of the time.
        starMask = (starMask > np.ceil(0.1*numImg)).astype(float)

        # Check that at least one star was detected (more than 15 pixels masked)
        if np.sum(starMask) > 15:
            # Now smooth the star mask with a gaussian to dialate it
            starMask1 = gaussian_filter(starMask, (4, 4))

            # Grab any pixels (and indices) above 0.05 value post-smoothing
            starMask  = (starMask1 > 0.05)
            numInStarPix = np.sum(starMask)

            # Notify user how many "in-star pixels" were masked
            print('\n\nMasked a total of {0} pixels'.format(numInStarPix))
        else:
            print('No pixels masked as "in-star" pixels')
            starMask[:,:] = False

        # Compute the number of subsections and display stats to user
        print('\nAiming to fit each stack into {0:g}MB of memory'.format(memLimit))
        print('\nBreaking stack of {0:g} images into {1:g} sections of {2:g} rows'
            .format(numImg, numSections, numRows))

        # Initalize an array to store the final averaged image
        outputImg = np.zeros((ny,nx))

        if weighted_mean == True:
            # Initalize an array to store the uncertainty of the weighted mean
            outputSig = np.ones((ny, nx))

        # Determine if uncertainty should be handled at all
        compute_uncertainty = ((effective_gain is not None) and
                               (read_noise is not None))
        # # Initalize a final uncertainty array
        # if compute_uncertainty:
        #     sigmaImg  = np.zeros((ny,nx))

        # Compute the stacked output of each section
        # Begin by computing how to iterate through sigma clipping
        bkgSigmaStep   = 0.5
        starSigmaStep  = 1.0
        bkgSigmaStart  = bkgClipSigma - bkgSigmaStep*iters
        starSigmaStart = starClipSigma - starSigmaStep*iters

        # Double check that these values are legal
        # (otherwise adjust sigmaStep values)
        if bkgSigmaStart < 0.1:
            bkgSigmaStart = 0.5
            bkgSigmaStep  = (bkgClipSigma - bkgSigmaStart)/iters
        if starSigmaStart < 30.0:
            starSigmaStart = 30.0
            starSigmaStep  = (starClipSigma - starSigmaStart)/iters

        # Loop through each "row section" of the stack
        for thisSec in range(numSections):
            # Calculate the row numbers for this section
            thisRows = (thisSec*numRows,
                        min([(thisSec + 1)*numRows, ny]))

            # Stack the selected region of the images.
            secRows = thisRows[1] - thisRows[0]
            stack   = np.ma.zeros((numImg, secRows, nx), dtype = dataType)
            for i in range(numImg):
                stack[i,:,:] = imgList[i].arr[thisRows[0]:thisRows[1],:]

            # Stack the selection region of the sigma arrays if needed
            if weighted_mean == True:
                stackSig = np.ma.zeros((numImg, secRows, nx), dtype = dataType)
                for i in range(numImg):
                    stackSig[i,:,:] = imgList[i].sigma[thisRows[0]:thisRows[1],:]

            # Catch and mask any NaNs or Infs (or -1e6 values)
            # before proceeding with the average
            NaNsOrInfs  = np.logical_not(np.isfinite(stack.data))
            stack.mask  = NaNsOrInfs
            stack.data[np.where(NaNsOrInfs)] = -1e6
            # Complement the NaNs search with bad pix value search
            badPix      = stack < -1e5
            NaNsOrInfs  = np.logical_or(NaNsOrInfs, badPix)
            stack.mask  = NaNsOrInfs

            # Now that the bad values have been saved,
            # replace them with signal "bad-data" values
            stack.data[np.where(NaNsOrInfs)] = -1*(10**6)

            print('\nAveraging rows {0[0]:g} through {0[1]:g}'.format(thisRows))

            # Iteratively clip outliers until answer converges.
            # Use the stacked median for first image estimate.
            outliers = np.zeros(stack.shape, dtype = bool)

            # This loop will iterate until the mask converges to an
            # unchanging state, or until clipSigma is reached.
            numPoints     = np.zeros((secRows, nx), dtype=int) + numImg
            scale         = (np.logical_not(starMask)*bkgSigmaStart +
                             starMask*starSigmaStart)
            for iLoop in range(iters):
                print('\tProcessing section for (bkgSigma, starSigma) = ({0:3.2g}, {1:3.2g})'.format(
                    bkgSigmaStart + bkgSigmaStep*iLoop,
                    starSigmaStart + starSigmaStep*iLoop))

                # Loop through the stack, and find the outliers.
                imgEstimate = np.ma.median(stack, axis = 0).data
                stackSigma  = np.ma.std(stack, axis = 0).data

                for j in range(numImg):
                    deviation       = np.absolute(stack.data[j,:,:] - imgEstimate)
                    outliers[j,:,:] = (deviation > scale*stackSigma)

                # Save the newly computed outliers to the mask
                stack.mask = np.logical_or(outliers, NaNsOrInfs)
                # Save the number of unmasked points along AXIS
                numPoints1 = numPoints
                # Total up the new number of unmasked pixels in each column
                numPoints  = np.sum(np.logical_not(stack.mask), axis = 0)
                # Figure out which pixel columns have improved results
                nextScale  = (numPoints != numPoints1).astype(int)

                if np.sum(nextScale) == 0:
                    # If there are no new data included, then break out of loop
                    break
                else:
                    # Otherwise increment scale where new data are included
                    scale = (scale +
                             nextScale*np.logical_not(starMask)*bkgSigmaStep +
                             nextScale*starMask*starSigmaStep)

            # Compute the final output image.
            if weighted_mean == True:
                # Compute as a weighted mean of the stack
                stackSig.mask = stack.mask.copy()
                weights       = 1.0/(stackSig**2.0)
                weightedSum   = np.ma.sum(weights*stack, axis = 0)
                sumOfWeights  = np.ma.sum(weights, axis = 0)
                tmpOut        = weightedSum/sumOfWeights

                # also compute this portion of the uncertainty image
                tmpSig = np.sqrt(1.0/sumOfWeights)
            else:
                # Compute as unweighted mean or sum
                if output.upper() == 'SUM':
                    # Figure out where there is no data and prevent divide-by-zero
                    denominator = numPoints.copy()
                    noSamples   = numPoints == 0
                    denominator[np.where(noSamples)] = numImg

                    # Compute the apropriate scaling up for the sum total
                    scaleFactor = (float(numImg)/denominator.astype(float))
                    tmpOut      = scaleFactor*np.sum(stack, axis = 0)

                    # Replace the values where we have no data with "NaN"
                    tmpOut[np.where(noSamples)] = np.NaN

                if output.upper() == 'MEAN':
                    # Compute the mean of the unmasked values
                    tmpOut = np.ma.mean(stack, axis = 0)

            # Place the output and uncertainty in their holders
            outputImg[thisRows[0]:thisRows[1],:] = tmpOut.data

            # Use the weighted uncertainty if requested
            if weighted_mean == True:
                outputSig[thisRows[0]:thisRows[1],:] = tmpSig.data

        # Get ready to return an AstroImage object to the user
        outImg = imgList[0].copy()
        outImg.arr = outputImg

        # Include the weighted uncertainty if requested
        if weighted_mean == True:
            outImg.sigma = outputSig

        # Compute uncertainty and store values
        if compute_uncertainty:
            # Find those pixels where no flux seems to be detected
            if output.upper() == 'SUM':
                poisCounts = outputImg + numImg*mean_bkg
                rn_mask    = (poisCounts <= (numImg*read_noise/effective_gain))
                sigmaImg   = np.sqrt(np.abs(poisCounts/effective_gain) +
                    numPoints*(read_noise/effective_gain)**2)
            if output.upper() == 'MEAN':
                poisCounts = outputImg + mean_bkg
                rn_mask    = poisCounts <= (read_noise/effective_gain)
                sigmaImg   = np.sqrt(np.abs(poisCounts/effective_gain) +
                    (read_noise/effective_gain)**2)/np.sqrt(numImg - 1)

            # Apply read_noise floor to the dimmest points in the image
            if np.sum(rn_mask) > 0:
                rn_Inds = np.where(rn_mask)
                sigmaImg[rn_Inds] = read_noise/effective_gain

            # Store the output uncertainty in the sigma attribute
            outImg.sigma = sigmaImg

        # Update the image shape in the header
        outImg.header['NAXIS1'] = outImg.arr.shape[1]
        outImg.header['NAXIS2'] = outImg.arr.shape[0]

        # Compute an "average" air-mass to attribute to the final image
        airMasses = [img.header['AIRMASS'] for img in imgList]
        outImg.header['AIRMASS'] = np.mean(airMasses)

        # Finally return the final result
        return outImg
    else:
        return imgList[0]

def astrometry(img, override = False):
    """A method to invoke astrometry.net and solve the astrometry of the image.
    """
    #######################
    # TODO THIS NEEDS TO BE RE-WRITTEN SO THAT IT WORKS AS A FUNCTION
    # (NOT A METHOD OF THE ASTRO-IMAGE CLASS)
    #######################
    # Test if the astrometry has already been solved
    try:
        # Try to grab the 'WCSAXES' card from the header
        tmp = img.header['WCSAXES']

        # If the user forces an override, then set doAstrometry=True
        doAstrometry = override
    except:
        # If there was no 'WCSAXES' card, then set doAstrometry=True
        doAstrometry = True

    if doAstrometry:
        # First test if the
        proc = subprocess.Popen(['where', 'solve-field'],
                                stdout=subprocess.PIPE,
                                universal_newlines=True)
        astrometryEXE = ((proc.communicate())[0]).rstrip()
        proc.terminate()

        if len(astrometryEXE) == 0:
            raise OSError('Astrometry.net is not properly installed.')

        # Make a copy of the image to be returned
        img1 = img.copy()

        # Test what kind of system is running
        if 'win' in sys.platform:
            # If running in Windows,
            # then define the "bash --login -c (cd ...)" command
            # using Cygwin's "cygpath" to convert to POSIX format
            proc = subprocess.Popen(['cygpath', os.getcwd()],
                                    stdout=subprocess.PIPE,
                                    universal_newlines=True)
            curDir = ((proc.communicate())[0]).rstrip()
            proc.terminate()

            # Convert filename to Cygwin compatible format
            proc = subprocess.Popen(['cygpath', img.filename],
                                    stdout=subprocess.PIPE,
                                    universal_newlines=True)
            inFile = ((proc.communicate())[0]).rstrip()
            proc.terminate()
            prefix = 'bash --login -c ("cd ' + curDir + '; '
            suffix = '")'
            delCmd = 'del '
            shellCmd = True
        else:
            # If running a *nix system,
            # then define null prefix/suffix strings
            inFile = img.filename
            prefix = ''
            suffix = ''
            delCmd = 'rm '
            shellCmd = False

        # Setup the basic input/output command options
        outputCmd    = ' --out tmp'
        noPlotsCmd   = ' --no-plots'
        overwriteCmd = ' --overwrite'
#            dirCmd       = ' --dir debug'
        dirCmd = ''

        # Provide a guess at the plate scale
        scaleLowCmd  = ' --scale-low 0.25'
        scaleHighCmd = ' --scale-high 1.8'
        scaleUnitCmd = ' --scale-units arcsecperpix'

        # Provide some information about the approximate location
        raCmd        = ' --ra ' + img.header['TELRA']
        decCmd       = ' --dec ' + img.header['TELDEC']
        radiusCmd    = ' --radius 0.3'

        # This is reduced data, so we won't need to clean up the image
#            imageOptions = '--no-fits2fits --no-background-subtraction'
        imageOptions = ''

        # Prevent writing any except the "tmp.wcs" file.
        # In the future it may be useful to set '--index-xyls'
        # to save star coordinates for photometry.
        noOutFiles = ' --axy none --corr none' + \
                     ' --match none --solved none' + \
                     ' --new-fits none --rdls none' + \
                     ' --solved none --index-xyls none'

        # Build the final command
        command      = 'solve-field' + \
                       outputCmd + \
                       noPlotsCmd + \
                       overwriteCmd + \
                       dirCmd + \
                       scaleLowCmd + \
                       scaleHighCmd + \
                       scaleUnitCmd + \
                       raCmd + \
                       decCmd + \
                       radiusCmd + \
                       imageOptions + \
                       noOutFiles + \
                       ' ' + inFile

        # Run the command in the terminal
        astroProc = subprocess.Popen(prefix + command +suffix)
        astroProc.wait()
        astroProc.terminate()
        # os.system(prefix + command + suffix)

        # Construct the path to the newly created WCS file
        filePathList = img.filename.split(os.path.sep)
        if len(filePathList) > 1:
            wcsPath = os.path.dirname(img.filename) + os.path.sep + 'tmp.wcs'
        else:
            wcsPath = 'tmp.wcs'

        # Read in the tmp.wcs file and create a WCS object
        if os.path.isfile(wcsPath):
            HDUlist = fits.open(wcsPath)
            HDUlist[0].header['NAXIS'] = img.header['NAXIS']
            wcsObj = WCS(HDUlist[0].header)
            HDUlist.close()

            # Double make sure that there is no other WCS data in the header
            if 'WCSAXES' in img1.header.keys():
                del img1.header['WCSAXES']

            if len(img1.header['CDELT*']) > 0:
                del img1.header['CDELT*']

            if len(img1.header['CUNIT*']) > 0:
                del img1.header['CUNIT*']

            if len(img1.header['*POLE']) > 0:
                del img1.header['*POLE']

            if len(img1.header['CD*_*']) > 0:
                del img1.header['CD*_*']

            if len(img1.header['PC*_*']) > 0:
                del img1.header['PC*_*']

            if len(img1.header['CRPIX*']) > 0:
                del img1.header['CRPIX*']

            if len(img1.header['CRVAL*']) > 0:
                del img1.header['CRVAL*']

            if len(img1.header['CTYPE*']) > 0:
                del img1.header['CTYPE*']

            # Grab the CD matrix from the WCS object
            CD1_1, CD1_2, CD2_1, CD2_2 = wcsObj.wcs.cd.flatten()

            # Grab the center pixels, values, units, and types
            CRPIX1, CRPIX2 = wcsObj.wcs.crpix
            CRVAL1, CRVAL2 = wcsObj.wcs.crval
            CTYPE1, CTYPE2 = wcsObj.wcs.ctype
            CTYPE1, CTYPE2 = CTYPE1[0:8], CTYPE2[0:8]

            # Grab the poles
            LATPOLE, LONPOLE = wcsObj.wcs.latpole, wcsObj.wcs.lonpole

            # Update the image header to contain the astrometry info
            img1.header['CD1_1']   = CD1_1
            img1.header['CD1_2']   = CD1_2
            img1.header['CD2_1']   = CD2_1
            img1.header['CD2_2']   = CD2_2
            img1.header['CRPIX1']  = CRPIX1
            img1.header['CRPIX2']  = CRPIX2
            img1.header['CRVAL1']  = CRVAL1
            img1.header['CRVAL2']  = CRVAL2
            img1.header['CTYPE1']  = CTYPE1
            img1.header['CTYPE2']  = CTYPE2
            img1.header['LATPOLE'] = LATPOLE
            img1.header['LONPOLE'] = LONPOLE

            # Cleanup the none and WCS file,
            rmProc = subprocess.Popen(delCmd + wcsPath, shell=shellCmd)
            rmProc.wait()
            rmProc.terminate()
            noneFile = os.path.join(os.getcwd(), 'none')
            rmProc = subprocess.Popen(delCmd + noneFile, shell=shellCmd)
            rmProc.wait()
            rmProc.terminate()

            # If everything has worked, then return a True success value
            return img1, True
        else:
            # If there was no WCS, then return a False success value
            return None, False
    else:
        print('Astrometry for {0:s} already solved.'.
          format(os.path.basename(img.filename)))
        return img, True

def build_starMask(arr, sigmaThresh = 2.0, neighborThresh = 2, kernelSize = 9):
    '''This function will idenify in-star pixels using a local median-filtered.
    This should work for both clean (e.g. PRISM) images and dirty (e.g. Mimir)
    images. The usual "detect_sources" method will identify dirty features as
    false positives, so this method is preferable.
    '''

    # Compute kernel shape
    medianKernShape = (kernelSize, kernelSize)

    # Filter the image
    medImg = median_filter(arr, size = medianKernShape)

    # get stddev of image background
    mean, median, stddev = sigma_clipped_stats(arr)

    # Look for deviates from the filter (positive values only)
    starMask = np.logical_and(np.abs(arr - medImg) > sigmaThresh*stddev,
                               arr > 0)
    # Count the number of masked neighbors for each pixel
    neighborCount = np.zeros_like(arr, dtype=int)
    for dx in range(-1,2,1):
        for dy in range(-1,2,1):
            neighborCount += np.roll(np.roll(starMask, dy, axis=0),
                                     dx, axis=1).astype(int)

    # Subtract the self-counting pixel count
    neighborCount -= 1

    # Find pixels with more than two masked neighbor (including self)
    starMask = np.logical_and(starMask, neighborCount > neighborThresh)

    return starMask

def inpaint_nans(arr, mask=None):
    '''This function will take an numpy array object, inpaint the bad (NaN)
    pixels (or the pixels specified by the mask keyword argument if that is
    provided), and return a copy of the AstroImage with its reparied pixels

    inputs:
    arr  -- a numpy ndarray instance which needs repairs
    mask -- a boolean array containing True at pixels to be inpainted and False
    everywhere else.

    returns:
    outImg -- an AstroImage instance with repaired pixels
    '''

    # Test if the mask keyword argument was provided.
    if not isinstance(mask, type(None)):
        # If the mask has been defined, check if it's an array
        if isinstance(mask, np.ndarray):
            # If the mask is an array, check if it's a bool
            if mask.dtype == bool:
                # If the mask is a bool, check that it has the right size
                if arr.shape == mask.shape:
                    # Everything seems to be in order, so don't panic.
                    # Just make a copy of the mask to be used for inpainting.
                    badPix = mask.copy()
                else:
                    # If mask shape doesn't match image shape, then raise error
                    raise ValueError('mask must have same shape as arr.')
            else:
                # If mask is not bool, then raise an error
                raise ValueError('mask keyword argument must by type bool')
        else:
            # If mask is not an array, then raise an error
            raise ValueError('mask keyword argument must be an ndarray')
    else:
        # If there is no mask provided, then simply make one out of the NaNs
        badPix = np.isnan(arr)

    # If no pixels were selected for inpainting, just return a copy of the image
    if np.sum(badPix) == 0:
        return arr.copy()

    # First get the indices for the good and bad pixels
    goodInds = np.where(np.logical_not(badPix))
    badInds  = np.where(badPix)

    # Replace badInds with image median value
    repairedArr1 = arr.copy()
    mean, median, stddev = sigma_clipped_stats(repairedArr1[goodInds])
    repairedArr1[badInds] = median


    # # On first pass, smooth the input image with kernel ~5% of image size.
    # ny, nx       = arr.shape
    # kernelSize   = np.int(np.round(0.05*np.sqrt(nx*ny)))
    # repairedArr1 = gaussian_filter(repairedArr1, kernelSize)
    #
    # # Replace good pix with good values
    # repairedArr1[goodInds] = arr[goodInds]

    # Iterative kernel size
    iterKernelSize = 10

    # Loop through and keep smoothing the array
    meanDiff = 1.0
    while meanDiff > 0.1:
        # Smooth the image over with a smaller, 10 pixel kernel
        repairedArr = gaussian_filter(repairedArr1, iterKernelSize)

        # Immediately replace the good pixels with god values
        repairedArr[goodInds] = arr[goodInds]

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
        repairedArr = gaussian_filter(repairedArr1, iterKernelSize)

        # Immediately replace the good pixels with god values
        repairedArr[goodInds] = arr[goodInds]

        # Compute the mean pixel difference
        pixelDiffs  = np.abs(repairedArr1[badInds] - repairedArr[badInds])
        meanDiff    = np.mean(pixelDiffs)

        # Now that the image has been smoothed, swap out the saved array
        repairedArr1 = repairedArr

    # Return the actual AstroImage instance
    return repairedArr

def build_pol_maps(Qimg, Uimg):
    '''This function will build polarization percentage and position angle maps
    from the input Qimg and Uimg AstroImage instances. If the DEL_PA and
    S_DEL_PA header keywords are set (and match), then the position angle maps
    '''
    # Check if the U and Q images are the same shape
    if Qimg.arr.shape != Uimg.arr.shape:
        raise ValueError('The U and Q images must be the same shape')

    # Quickly build the P map
    Pmap  = np.sqrt(Qimg**2 + Uimg**2)

    # Catch any stupid values (nan, inf, etc...)
    badVals = np.logical_not(
              np.logical_and(
              np.isfinite(Pmap.arr),
              np.isfinite(Pmap.sigma)))

    # Replace the stupid values with zeros\
    badInds = np.where(badVals)
    if len(badInds[0]) > 0:
        Pmap.arr[badInds] = 0.0
        Pmap.sigma[badInds] = 0.0

    # Apply the Ricean correction
    # First compute a temporary "de-baised" array
    tmpArr = Pmap.arr**2 - Pmap.sigma**2

    # Check if any of the debiased values are less than zero
    zeroInds = np.where(tmpArr < 0)
    if len(zeroInds[0]) > 0:
        # Set all insignificant detections to zero
        tmpArr[zeroInds] = 0

    # Now we can safely take the sqare root of the debiased values
    Pmap.arr = np.sqrt(tmpArr)

    # Parse the header information for building the PA map
    # Check for a DELPA keyword in the headers
    Qhas_DPA = ('DELTAPA' in Qimg.header.keys())
    Uhas_DPA = ('DELTAPA' in Uimg.header.keys())

    # Retrieve the DELpa values
    if Qhas_DPA and Uhas_DPA:
        QDPA = Qimg.header['DELTAPA']
        UDPA = Uimg.header['DELTAPA']
        if QDPA == UDPA:
            deltaPA = QDPA
        else:
            print('DELTAPA values do not match.')
            pdb.set_trace()
    else:
        deltaPA = 0.0

    # Check if PA map needs to be made more uncertain...
    Qhas_s_DPA = 'S_DPA' in Qimg.header.keys()
    Uhas_s_DPA = 'S_DPA' in Uimg.header.keys()

    if Qhas_s_DPA and Uhas_s_DPA:
        Q_s_DPA = Qimg.header['S_DPA']
        U_s_DPA = Uimg.header['S_DPA']
        if Q_s_DPA == U_s_DPA:
            s_DPA = Q_s_DPA
        else:
            print('S_DPA values do not match.')
            pdb.set_trace()
    else:
        s_DPA = 0.0

    # Now read the astrometry from the header and add instrument-to-equatorial
    # rotation to the deltaPA value
    wcsQ = WCS(Qimg.header)
    wcsU = WCS(Uimg.header)

    # Check if both Q and U have astrometry
    if wcsQ.has_celestial and wcsU.has_celestial:
        # Check if the CD matrices are the same in both images
        if np.sum(np.abs((wcsQ.wcs.cd - wcsU.wcs.cd))) == 0:
            # Grab the cd matrix
            cd = wcsQ.wcs.cd
            # Check if the frames have non-zero rotation
            if cd[0,0] != 0 or cd[1,1] != 0:
                # Compute rotationg angles
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
                    print('Rotation angles do not agree!')
                    pdb.set_trace()

                # Check if the longitude pole is located where expected
                if wcsQ.wcs.lonpole != 180.0:
                    rotAng += (180.0 - wcsQ.wcs.lonpole)

                # Add rotation angle to final deltaPA
                deltaPA += rotAng
        else:
            print('The astrometry in U and Q do not seem to match.')
            pdb.set_trace()

    # Build the PA map and add the uncertaies in quadrature
    PAmap = (np.rad2deg(0.5*np.arctan2(Uimg, Qimg)) + deltaPA + 720.0) % 180.0
    if s_DPA > 0.0:
        PAmap.sigma = np.sqrt(PAmap.sigma**2 + s_DPA**2)

    return Pmap, PAmap

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


###### CHECK OUT ALTERNATIVE COLOR MAPPING POSSIBILITIES
# http://matplotlib.org/examples/pylab_examples/custom_cmap.html


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]

    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


#**** Example use of this function***
#
#colConv = mcolors.ColorConverter().to_rgb
#pdb.set_trace()
#rvb = make_colormap(
#   [c('red'), c('violet'), 0.33, c('violet'), c('blue'), 0.66, c('blue')])
# rvb = make_colormap(
#    [(1.0, 0.0, 0.0), (1.0, 0.5, 0.0), 1.0/12.0, # Red-Orange
#     (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), 2.0/12.0, # Orange-Yellow
#     (1.0, 1.0, 0.0), (0.5, 1.0, 0.0), 3.0/12.0, # Yellow-Spring Green
#     (0.5, 1.0, 0.0), (0.0, 1.0, 0.0), 4.0/12.0, # Spring Green-Green
#     (0.0, 1.0, 0.0), (0.0, 1.0, 0.5), 5.0/12.0, # Green-Turquoise
#     (0.0, 1.0, 0.5), (0.0, 1.0, 1.0), 6.0/12.0, # Turquoise-Cyan
#     (0.0, 1.0, 1.0), (0.0, 0.5, 1.0), 7.0/12.0, # Cyan-Ocean
#     (0.0, 0.5, 1.0), (0.0, 0.0, 1.0), 8.0/12.0, # Ocean-Blue
#     (0.0, 0.0, 1.0), (0.5, 0.0, 1.0), 9.0/12.0, # Blue-Violet
#     (0.5, 0.0, 1.0), (1.0, 0.0, 1.0), 10.0/12.0, # Violet-Magenta
#     (1.0, 0.0, 1.0), (1.0, 0.0, 0.5), 11.0/12.0, # Magenta-Raspbery
#     (1.0, 0.0, 0.5), (1.0, 0.0, 0.0), 12.0/12.0, # Raspbery-Red
#     (1.0, 0.0, 0.0)                              # Cap it off with red
#     ])
#
# N = 1000
# array_dg = np.random.uniform(0, 10, size=(N, 2))
# colors = np.random.uniform(-2, 2, size=(N,))
# plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=rvb)
# plt.colorbar()
# plt.show()
