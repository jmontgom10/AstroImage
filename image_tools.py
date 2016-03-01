import numpy as np
import psutil
import subprocess
import os
import sys
from astropy.io import fits
from wcsaxes import WCS
from astropy.wcs.utils import pixel_to_skycoord
from scipy.ndimage.filters import median_filter, gaussian_filter
from photutils import daofind
#from scipy.ndimage import generate_binary_structure, binary_dilation
from astropy.stats import sigma_clipped_stats

import pdb

def combine_images(imgList, output = 'SUM',
                   bkgClipSigma = 5.0, starClipSigma = 40.0, iters=5,
                   effective_gain = None, read_noise = None):
    """Compute the median filtered mean of a stack of images.
    Standard deviation is computed from the variance of the stack of
    pixels.

    parameters:
    imgList        -- a list containing Image class objects.
    output         -- ['SUM, 'MEAN']
                      the desired array to be returned be the function.
    bkgClipSigma   -- the level at which to trim outliers in the relatively
                      flat background (default = 6.0)
    starClipSigma  -- the level at which to trim outliers within a bright, star
                      PSF region (default = 30.0). This number should be quite
                      large because the naturally varrying PSF will cause large
                      deviations, which should not actually be rejected.
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
            print('Building star mask for image {0:g}'.format(imgNum + 1))
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

          # gradArr   = generic_gradient_magnitude(thisArr, sobel)
          # starMask1 = gradArr > 400 # Do I nueed to adjust for binning?

            # Accumulate these pixels into the final star mask
            starMask += starMask1

        # Cleanup temporary variables
        del thisArr, badInds#, medImg

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
            print('Masked a total of {0} pixels'.format(numInStarPix))
        else:
            print('No pixels masked as "in-star" pixels')
            starMask[:,:] = False

        # Compute the number of subsections and display stats to user
        print('\nAiming to fit each stack into {0:g}MB of memory'.format(memLimit))
        print('\nBreaking stack of {0:g} images into {1:g} sections of {2:g} rows'
            .format(numImg, numSections, numRows))

        # Initalize an array to store the final averaged image
        outputImg = np.zeros((ny,nx))

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

        # Get ready to return an AstroImage object to the user
        outImg = imgList[0].copy()
        outImg.arr = outputImg

        # Compute uncertainty and store values
        if compute_uncertainty:
            # Find those pixels where no flux seems to be detected
            if output.upper() == 'SUM':
                rn_mask = outputImg <= (numImg*read_noise/effective_gain)
                sigmaImg = np.sqrt(np.abs(outputImg/effective_gain) +
                    numPoints*(read_noise/effective_gain)**2)
            if output.upper() == 'MEAN':
                rn_mask = outputImg <= (read_noise/effective_gain)
                sigmaImg = np.sqrt(np.abs(outputImg/effective_gain) +
                    (read_noise/effective_gain)**2)

            # apply read_noise floor to the dimmest points in the image
            sigmaImg[rn_mask] = read_noise/effective_gain

            # Store the output uncertainty in the sigma attribute
            outImg.sigma = sigmaImg

        # Now that an average image has been computed,
        # Clear out the old astrometry
        if 'WCSAXES' in outImg.header.keys():
            del outImg.header['WCSAXES']
            del outImg.header['PC*']
            del outImg.header['CDELT*']
            del outImg.header['CUNIT*']
            del outImg.header['*POLE']
            outImg.header['CRPIX*'] = 1.0
            outImg.header['CRVAL*'] = 1.0
            outImg.header['CTYPE*'] = 'Linear Binned ADC Pixels'
            outImg.header['NAXIS1'] = outImg.arr.shape[1]
            outImg.header['NAXIS2'] = outImg.arr.shape[0]

        # Finally return the final result
        return outImg
    else:
        return imgList[0]

def astrometry(img, override = False):
    """A method to invoke astrometry.net
    and solve the astrometry of the image.
    """
    #######################
    # TODO THIS NEEDS TO BE RE-WRITTEN SO THAT IT WORKS AS A FUNCTION
    # (NOT A METHOD OF THE ASTRO-IMAGE CLASS)
    #######################
    # Test if the astrometry has already been solved
    try:
        # Try to grab the 'WCSAXES' card from the header
        img.header['WCSAXES']

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

            # Build a quick header from the WCS object
            wcsHead = wcsObj.to_header()

            # Update the image header to contain the astrometry info
            for key in wcsHead.keys():
                img1.header[key] = wcsHead[key]

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

def align_images(imgList, padding=0, mode='wcs', subPixel=False):
    """A method to align the a whole stack of images using the astrometry
    from each header to shift an INTEGER number of pixels.

    parameters:
    imgList -- the list of image to be aligned.
    padding -- the value to use for padding the edges of the aligned
               images. Common values are 0 and NaN.
    mode    -- ['wcs' | 'cross_correlate'] the method to be used for
               aligning the images in imgList. 'wcs' uses the astrometry
               in the header while 'cross_correlation' selects a reference
               image and computes image offsets using cross-correlation.
    """

    # Catch the case where a list of images was not passed
    if not isinstance(imgList, list):
        raise ValueError('imgList variable must be a list of images')

    # Catch the case where imgList has only one image
    if len(imgList) <= 1:
        print('Must have more than one image in the list to be aligned')
        return imgList[0]

    # Catch the case where imgList has only two images
    if len(imgList) <= 2:
        return imgList[0].align(imgList[1], mode=mode)

    #**********************************************************************
    # Get the offsets using whatever mode was selected
    #**********************************************************************
    if mode == 'wcs':
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

    elif mode == 'cross_correlate':
        # Use the first image in the list as the "reference image"
        refImg = imgList[0]

        # Initalize empty lists for storing offsets and shapes
        shapeList = [refImg.arr.shape]
        imgXpos   = [0.0]
        imgYpos   = [0.0]

        # Loop through the rest of the images.
        # Use cross-correlation to get relative offsets,
        # and accumulate image shapes
        for img in imgList[1:]:
            dx, dy = refImg.align(img, mode='cross_correlate', offsets=True)
            shapeList.append(img.arr.shape)
            imgXpos.append(-dx)
            imgYpos.append(-dy)
    else:
        print('mode not recognized')
        pdb.set_trace()

    # Make sure all the images are the same size
    shapeList = np.array(shapeList)
    nyFinal   = np.max(shapeList[:,0])
    nxFinal   = np.max(shapeList[:,1])

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
    epsilon = 1e-4
    dx = x1 - np.array(imgXpos) + epsilon
    dy = y1 - np.array(imgYpos) + epsilon

    # Check for integer shifts
    for dx1, dy1 in zip(dx, dy):
        if dx1.is_integer(): pdb.set_trace()
        if dy1.is_integer(): pdb.set_trace()

    # Compute the total image padding necessary to fit the whole stack
    padLf     = np.int(np.round(np.abs(np.min(dx))))
    padRt     = np.int(np.round(np.max(dx)))
    padBot    = np.int(np.round(np.abs(np.min(dy))))
    padTop    = np.int(np.round(np.max(dy)))
    totalPadX = padLf  + padRt
    totalPadY = padBot + padTop

    # Test for sanity
    if ((totalPadX > np.max(shapeList[:,1])) or
        (totalPadY > np.max(shapeList[:,0]))):
        print('there is a problem with the alignment')
        pdb.set_trace()

    # compute padding
    padX     = (padLf, padRt)
    padY     = (padBot, padTop)
    padWidth = np.array((padY,  padX), dtype=np.int)

    # Create an empty list to store the aligned images
    alignedImgList = []

    # Loop through each image and pad it accordingly
    for i in range(len(imgList)):
        # Make a copy of the image
        newImg     = imgList[i].copy()

        # Check if this image needs an initial padding to match final size
        if (nyFinal, nxFinal) != imgList[i].arr.shape:
            padX       = nxFinal - imgList[i].arr.shape[1]
            padY       = nyFinal - imgList[i].arr.shape[0]
            initialPad = ((0, padY), (0, padX))
            newImg.pad(initialPad, mode='constant', constant_values=padding)

        # Apply the more padding to prevent data loss in final shift
        newImg.pad(padWidth, mode='constant', constant_values=padding)

        # Shift the images to their final positions
        if subPixel:
            # If sub-pixel shifting was requested, then use it
            shiftX = dx[i]
            shiftY = dy[i]
        else:
            # otherwise just take the nearest integer shifting offset
            shiftX = np.int(np.round(dx[i]))
            shiftY = np.int(np.round(dy[i]))

        # Actually apply the shift (along with the error-propagation)
        newImg.shift(shiftX, shiftY)

        # Check that the header is already correct!
        # Update the header information
        newImg.header['CRPIX1'] = newImg.header['CRPIX1'] + padWidth[1][0]
        newImg.header['CRPIX2'] = newImg.header['CRPIX2'] + padWidth[0][0]
        newImg.header['NAXIS1'] = newImg.arr.shape[1]
        newImg.header['NAXIS2'] = newImg.arr.shape[0]

        # Append the shifted image
        alignedImgList.append(newImg)

    return alignedImgList

def build_pol_maps(Qimg, Uimg):
    ''' This function will build polarization percentage and position angle maps
    from the input Qimg and Uimg AstroImage instances. If the DEL_PA and
    S_DEL_PA header keywords are set (and match), then the position angle maps
    '''

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
    # First check for where the Pmap is an insignificant detection
    zeroInds = np.where(Pmap.arr <= Pmap.sigma)
    if len(zeroInds[0]) > 0:
        Pmap.arr[zeroInds] = Pmap.sigma[zeroInds]

    # Then actually de-bias the map.
    Pmap.arr = np.sqrt(Pmap.arr**2 - Pmap.sigma**2)

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
