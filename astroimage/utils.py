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
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord, ICRS

# Import AstroImage in order to check types...
from . import astroimage

# Import pdb for debugging
import pdb

### DEFINE TWO FUNCTIONS TO BE USED IN TREATING PSF GAUSSIANS
def build_cov_matrix(sx, sy, rhoxy):
    # build the covariance matrix from sx, sy, and rhoxy
    cov_matrix = np.matrix([[sx**2,       rhoxy*sx*sy],
                            [rhoxy*sx*sy, sy**2      ]])

    return cov_matrix

# Define two functions for swapping between (sigma_x, sigma_y, theta) and
# (sigma_x, sigma_y, rhoxy)
def convert_angle_to_covariance(sx, sy, theta):
    # Define the rotation matrix using theta
    rotation_matrix = np.matrix([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta),  np.cos(theta)]])

    # Build the eigen value matrix
    lamda_matrix = np.matrix(np.diag([sx, sy]))

    # Construct the covariance matrix
    cov_matrix = rotation_matrix*lamda_matrix*lamda_matrix*rotation_matrix.I

    # Extract the variance and covariances
    sx1, sy1 = np.sqrt(cov_matrix.diagonal().A1)
    rhoxy    = cov_matrix[0,1]/(sx1*sy1)

    return sx1, sy1, rhoxy
###

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

def build_pol_maps(Qimg, Uimg, minimum_SNR=1.0, p_estimator='naive', pa_estimator='naive'):
    '''This function will build polarization percentage and position angle maps
    from the input Qimg and Uimg AstroImage instances. If the DEL_PA and
    S_DEL_PA header keywords are set (and match), then the position angle maps

    parameters:
    Qimg         -- an AstroImage instance of the stokes Q parameter
    Uimg         -- an AstroImage instance of the stokes U parameter
    minimum_SNR  -- a float which determins the minimum SNR value required
                    before rejecting the null hypothesis, "this source is
                    unpolarized"
    p_estimator  -- ['naive', 'wardle_kronberg', 'maier_piecewise',
                    'modified_asymptotic']
                    this specifies which estimator of the polarization will be
                    used. The default 'naive' estimator is a direct computation
                    of the polarization percentage while the other estimators
                    make SOME kind of an attempt to correct for the biasing
                    effects.
    pa_estimator -- ['naive', 'max_likelihood_1d', 'max_likelihood_2d']
    '''
    # Check if the U and Q images are the same shape
    if Qimg.arr.shape != Uimg.arr.shape:
        raise ValueError('The U and Q images must be the same shape')

    # P estimation
    ############################################################################
    # Estimate of the polarization percentage using the requested method.
    ############################################################################
    if p_estimator.upper() == 'NAIVE':
        #####
        # Compute a raw estimation of the polarization map
        #####
        Pmap  = np.sqrt(Qimg**2 + Uimg**2)

        # The sigma which determines the Rice distribution properties is the
        # width of the Stokes Parameter distribution, so we will simply compute
        # an average uncertainty and assign it to the Pmap AstroImage
        Pmap.sigma = 0.5*(Qimg.sigma + Uimg.sigma)

    elif p_estimator.upper() == 'WARDLE_KRONBERG':
        #####
        # Handle the ricean correction using Wardle and Kronberg (1974) method
        #####
        # Quickly build the P map
        Pmap = np.sqrt(Qimg**2 + Uimg**2)

        # Apply the bias correction
        # The sigma which determines the Rice distribution properties is the
        # width of the Stokes Parameter distribution, so we will simply compute
        # an average uncertainty and assign it to the Pmap AstroImage
        smap = 0.5*(Qimg.sigma + Uimg.sigma)

        # # This is the old correction we were using before I reread the Wardle
        # # and Kronberg paper... it is even MORE aggresive than the original
        # # recomendation
        # smap = Pmap.sigma

        # Catch any stupid values (nan, inf, etc...)
        badVals = np.logical_not(np.logical_and(
            np.isfinite(Pmap.arr),
            np.isfinite(smap)))

        # Replace the stupid values with zeros
        if np.sum(badVals) > 0:
            badInds           = np.where(badVals)
            Pmap.arr[badInds] = 0.0
            smap[badInds]     = 1.0

        # Check which measurements don't match the minimum SNR
        zeroVals = Pmap.arr/smap <= minimum_SNR
        numZero  = np.sum(zeroVals.astype(int))
        if numZero > 0:
            # Make sure the square-root does not produce NaNs
            zeroInds           = np.where(zeroVals)
            Pmap.arr[zeroInds] = 2*smap[zeroInds]

        # Compute the "debiased" polarization map
        tmpPmap =  Pmap.arr*np.sqrt(1.0 - (smap/Pmap.arr)**2)

        if numZero > 0:
            # Set all insignificant detections to zero
            tmpPmap[zeroInds] = 0.0

        # Now we can safely take the sqare root of the debiased values
        Pmap.arr   = tmpPmap
        Pmap.sigma = smap

    elif p_estimator.upper() == 'MAIER_PIECEWISE':
        # The following is taken from Maier et al. (2014) s = sigma_q == sigma_u
        # and the quantity of interest is the SNR value  p = P/s

        # Compute the raw polarization and uncertainty
        Pmap = np.sqrt(Qimg.arr**2 + Uimg.arr**2)
        smap = 0.5*(Qimg.sigma + Uimg.sigma)

        # Compute the SNR map (called "p" in most papers)
        pmap = Pmap/smap

        # Find those values which don't meet the minimum_SNR requirement
        zeroInds = np.where(Pmap/smap <= minimum_SNR)
        if len(zeroInds[0]) > 0:
            # Set all insignificant detections to zero
            pmap[zeroInds] = 0.0


        # Make a copy of the pmap for modification
        p1map = pmap.copy()

        # Classify each pixel in the SNR map to be computed using the least
        # biased estimator, as described in Maier et al. (2014)
        classRanges = [0, np.sqrt(2), 1.70, 2.23, 2.83, np.inf]
        for iClass in range(len(classRanges) - 1):
            # Define the estimator for this class
            if iClass == 0:
                def p1(p):
                    return 0

            elif iClass == 1:
                def p1(p):
                    pshift = p - np.sqrt(2)
                    return pshift**0.4542 + pshift**0.4537 + pshift/4.0

            elif iClass == 2:
                def p1(p):
                    return 22*(p**(0.11)) - 22.076

            elif iClass == 3:
                def p1(p):
                    return 1.8*(p**(0.76)) - 1.328

            elif iClass == 4:
                def p1(p):
                    return (p**2 - 1.0)**(0.5)

            # Grab the limiting SNRs for this case
            SNRmin, SNRmax = classRanges[iClass], classRanges[iClass + 1]

            # Locate the pixels where
            classPix = np.logical_and(pmap >= SNRmin, pmap < SNRmax)

            # If some of the pixels fall into this class, then recompute the
            # corrected pmap1 value
            if np.sum(classPix) > 0:
                classInds        = np.where(classPix)
                p1map[classInds] = p1(pmap[classInds])

        # Now that each class of SNR values has been evaluated, scale up the
        # estimated SNR values by their own respective sigma value
        Pmap       = np.sqrt(Qimg**2 + Uimg**2)
        Pmap.arr   = p1map*smap
        Pmap.sigma = smap

    elif p_estimator.upper() == 'MODIFIED_ASYMPTOTIC':
        # The following is taken from Plaszczynski et al. (2015). This makes the
        # additional assumption/simplicication that (s = sigma_q == sigma_u).

        # Compute the raw polarization and uncertainty
        Pmap = np.sqrt(Qimg.arr**2 + Uimg.arr**2)
        smap = 0.5*(Qimg.sigma + Uimg.sigma)

        # Apply the asymptotic debiasing effect.
        P1map = Pmap - (smap**2 * ((1 - np.exp(-(Pmap/smap)**2)/(2*Pmap))))

        # Locate any
        zeroInds = np.where(Pmap/smap <= minimum_SNR)
        if len(zeroInds[0]) > 0:
            # Set all insignificant detections to zero
            P1map[zeroInds] = 0.0

        # Now compute an AstroImage object and store the corrected P1map
        Pmap       = np.sqrt(Qimg**2 + Uimg**2)
        Pmap.arr   = P1map
        Pmap.sigma = smap

    else:
        raise ValueError("Did not recognize 'p_estimator' keyword value")

    ############################################################################
    # Parse the header information for building the PA map
    ############################################################################
    # Check for a DELTAPA keyword in the headers
    Qhas_DPA = ('DELTAPA' in Qimg.header.keys())
    Uhas_DPA = ('DELTAPA' in Uimg.header.keys())

    # Retrieve the DELpa values
    if Qhas_DPA and Uhas_DPA:
        QDPA = Qimg.header['DELTAPA']
        UDPA = Uimg.header['DELTAPA']
        if QDPA == UDPA:
            deltaPA = QDPA
        else:
            raise ValueError('The DELTAPA values in U and Q do not match.')
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
            raise ValueError('The S_DPA values in U and Q do not match.')
    else:
        s_DPA = 0.0

    # Grab the rotation of these images with respect to celestial north
    Qrot = Qimg.get_rotation()
    Urot = Uimg.get_rotation()

    # Check if both Q and U have astrometry
    if Qrot == Urot:
        # Add rotation angle to final deltaPA
        deltaPA += Qrot
    else:
        raise ValueError('The astrometry in U and Q do not seem to match.')

    # PA estimation
    ############################################################################
    # Estimate of the polarization position angle using the requested method.
    ############################################################################
    if pa_estimator.upper() == 'NAIVE':
        # Build the PA map and add the uncertaies in quadrature
        PAmap = (np.rad2deg(0.5*np.arctan2(Uimg, Qimg)) + deltaPA + 720.0) % 180.0
        if s_DPA > 0.0:
            PAmap.sigma = np.sqrt(PAmap.sigma**2 + s_DPA**2)
    elif pa_estimator.upper() == 'MAX_LIKELIHOOD_1D':
        raise NotImplementedError()
    elif pa_estimater.upper() == 'MAX_LIKELIHOOD_2D':
        raise NotImplementedError()
    else:
        raise ValueError("Did not recognize 'pa_estimator' keyword value")

    # Now that everything has been computed, simply return the Pmap, PAmap tuple
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
