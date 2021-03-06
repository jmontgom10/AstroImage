# This tells Python 2.x to handle imports, division, printing, and unicode the
# way that `future` versions (i.e. Python 3.x) handles those things.
from __future__ import absolute_import, division, print_function, unicode_literals

# Core library imports
import os
import sys
import copy
import subprocess
import warnings
from functools import lru_cache

# Scipy imports
import numpy as np
from scipy import ndimage
from scipy import optimize

# Astropy imports
from astropy.nddata import NDDataArray, StdDevUncertainty
from astropy.modeling import models, fitting
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales, proj_plane_pixel_area
from astropy import units as u
from astropy.stats import sigma_clip, sigma_clipped_stats
from photutils import (DAOStarFinder, data_properties,
    CircularAperture, CircularAnnulus, aperture_photometry)

# AstroImage imports
from ..reduced import ReducedScience

# Define which functions, classes, objects, etc... will be imported via the command
# >>> from .astroimage import *
__all__ = ['PhotometryAnalyzer']

# Define the analyzer class
class PhotometryAnalyzer(object):
    """
    Performs curve-of-growth and PSF analysis along with aperture photometry.
    """

    def __init__(self, image):
        """
        Constructs a PhotometryAnalyzer instance for operating on an image.

        Params
        ------
        image : astroimage.ReducedScience (or subclass)
            The astroimage object on which to perform the photometry.
        """
        if not isinstance(image, ReducedScience):
            raise TypeError('`image` must be a ReducedScience instance')

        # Store the image for later use
        self.image = image

    ##################################
    ### START OF ANALYZERS        ###
    ##################################

    @lru_cache()
    def get_psf(self, satLimit=16e3):
        """
        Computes the average PSF properties from the brightest stars in the
        image.

        Parameters
        ----------
        satLimit : int or float, optional, default: 16e3
            Sources which contain any pixels with more than this number of
            counts will be discarded from the returned list of sources on
            account of being saturated.

        Returns
        -------
        medianPSF : numpy.ndarray
            A small postage stamp of the PSF computed from the median star profiles.

        PSFparams : dict
            The properties of the Gaussian which best fits the array in
            PSFstamp. See astropy.modeling.functional_models.Gaussian2D for more
            information about how these values are defined.

            The keys of the dictionary are
            'smajor': float
                semimajor axis width in pixels
            'sminor': float
                semiminor axis width in pixels
            'theta':
                Rotation of the major axis in CCW degreesfrom horizontal axis
        """
        # Set the standard stellar cutout size
        cutoutSize = 21

        # Grab the star positions
        xStars, yStars = self.image.get_sources(
            satLimit = satLimit,
            crowdLimit = np.sqrt(2)*cutoutSize ,
            edgeLimit = cutoutSize + 1
        )

        # Catch the case where no stars were located
        if ((xStars.size == 1 and xStars[0] is None) or
            (yStars.size == 1 and yStars[0] is None)):
            warnings.warn('There are no well behaving stars')
            outStamp = np.zeros((cutoutSize, cutoutSize))
            outDict  = {'smajor':None, 'sminor':None, 'theta':None}
            return outStamp, outDict

        # Count the number of stars and limit the list to either 50 stars or
        # the brightest 25% of the stars
        numberOfStars = xStars.size
        if numberOfStars > 50:
            xStars, yStars = xStars[0:50], yStars[0:50]

        # Grab the list of star cutouts
        starCutouts = self.extract_star_cutouts(xStars, yStars, cutoutSize=cutoutSize)

        # Loop through each cutout and grab its data properties
        sxList = []
        syList = []

        for starCutout in starCutouts:
            # Retrieve the properties of the star in this patch
            props = data_properties(starCutout)

            # Store the gaussian component eigen-values
            sxList.append(props.semimajor_axis_sigma.value)
            syList.append(props.semiminor_axis_sigma.value)

        # Find potential outliers and mask them
        sxArr    = sigma_clip(sxList)
        syArr    = sigma_clip(syList)

        # Find out which stars have good values in BOTH sx and sy
        badSXSY  = np.logical_or(sxArr.mask, syArr.mask)
        goodSXSY = np.logical_not(badSXSY)

        # Cut out any patches with bad sx or bad sy values
        if np.sum(goodSXSY) == 0:
            warnings.warn('There are no well behaving stars')
            outStamp = np.zeros((cutoutSize, cutoutSize))
            outDict  = {'smajor':None, 'sminor':None, 'theta':None}
            return outStamp, outDict
            # raise IndexError('There are no well behaving stars')

        # If some of the patches are bad, then cut those out of the patch list
        if np.sum(badSXSY) > 0:
            goodInds    = (np.where(goodSXSY))[0]
            starCutouts = starCutouts[goodInds, :, :]

        # Compute an "median patch"
        starCutoutArray = np.array(starCutouts)
        medianPSF       = np.median(starCutoutArray, axis=0)

        # Build a gaussian + 2Dpolynomial (1st degree) model to fit median patch
        # Build a gaussian model for fitting stars
        gauss_init = models.Gaussian2D(
            amplitude=1000.0,
            x_mean=10.0,
            y_mean=10.0,
            x_stddev=3.0,
            y_stddev=3.0,
            theta=0.0
        )
        # Build a 2Dpolynomial (1st degree) model to fit the background level
        bkg_init = models.Polynomial2D(1)
        PSF_init = gauss_init + bkg_init
        fitter   = fitting.LevMarLSQFitter()

        # Generate arrays for the x and y pixel positions
        yy, xx = np.mgrid[0:cutoutSize, 0:cutoutSize]

        # Finllay, re-fit a gaussian to this median patch
        # Ignore model warning from the fitter
        with warnings.catch_warnings():
            # Fit the model to the patch
            warnings.simplefilter('ignore')
            PSF_model = fitter(PSF_init, xx, yy, medianPSF)

        # Modulate the fitted theta value into a reasonable range
        goodTheta         = (PSF_model.theta_0.value % (2*np.pi))
        PSF_model.theta_0 = goodTheta

        # Build a 2D polynomial background to subtract
        bkg_model = models.Polynomial2D(1)

        # Transfer the background portion of the PSF model to the
        # polynomial plane model.
        bkg_model.c0_0 = PSF_model.c0_0_1
        bkg_model.c1_0 = PSF_model.c1_0_1
        bkg_model.c0_1 = PSF_model.c0_1_1

        # Subtract the planar background and renormalize the median PSF
        medianPSF -= bkg_model(xx, yy)
        medianPSF /= medianPSF.sum()

        # Return the fitted PSF values
        smajor, sminor, theta = (
            PSF_model.x_stddev_0.value,
            PSF_model.y_stddev_0.value,
            PSF_model.theta_0.value
        )

        # Define return values and return them to the user
        PSFparams = {'smajor':smajor, 'sminor':sminor, 'theta':theta}

        return (medianPSF, PSFparams)

    def aperture_photometry(self, xStars, yStars, starApr, skyAprIn, skyAprOut):
        """
        Computes the aperture photometry for the specified locations

        Paramaters
        ----------
        xStars : array_like (length - numStars)
            An array of star locations (in pixels) along the x-axis

        yStars : array_like (length - numStars)
            An array of star locations (in pixels) along the y-axis

        starApr : scalar or array_like (length - numApr)
            The size of the circular aperture (in pixels) within which to sum
            up the star counts.

        skyAprIn : int or float
            The inner radius (in pixels) of the circular annulus within which to
            sum up the sky counts.

        skyAprOut : int or float
            The outer radius (in pixels) of the cirucal annulus within which to
            sum up the sky counts.

        Returns
        -------
        instrumentalMagnitudes : numpy.ndarray (shape - (numStars, numApr))
            Instrumental magnitudes computed using the supplied parameters
        """
        multipleApr = hasattr(starApr, '__iter__')
        if multipleApr:
            # Construct the stellar apertures
            starApertures = [CircularAperture((xStars, yStars), r=r) for r in starApr]
        else:
            # Treat the starApr variable as a scalar
            try:
                starApr = float(starApr)
                starApertures = CircularAperture((xStars, yStars), r=starApr)
            except:
                raise

        # Compute the raw stellar photometry
        starRawPhotTable = aperture_photometry(
            self.image.data,
            starApertures,
            error=self.image.uncertainty,
            pixelwise_error=True
        )

        # Construct the sky apertures
        skyApertures = CircularAnnulus((xStars, yStars),
            r_in=skyAprIn, r_out=skyAprOut)

        # Compute the raw sky photometry
        skyRawPhotTable = aperture_photometry(
            self.image.data,
            skyApertures,
            error=self.image.uncertainty,
            pixelwise_error=True
        )

        # Compute the mean packgroud value at each star
        bkg_mean = skyRawPhotTable['aperture_sum'] / skyApertures.area()

        # Subtract the average sky background and store the resultself
        if multipleApr:
            bkg_sum = [bkg_mean * sa.area() for sa in starApertures]
            subtractedStarPhot = np.array([
                starRawPhotTable['aperture_sum_{}'.format(i)] - bkg_sum[i]
                for i in range(len(starApr))])

            # Compute the uncertainty in the background subtracted photometry.
            subtractedPhotUncert = np.array([
                np.sqrt(
                    starRawPhotTable['aperture_sum_err_{}'.format(i)]**2 +
                    skyRawPhotTable['aperture_sum_err']**2
                ) for i in range(len(starApr))
            ])
        else:
            bkg_sum = bkg_mean * starApertures.area()
            subtractedStarPhot = starRawPhotTable['aperture_sum'] - bkg_sum
            subtractedPhotUncert = np.sqrt(
                starRawPhotTable['aperture_sum_err']**2 +
                skyRawPhotTable['aperture_sum_err']**2
            )

        return subtractedStarPhot, subtractedPhotUncert

    @lru_cache()
    def get_curve_of_growth(self, satLimit=16e3):
        """
        Computes the parameters for a King profile curve of growth.

        See Stetson (PASP 102: 932-948, August 1990) for more information.

        The King profile is given by
        S(r; Ri,A,B,C,D) = (B*M(r; A) + (1-B)*(C*G(r; Ri) + (1-c)*H(r; D*Ri)))

        where M, G, and H are Moffat, Gaussian, and exponential functions,
        respectively.

        M(r; A)   = (A-1)/pi * (1 + r**2)**(-A)
        G(r; Ri)  = 1/(2*pi*(Ri**2)) * exp(-0.5*(r/Ri)**2)
        H(r; D*Ri) = 1/(2*pi*(D*Ri)**2) * exp(-r/(D*Ri))

        Ri = Radial seeing parameter
        A  = Moffat exponential parameter
        B  = Sets the fraction of contribution from the Gaussian function
        C  = Sets the fraction of contribution from the exponential function
        D  = The ratio of guassian and exponential widths (should be ~0.9)

        Returns
        -------
        parameterDict : dict
            A dictionary containing the parameters for the best fit King profile
            based on the bright stars in the image.
        """
        # Grab the star positions using the same default values as get_psf
        xStars, yStars = self.image.get_sources(
            satLimit = satLimit,
            crowdLimit = np.sqrt(2)*21 ,
            edgeLimit = 22
        )

        # # Count the number of stars and limit the list to either 50 stars or
        # # the brightest 25% of the stars
        # numberOfStars = xStars.size
        # if numberOfStars > 50:
        #     xStars, yStars = xStars[0:50], yStars[0:50]

        # # Define the mofat function in terms of Steton's formulation
        # # alpha = (A*np.pi/gamma) + 1
        # alpha2amplitude = lambda model: (model.alpha - 1)/np.pi
        # M = models.Moffat1D(
        #     gamma=1.0,
        #     tied={'amplitude':alpha2amplitude}
        #     fixed={'gamma':True}
        #     bounds={'amplitude':[1,1e6]}
        # )
        #
        # # Define the gaussian function
        # stddev2amplitude = lambda model: 1.0/(2*np.pi*model.stddev**2)
        # G = models.Gaussian1D(
        #     mean=0.0,
        #     tied={'amplitude':stddev2amplitude}
        #     fixed={'mean':True}
        # )
        #
        # # Define the exponential model
        # def exp_model(r, Dratio=0.9, Ri_width=1.0):
        #     return np.exp(-r/(Dratio*Ri_width))/(2*np.pi*Dratio*Ri_width)
        # def exp_deriv(x, Dratio=0.9, Ri_width=1.0):
        #     return np.exp(-r/(Dratio*Ri_width))/(2*np.pi*(Dratio*Ri_width)**2)
        #
        # expModel = models.custom_model(exp_model, fit_deriv=exp_deriv)

        # Measure the photometry at 15 different apertures
        for apr in range(3, 18):
            # Call the "do_photometry" method
            instrumentalMagnitudes = self.aperture_photometry()

        # Build the compound King profile
        def king_model(r, Ri, A, B, C, D):
            # Generate the terms of the King profile
            M = (A-1)/pi * (1 + r**2)**(-A)
            G = 1/(2*pi*(Ri**2)) * exp(-0.5*(r/Ri)**2)
            H = 1/(2*pi*(D*Ri)**2) * exp(-r/(D*Ri))

            # Compute the king profile value
            S = B*M + (1-B)*(C*G + (1-C)*H)

            return S

        # def king_deriv(r, Ri=1.0, A=0.5, B=0.5, C=0.5, D=0.9):


        #        ( Ri=1.0,   A=1.5,    B=0.5,    C=0.5,    D=0.9  )
        p_init = ( 1.0,      1.5,      0.5,      0.5,      0.9    )
        bounds = ((0, 100), (1, 100), (0, 1e6), (0, 1e6), (0, 1e6))
        p_opt  = optimize.curve_fit(
            king_model,
            xdata,
            ydata,
            p0=p_init,
            sigma=y_uncertainty
        )


    ##################################
    ### END OF ANALYZERS           ###
    ##################################
