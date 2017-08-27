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

    ##################################
    ### START OF STATIC METHODS    ###
    ##################################

    @staticmethod
    def _king_profile(r, Ri, A, B, C, D):
        """
        Returns the King profile value for the provided parameters

        See Stetson (PASP 102: 932-948, August 1990) for more information.

        The King profile is given by

        S(r; Ri,A,B,C,D) = (B*M(r; A) + (1-B)*(C*G(r; Ri) + (1-c)*H(r; D*Ri)))

        where M, G, and H are Moffat, Gaussian, and exponential functions,
        respectively.

        M(r; A)   = (A-1)/pi * (1 + r**2)**(-A)
        G(r; Ri)  = 1/(2*pi*(Ri**2)) * exp(-0.5*(r/Ri)**2)
        H(r; D*Ri) = 1/(2*pi*(D*Ri)**2) * exp(-r/(D*Ri))

        Parameters
        ----------
        r : float array_like or scalar
            The radius (in pixels) at which to evaluate the King profile

        Ri : float
            The radial seeing parameter

        A : float
            The Moffat exponential parameter

        B : float
            The fraction of contribution from the Gaussian function

        C : float
            The fraction of contribution from the exponential function

        D : float
            The ratio of guassian and exponential widths (should be ~0.9)

        Returns
        -------
        kingProfile : array_like or scalar
            The value of the King profile at the specified radius using
            the supplied parameters
        """
        # Generate the terms of the King profile
        M = (A-1)/np.pi * (1 + r**2)**(-A)
        G = 1/(2*np.pi*(Ri**2)) * np.exp(-0.5*(r/Ri)**2)
        H = 1/(2*np.pi*(D*Ri)**2) * np.exp(-r/(D*Ri))

        # Compute the king profile value
        S = B*M + (1-B)*(C*G + (1-C)*H)

        return S

    @staticmethod
    def _integrated_king_profile(r, Ri, A, B, C, D):
        """
        The King profile integrated out to the specified radius

        For parameter description see `_king_profile`

        Returns
        -------
        integratedKingProfile : array_like or scalar
            The value of the King profile integrated from a radius of
            zero out to the specified radius
        """
        # Generate each term of the integrated King profile
        integratedM = 1.0 - (1+r**2)**(1-A)
        integratedG = 1.0 - np.exp(-0.5*(r/Ri)**2)
        integratedH = 1.0 - ((r + D*Ri)*np.exp(-r/(D*Ri)))/(D*Ri)

        # Compute the total integrated value
        integratedS = (
            B*integratedM +
            (1-B)*(
                C*integratedG +
                (1-C)*integratedH
            )
        )

        return integratedS

    @staticmethod
    def _king_COG_value(aprRad, Ri, A, B, C, D):
        """
        Computes the magnitude differences between subsequent apertures

        Parameters
        ----------
        aprRad : array_like
            The apertures at which magnitudes were measured, and from
            which to compute a curve-of-growth.

        For the King profile paramater values, see `_king_profile`

        Returns
        -------
        kingCOGvals : `numpy.array` (length - (numApr - 1))
            The curve-of-growth values for the supplied apertures and
            King profile parameters.
        """
        # Convert aprRad to a numpy array and test for sorting
        r = np.arry(aprRad)
        assert np.all(r.argsort() == np.arange(r.size))

        # Compute the integrated King profile for each aperture provided
        integratedKing = self._integrated_king_profile(r, Ri, A, B, C, D)

        # Compute the difference between subsequent radii
        kingCOGvalues = integratedKing[1:] - integratedKing[0:-1]

        return kingCOGvalues

    ##################################
    ### END OF STATIC METHODS      ###
    ##################################

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
        starCutouts = self.image.extract_star_cutouts(xStars, yStars, cutoutSize=cutoutSize)

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

        # Convert the starCutouts into an array for easy indexing
        starCutoutArray = np.array(starCutouts)

        # If some of the patches are bad, then cut those out of the patch list
        if np.sum(badSXSY) > 0:
            goodInds        = (np.where(goodSXSY))[0]
            starCutoutArray = starCutoutArray[goodInds]

        # Compute an "median patch"
        medianPSF = np.median(starCutoutArray, axis=0)

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

    def locate_COG_stars(self, satLimit=16e3):
        """
        Finds the location of non-saturated stars good for building COG

        Parameters
        ----------
        satLimit : int or float, optional, default: 16e3
            Sources which contain any pixels with more than this number of
            counts will be discarded from the returned list of sources on
            account of being saturated.

        Returns
        -------
        xCOGstars, yCOGstars : array_like (length - numCOGstars)
            An array of locations (in pixels) along the x- and y-axes of
            bright stars appropriate for determining the curve-of-growth
        """
        pass

    @lru_cache()
    def get_curve_of_growth(self, xCOGstars, yCOGstars):
        """
        Computes the parameters for a King profile curve of growth.


        Parameters
        ----------
        xCOGstars, yCOGstars : array_like (length - numCOGstars)
            An array of locations (in pixels) along the x- and y-axes of
            bright stars appropriate for determining the curve-of-growth

        Returns
        -------
        kingParams : dict
            A dictionary containing the King Profile parameters which best
            fit the observations
        """
        # Grab the star positions using the same default values as get_psf
        xStars, yStars = self.image.get_sources(
            satLimit = satLimit,
            crowdLimit = np.sqrt(2)*21 ,
            edgeLimit = 22
        )

        import pdb; pdb.set_trace()

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

        # def king_deriv(r, Ri=1.0, A=0.5, B=0.5, C=0.5, D=0.9):
        #        ( Ri=1.0,   A=1.5,    B=0.5,    C=0.5,    D=0.9  )
        p_init = ( 1.0,      1.5,      0.5,      0.5,      0.9    )
        bounds = ((0, 100), (1, 100), (0, 1e6), (0, 1e6), (0, 1e6))
        p_opt  = optimize.curve_fit(
            self._king_model,
            xdata,
            ydata,
            p0=p_init,
            sigma=y_uncertainty
        )

        # construct the parameter dictionary.
        kingParams = dict(zip(['Ri', 'A', 'B', 'C', 'D'], p_opt))
        import pdb; pdb.set_trace9)

        return kingParams

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
        instrumentalFlux : numpy.ndarray (shape - (numStars, numApr))
            Instrumental flux computed using the supplied parameters. The
            array contains one entry per star, per aperture.

        sigmaFlux : numpy.ndarray (shape - (numStars, numApr))
            Uncertainty in the computed instrumental fluxes
        """
        # Test if the supplied `starApr` is an array, then use an array of
        # apertures. If the supplied `starApr` is a scalar, then use the same
        # aperture for all the stars.
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

        # TODO: Check if uncertainty attribute exist, if not, then require gain

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

    def locate_maximum_SNR_apertures(self, xStars, yStars):
        """
        Finds the aperture at which each stellar flux has a maximum SNR

        Paramaters
        ----------
        xStars : array_like (length - numStars)
            An array of star locations (in pixels) along the x-axis

        yStars : array_like (length - numStars)
            An array of star locations (in pixels) along the y-axis

        Returns
        -------
        starApr : `numpy.array` (length - numStars)
            The aperture radius (in pixels) at which the apeture flux
            reaches a maximum signal-to-noise ratio (SNR).
        """
        pass

    def apply_aperture_corrections(self, starApr, instrumentalMagnitudes, kingParams):
        """
        Corrects aperture photometry using the supplied King Profile

        Estimates the fraction of light contained *outside* the apertures
        used  to compute the supplied magnitudes.

        Parameters
        ----------
        starApr : array_like ( length - numStars)
            The aperture radius (in pixels) at which the apeture flux is a
            maximum signal-to-noise ratio (SNR).

        instrumentalMagnitudes : array_like (shape - (numStars, numApr))
            Instrumental magnitudes computed using the supplied parameters

        kingParams : dict
            A dictionary containing the King Profile parameters which best
            fit the observations
        """
        pass

    ##################################
    ### END OF ANALYZERS           ###
    ##################################
