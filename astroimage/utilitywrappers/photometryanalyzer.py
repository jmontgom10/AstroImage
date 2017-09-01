# This tells Python 2.x to handle imports, division, printing, and unicode the
# way that `future` versions (i.e. Python 3.x) handles those things.
from __future__ import absolute_import, division, print_function, unicode_literals

# Core library imports
import warnings
from functools import lru_cache

# Scipy imports
import numpy as np
from scipy import optimize

# Astropy imports
from astropy.modeling import models, fitting
from astropy.table import Table
from astropy.coordinates import SkyCoord
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
    ### START OF COG METHODS       ###
    ##################################

    @staticmethod
    def _king_profile(r, Ri, A, B, C, D):
        """
        Returns the King profile value for the provided parameters

        See Stetson (PASP 102: 932-948, August 1990) for more information.

        The King profile is given by

        S(r; Ri,A,B,C,D) = (B*M(r; A) + (1-B)*(C*G(r; Ri) + (1-C)*H(r; D*Ri)))

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
        kingProfile = B*M + (1-B)*(C*G + (1-C)*H)

        return kingProfile

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
        integratedKingProfile = (
            B*integratedM +
            (1-B)*(
                C*integratedG +
                (1-C)*integratedH
            )
        )

        return integratedKingProfile

    def _king_COG_values(self, aprRad, Ri, A, B, C, D):
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
        kingCOGvals : `numpy.ndarray` (length - (numApr - 1))
            The curve-of-growth values for the supplied apertures and
            King profile parameters.
        """
        # Convert aprRad to a numpy array and test for sorting
        r = np.array(aprRad)
        assert np.all(r.argsort() == np.arange(r.size))

        # Add on a *minimum* aperture to account for the fact that one aperture
        # is "gobbled up" by the differencing procedure
        r = np.insert(r, 0, self.minimumCOGapr)

        # Compute the integrated King profile for each aperture provided
        integratedKing = self.__class__._integrated_king_profile(r, Ri, A, B, C, D)

        # Compute the difference between subsequent radii (in Pogson magnitudes)
        kingCOGvalues = -2.5*np.log10(integratedKing[1:]/integratedKing[0:-1])

        return kingCOGvalues

    ##################################
    ### END OF STATIC METHODS      ###
    ##################################

    def __init__(self, image):
        """
        Constructs a PhotometryAnalyzer instance for operating on an image.

        Parameters
        ----------
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

    def get_COG_stars(self, fluxLimits=(5e2,16e3)):
        """
        Finds the location of stars good for building a curve-of-growth (COG)

        Parameters
        ----------
        fluxLimits : tuple, default: (1e3, 16e3)
            Only sources with maximum pixels values between these limits will be
            selected as candidate stars from which to build a curve-of-growth

        Returns
        -------
        xCOGstars, yCOGstars : array_like (length - numCOGstars)
            An array of locations (in pixels) along the x- and y-axes of
            bright stars appropriate for determining the curve-of-growth
        """
        # Parse the flux limits
        try:
            fluxLimts = tuple(fluxLimits)
        except:
            raise

        if len(fluxLimits) != 2:
            raise ValueError('`fluxLimit` must be an iterable of length 2')

        fluxMin, fluxMax = np.min(fluxLimits), np.max(fluxLimits)

        # Find all the stars in the image using the slightly modified settings
        xs, ys = self.image.get_sources(
            FWHMguess=3.0, minimumSNR=7.0, satLimit=np.max(fluxLimits),
            crowdLimit=21, edgeLimit=50
        )

        # Extract all those stars from the image
        starCutouts = self.image.extract_star_cutouts(
            xs, ys, cutoutSize=21
        )

        # Loop through sources and kill off those outside flux criteria
        keepInds   = []
        for iStar, starCutout in enumerate(starCutouts):
            # Grab the min and max fluxes and attempt test their values
            starMax = np.max(starCutout)
            if starMax > fluxMin and starMax < fluxMax:
                keepInds.append(iStar)

        # Test that at least *some* stars meet the flux limits
        if len(keepInds) == 0:
            raise RuntimeError('No stars were found between fluxLimits = ({}, {})'.format(*fluxLimits))

        # Convert the indices and fluxes to keep into a numpy array
        keepInds   = np.array(keepInds)

        # Only keep the starCutouts meeting the flux criteria
        xCOGstars = xs[keepInds]
        yCOGstars = ys[keepInds]

        return xCOGstars, yCOGstars

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
        # Test that xCOGstars and yCOGstars match
        if len(xCOGstars) != len(yCOGstars):
            raise ValueError('The size of `xCOGstars` and `yCOGstars` must match')

        # Test that enough stars have been provided to constrain the curve-of-growth
        # if len(xCOGstars) < 6:
        #     raise ValueError('Too few stars to constrain the curve-of-growth')

        # Estimate the FWHM and build the COG apertures from that
        _, psfParams = self.get_psf()
        psfFWHM      = np.sqrt(psfParams['sminor']*psfParams['smajor'])

        # Generate the appropriate apertures to use for COG building
        starApr   = np.linspace(0.5, 5.0, 20.0)*psfFWHM
        skyAprIn  = 5.5*psfFWHM
        skyAprOut = skyAprIn + 2.0*psfFWHM

        # Store the minimum starApr for use in the King profile fitting
        self.minimumCOGapr = np.min(starApr)

        # Call the "aperture_photometry" method
        starFlux, fluxUncert = self.aperture_photometry(
            xCOGstars, yCOGstars, starApr, skyAprIn, skyAprOut
        )

        # Construct the abscissa and ordinate values for the King COG
        xCOG      = 0.5*(starApr[1:] + starApr[0:-1])
        fluxRatio = starFlux[:,1:]/starFlux[:,0:-1]
        yCOG      = -2.5*np.log10(fluxRatio)

        # Apply a median filtered mean to the yCOG data
        yCOGmed = []
        yCOGstd = []
        for yData in yCOG.T:
            mean, median, stddev = sigma_clipped_stats(yData)
            yCOGmed.append(median)
            yCOGstd.append(stddev)
        yCOGmed = np.array(yCOGmed)
        yCOGstd = np.array(yCOGstd)

        # Store the COG data for later use
        self.starApr = starApr
        self.xCOG    = xCOG
        self.yCOG    = yCOG
        self.yCOGmed = yCOGmed
        self.yCOGstd = yCOGstd

        # Do any required error-propagation for this quantity
        if self.image.has_uncertainty:
            fluxRatioUncert = np.sqrt(
                (fluxUncert[:,1:]/starFlux[:,1:])**2 +
                (fluxUncert[:,0:-1]/starFlux[:,0:-1])**2
            )
            yUncert = 2.5*fluxRatioUncert/(np.log(10)*fluxRatio)
        else:
            yUncert = np.array([1.0 for i in range(fluxRatio.size)])

        # Initalize some parameter values for the King profile COG
        #        ( Ri=1.0,      A=1.5,    B=0.5,    C=0.5,    D=0.9  )
        p_init = ( 0.5*psfFWHM, 1.5,      0.5,      0.5,      0.9    )
        # bounds = ((0, 100), (1.0+1.e-6, 2.0), (0.0, 1.0), (0.0, 1.0), (0.0, np.inf))
        lowBounds = (0, 1.0+1e-6, 0, 0, 0)
        hiBounds  = (100, 2, 1, 1, np.inf)
        bounds    = (lowBounds, hiBounds)
        p_opt, p_cov = optimize.curve_fit(
            self._king_COG_values,
            starApr[1:],
            yCOGmed,
            p0=p_init,
            bounds=bounds,
            sigma=yCOGstd
        )

        # Construct the parameter dictionary. Store it and return it
        kingParams = dict(zip(['Ri', 'A', 'B', 'C', 'D'], p_opt))
        self.kingParams = kingParams

        return kingParams

    def show_curve_of_growth(self):
        """Displays the curve-of-growth as a sanity check for the user."""
        # Check if the curve-of-growth has already been determined
        if not hasattr(self, 'kingParams'):
            raise RuntimeError(
                'The curve-of-growth has not yet been determined for this object'
            )

        # Import plotting functionality
        import matplotlib.pyplot as plt

        # Generate plot
        plt.ion()
        plt.figure()

        # Show the curve-of-growth data
        for yData in self.yCOG:
            plt.plot(self.xCOG, yData, marker='.', color='k')

        # Show the median vaules
        plt.errorbar(
            self.xCOG, self.yCOGmed, yerr=self.yCOGstd,
            color='red', linewidth=5.0
        )

        # Show the best fit to the median values
        plt.plot(
            self.xCOG,
            self._king_COG_values(
                self.starApr[1:], **self.kingParams
            ),
            color='blue', linewidth=3.0
        )

        # Label the axes
        plt.xlabel('Aperture Radius [pix]')
        plt.ylabel('Curve-Of-Growth value [Delta Mags]')

        # Turn off interative plotting
        plt.ioff()

    def aperture_photometry(self, xStars, yStars, starApr, skyAprIn, skyAprOut, mask=None):
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

        mask : array_like (bool), optional
            An optional mask provided by the user to specify which pixels
            contain bad data and should be ignored when doing photometry. Pixels
            with a True value in the mask will be ignored during photometry.

        Returns
        -------
        starFlux : numpy.ndarray (shape - (numStars, numApr))
            Instrumental flux computed using the supplied parameters. The
            array contains one entry per star, per aperture.

        fluxUncert : numpy.ndarray (shape - (numStars, numApr))
            Uncertainty in the computed instrumental fluxes
        """
        # Test if the supplied `starApr` is an array, then use an array of
        # apertures. If the supplied `starApr` is a scalar, then use the same
        # aperture for all the stars.
        multipleApr = hasattr(starApr, '__iter__') and hasattr(starApr, '__len__')
        if multipleApr:
            # Double check that this is *really* a multiple aperture case
            if len(starApr) == 1:
                # If it's just one aperture stored in an iterable object...
                multipleApr   = False
                starApertures = CircularAperture((xStars, yStars), r=starApr)
            else:
                # Construct the stellar apertures
                starApertures = [CircularAperture((xStars, yStars), r=r) for r in starApr]
        else:
            # Treat the starApr variable as a scalar
            try:
                starApr       = float(starApr)
                starApertures = CircularAperture((xStars, yStars), r=starApr)
            except:
                raise

        # Compute the raw stellar photometry
        starRawPhotTable = aperture_photometry(
            self.image.data,
            starApertures,
            error=self.image.uncertainty,
            pixelwise_error=True,
            mask=mask
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
            # Loop through each aperture and compute the background subtracted value
            subtractedStarPhot   = []
            subtractedPhotUncert = []
            for iApr, sa in enumerate(starApertures):
                # Compute the *total* sky contribution to each star for this aperture
                thisBkg = bkg_mean.data * sa.area()

                # Compute the sky-subtracted stellar flux
                subtractedStarPhot.append(np.array(
                    starRawPhotTable['aperture_sum_{}'.format(iApr)] - thisBkg
                ))

                # If the image included an uncertainty array, then do error-prop
                if self.image.has_uncertainty:
                    subtractedPhotUncert.append(np.array(np.sqrt(
                        starRawPhotTable['aperture_sum_err_{}'.format(iApr)]**2 +
                        (skyRawPhotTable['aperture_sum_err'] * (sa.area() / skyApertures.area()))**2
                    )))

        else:
            bkg_sum = bkg_mean * starApertures.area()
            subtractedStarPhot = starRawPhotTable['aperture_sum'] - bkg_sum

            # If the image included an uncertainty array, then do error-prop
            if self.image.has_uncertainty:
                subtractedPhotUncert = np.sqrt(
                    starRawPhotTable['aperture_sum_err']**2 +
                    skyRawPhotTable['aperture_sum_err']**2
                )

        # Re-convert list of photometry to a numpy array
        subtractedStarPhot   = np.array(subtractedStarPhot).T

        if self.image.has_uncertainty:
            # Convert uncertainties back into an array, too, if they exist
            subtractedPhotUncert = np.array(subtractedPhotUncert).T

            return subtractedStarPhot, subtractedPhotUncert

        else:
            return subtractedStarPhot, None

    def get_maximum_SNR_apertures(self, xStars, yStars):
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
        starApr : `numpy.ndarray` (length - numStars)
            The aperture radius (in pixels) at which the apeture flux
            reaches a maximum signal-to-noise ratio (SNR).
        """
        # TODO: Make this actually work! For now, just return 3.2 for all stars
        return np.ones((len(xStars),))*3.2

    def compute_aperture_corrections(self, starApr, kingParams):
        """
        Corrects aperture photometry using the supplied King Profile

        Estimates the fraction of light contained *outside* the apertures
        used  to compute the supplied magnitudes.

        Parameters
        ----------
        starApr : array_like (length - numStars)
            The aperture radius (in pixels) at which the apeture flux is a
            maximum signal-to-noise ratio (SNR).

        kingParams : dict
            A dictionary containing the King Profile parameters which best
            fit the observations

        Returns
        -------
        appertureCorrections : array_like (length - numStars)
            The number of magnitudes to be added to each star provided its
            aperture photometry was measured using the corresponding starApr
        """
        # Disect the kingParams
        Ri = kingParams['Ri']
        A  = kingParams['A']
        B  = kingParams['B']
        C  = kingParams['C']
        D  = kingParams['D']

        # Generate each term of the integrated King profile
        integratedM = (1+starApr**2)**(1-A)
        integratedG = np.exp(-0.5*(starApr/Ri)**2)
        integratedH = ((starApr + D*Ri)*np.exp(-starApr/(D*Ri)))/(D*Ri)

        # Generate the aperture correction
        fractionOfFluxOutsideOfApertures =  1.0 - (
            B*integratedM +
            (1-B)*(
                C*integratedG +
                (1-C)*integratedH
            )
        )
        apertureCorrections = -2.5*np.log10(fractionOfFluxOutsideOfApertures)

        return apertureCorrections
    ##################################
    ### END OF ANALYZERS           ###
    ##################################
