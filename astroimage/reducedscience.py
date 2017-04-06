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

# Astropy imports
from astropy.nddata import NDDataArray, StdDevUncertainty
from astropy.modeling import models, fitting
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales, proj_plane_pixel_area
from astropy import units as u
from astropy.stats import sigma_clip, sigma_clipped_stats
from photutils import DAOStarFinder, data_properties

# Matplotlib imports
import matplotlib as mpl
import matplotlib.colors as mcol
import matplotlib.pyplot as plt

# AstroImage imports
from .baseimage import ClassProperty
from .reducedimages import ReducedImage
from .imagenumericsmixin import ImageNumericsMixin

# Define which functions, classes, objects, etc... will be imported via the command
# >>> from .astroimage import *
__all__ = ['ReducedScience']

class ReducedScience(ImageNumericsMixin, ReducedImage):
    """
    A class for handling fully reduced science frames.

    Provides methods for mathematical operations, solving astrometry (for
    systems with Astrometry.net engine installed), visually displaying the image
    with its solved astrometry, applying airmass correction, finding sources,
    and performing photometry.
    """

    ##################################
    ### START OF CLASS VARIABLES   ###
    ##################################

    # Extend the list of acceptable properties for this class
    __properties = copy.deepcopy(ReducedImage.properties)
    __properties.extend([
        'bscale',
        'wcs'
    ])

    ##################################
    ### END OF CLASS VARIABLES     ###
    ##################################

    ##################################
    ### END OF CLASS METHODS       ###
    ##################################

    @ClassProperty
    @classmethod
    def properties(cls):
        return cls.__properties

    ##################################
    ### END OF CLASS METHODS       ###
    ##################################

    def __init__(self, *args, **kwargs):
        # Invoke the parent class __init__ method
        super(ReducedScience, self).__init__(*args, **kwargs)

        # Test if there is any WCS present in this header
        # NOTE: IT WOULD SEEM THAT THE WCS(self.header) CALL IS WHAT IS
        # CAUSING THE 'FK5' ERROR
        #
        # WARNING: FITSFixedWarning: RADECSYS= 'FK5 '
        # the RADECSYS keyword is deprecated, use RADESYSa. [astropy.wcs.wcs]

        # Check if a BSCALE keyword was provided and store it
        headerKeys = self.header.keys()
        if 'BSCALE' in headerKeys:
            bscaleVal = self.header['BSCALE']
            if 'BUNIT' in headerKeys:
                bunitStr  = self.header['BUNIT'].strip()
                self.__bscale = bscaleVal*u.Unit(bunitStr)
            else:
                self.__bscale = bscaleVal

        # The default read-in properties yields an unscaled array
        self.__is_scaled = False

    ##################################
    ### START OF PROPERTIES        ###
    ##################################

    @property
    def bscale(self):
        """The scaling factor converting from ADU to physical units"""
        return self.__bscale

    @property
    def wcs(self):
        """`astropy.wcs.WCS` instance containing the astrometry of the image"""
        return self._BaseImage__fullData.wcs

    @property
    def has_wcs(self):
        """Boolean flag if the `wcs` property exists"""
        if self.wcs is not None:
            return self.wcs.has_celestial
        else:
            return False

    @property
    def pixel_scales(self):
        """Image plate scales along each axis un units of degrees/pixel"""
        if self.has_wcs:
            return proj_plane_pixel_scales(self.wcs) * (u.deg/u.pix)
        else:
            raise AttributeError('This `ReducedScience` does not have a wcs defined')

    @property
    def pixel_area(self):
        """Returns the area of the pixels in units of degrees^2/pixel"""
        if self.has_wcs:
            return proj_plane_pixel_area(self.wcs) * (u.deg**2/u.pix)
        else:
            raise AttributeError('This `ReducedScience` does not have a wcs defined')

    @property
    def rotation(self):
        """Returns the rotation of the image in degrees east of north"""
        # Check if it has a celestial coordinate system
        if self.has_wcs:
            if self.wcs.wcs.has_cd():
                # Grab the cd matrix
                cd = self.wcs.wcs.cd
            elif self.wcs.wcs.has_pc():
                # Convert the pc matrix into a cd matrix
                cd = self.wcs.wcs.cdelt*self.wcs.wcs.pc

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
            raise AttributeError('This `ReducedScience` does not have a wcs defined')

    @property
    def is_scaled(self):
        """Boolean flag of whether `arr` is in scaled units or ADU"""
        return self.__is_scaled

    ##################################
    ### END OF PROPERTIES          ###
    ##################################

    ##################################
    ### START OF GETTERS           ###
    ##################################
    @lru_cache()
    def get_sources(self, satLimit=16e3, crowdLimit=0, edgeLimit=0):
        """Implements the daofind algorithm to extract source positions.

        Parameters
        ----------
        satLimit : int or float, optional, default: 16e3
            Sources which contain any pixels with more than this number of
            counts will be discarded from the returned list of sources on
            account of being saturated.

        crowdLimit : int or float, optional, default: 0
            Sources with a neighest neighbor closer than this distance (in
            pixels) will be discarded from the returned list of sources on
            account of being too crowded.

        edgeLimit : int, or float, optional, default: 0
            Sources detected within this distance (in pixels) of the image edge
            will be discarded from the returned list of sources on account of
            being too close to the image edge.

        Returns
        -------
        x, y : numpy.ndarray
            A list of source positions within the image

        meimajor, semiminor : numpy.ndarray
            A list of semimajor and semiminor axis values
        """
        # Double check that edge-stars will be rejected before checking for
        # crowding...
        if edgeLimit < crowdLimit:
            edgeLimit = crowdLimit + 1

        # Ensure there are no problematic values
        tmpData = self.data.copy()
        badPix  = np.logical_not(np.isfinite(tmpData))
        if np.sum(badPix) > 0:
            badInds = np.where(badPix)
            tmpData[badInds] = np.nanmin(tmpData)

        # Grab the image sky statistics
        tmpImg = ReducedImage(tmpData)
        mean, median, std = tmpImg.sigma_clipped_stats()

        # Start by instantiating a DAOStarFinder object
        daofind = DAOStarFinder(fwhm=3.0, threshold=5.0*std)

        # Use that object to find the stars in the image
        sources = daofind(np.nan_to_num(tmpData) - median)

        # Grab the image shape for later use
        ny, nx = tmpData.shape

        # Cut out edge stars if requested
        if edgeLimit > 0:
            nonEdgeStars = sources['xcentroid'] > edgeLimit
            nonEdgeStars = np.logical_and(nonEdgeStars,
                sources['xcentroid'] < nx - edgeLimit - 1)
            nonEdgeStars = np.logical_and(nonEdgeStars,
                sources['ycentroid'] > edgeLimit)
            nonEdgeStars = np.logical_and(nonEdgeStars,
                sources['ycentroid'] < ny - edgeLimit - 1)

            # Cull the sources list to only include non-edge stars
            if np.sum(nonEdgeStars) > 0:
                nonEdgeInds = np.where(nonEdgeStars)
                sources     = sources[nonEdgeInds]
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

            # Grab the values within 15 pixels of this source, and test
            # if the source is saturated
            nearInds = np.where(dists < 15.0)
            notSaturated.append(tmpData[nearInds].max() < satLimit)

        # Cull the sources list to ONLY include non-saturated sources
        if np.sum(notSaturated) > 0:
            notSaturatedInds = np.where(notSaturated)
            sources          = sources[notSaturatedInds]
        else:
            raise IndexError('No sources passed the saturation test')

        # Perform the crowding test
        isolatedSource = []
        if crowdLimit > 0:
            # Generate pixel positions for the patch_data
            yy, xx = np.mgrid[0:np.int(crowdLimit), 0:np.int(crowdLimit)]

            # Loop through the sources and test if they're crowded
            for source in sources:
                # Extract the posiition for this source
                xs, ys = source['xcentroid'], source['ycentroid']

                # Compute the distance between other sources and this source
                dists = np.sqrt((sources['xcentroid'] - xs)**2 +
                                (sources['ycentroid'] - ys)**2)

                # Test if there are any OTHER stars within crowdLimit
                isolatedBool1 = np.sum(dists < crowdLimit) < 2

                # Do a double check to see if there are any EXTRA sources
                # Start by cutting out the patch surrounding this star
                # Establish the cutout bondaries
                btCut = np.int(np.round(ys - np.floor(0.5*crowdLimit)))
                tpCut = np.int(np.round(btCut + crowdLimit))
                lfCut = np.int(np.round(xs - np.floor(0.5*crowdLimit)))
                rtCut = np.int(np.round(lfCut + crowdLimit))

                # Cut out that data and subtract the floor.
                patch_data  = tmpData[btCut:tpCut,lfCut:rtCut]
                patch_data -= patch_data.min()

                # QUickly check the shape of this image
                props = data_properties(patch_data)
                sizeParam = np.sqrt(
                    props.semimajor_axis_sigma.value *
                    props.semiminor_axis_sigma.value
                )

                # TODO: Tuck each of these sanity checks into individual methods

                # Check if the source size is greater than 6.0 pixels
                reasonableSize = sizeParam < 6.0

                # TODO: think through a better algorithum for this whole section
                # Perhaps the first section is a sufficient test

                # # Null out data beyond the crowdLimit from the center
                # xs1, ys1 = xs - lfCut, ys - btCut
                # pixDist  = np.sqrt((xx - xs1)**2 + (yy - ys1)**2)
                # nullInds = np.where(pixDist > crowdLimit)
                #
                # # Null those pixels
                # patch_data[nullInds] = 0

                with warnings.catch_warnings():
                    # Ignore model linearity warning from the fitter
                    warnings.simplefilter('ignore')

                    # Use that object to check for other sources in this patch
                    sources1 = daofind(patch_data)

                # Test if more than one source was found
                isolatedBool2 = len(sources1) < 2

                # Check if there are other sources nearby
                if isolatedBool1 and isolatedBool2 and reasonableSize:
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

    def extract_star_cutouts(self, xStars, yStars, cutoutSize=21):
        """
        Locates stars and extracts 21x21 pixel cutouts center on each star

        Parameters
        ----------
        xStars : array_like
            The horizonal-axis location of each of the stars to be cutout

        yStars : array_like
            The vertical-axis location of each of the stars to be cutout

        cutoutSize : int, optional, default: 21
            The length of each side of the desired star cutouts

        Returns
        -------
        starCutouts : numpy.ndarray
            An array with shape (number of stars, cutoutSize, cutoutSize). Each
            layer of the array is the background subtracted and normalized
            cutout centered on the star.
        """

        # Define a plane fitting function for use within this method only
        def planeFit(points):
            """
            p, n = planeFit(points)

            Given an array, points, of shape (d,...)
            representing points in d-dimensional space,
            fit an d-dimensional plane to the points.
            Return a point, p, on the plane (the point-cloud centroid),
            and the normal, n.
            """

            points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
            assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
            ctr = points.mean(axis=1)
            x = points - ctr[:,np.newaxis]
            M = np.dot(x, x.T) # Could also use np.cov(x) here.
            return ctr, np.linalg.svd(M)[0][:,-1]

        # Find the isolated sources for gaussian fitting
        crowdLimit = np.sqrt(2)*cutoutSize

        # Setup the pixel coordinates for the star patch
        yy, xx = np.mgrid[0:cutoutSize, 0:cutoutSize]

        # Define the (x, y) points to sample for the plane fitting algorithm
        xyPts = (
            np.array([0,  1,  1,  0,
                      cutoutSize-2, cutoutSize-1, cutoutSize-1, cutoutSize-2,
                      cutoutSize-2, cutoutSize-1, cutoutSize-1, cutoutSize-2,
                      0,  1,  1,  0]),
            np.array([0,  0,  1,  1,
                      0,  0,  1,  1,
                      cutoutSize-2, cutoutSize-2, cutoutSize-1, cutoutSize-1,
                      cutoutSize-2, cutoutSize-2, cutoutSize-1, cutoutSize-1])
        )

        # Loop through the sources and store the background subtracted patches
        starCutouts = []
        starFluxes  = []
        for xs, ys in zip(xStars, yStars):
            # Start by cutting out the patch surrounding this star
            # Establish the cutout bondaries
            btCut = np.int(np.round(ys - np.floor(0.5*cutoutSize)))
            tpCut = np.int(np.round(btCut + cutoutSize))
            lfCut = np.int(np.round(xs - np.floor(0.5*cutoutSize)))
            rtCut = np.int(np.round(lfCut + cutoutSize))

            # Cut out the star patch and get its properties
            starCutout = self.data[btCut:tpCut, lfCut:rtCut].copy()

            # Fit a plane to the corner samples
            xyzPts = np.array(xyPts + (starCutout[xyPts],))
            point, normalVec = planeFit(xyzPts)

            # Compute the value of the fited plane background
            planeVals = (
                point[2] +
                (normalVec[0]/normalVec[2])*(xx - point[0]) +
                (normalVec[1]/normalVec[2])*(yy - point[1])
            )

            # Subtract the fitted background values plane
            starCutout -= planeVals

            # Store the total of this array for sorting later
            starFlux = starCutout.sum()
            starFluxes.append(starFlux)

            # Normalize the cutout to be have a total of one
            starCutout /= starFlux

            # Store the patch in the starCutouts
            starCutouts.append(starCutout)

        # Resort these stars from brightest to dimmest
        sortInds    = np.array(starFluxes).argsort()
        starCutouts = np.array(starCutouts)[sortInds]

        return starCutouts

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
        cutoutSize = 21

        # Grab the star positions
        xStars, yStars = self.get_sources(
            satLimit = satLimit,
            crowdLimit = crowdLimit,
            edgeLimit = cutoutSize + 1
        )

        # Grab the list of star cutouts
        starCutouts = self.extract_star_cutouts(xStars, yStars, cutoutSize=cutoutSize)

        # Loop through each cutout and grab its data properties
        sxList      = []
        syList      = []

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
            raise IndexError('There are no well behaving stars')

        # If some of the patches are bad, then cut those out of the patch list
        if np.sum(badSXSY) > 0:
            goodInds    = (np.where(goodSXSY))[0]
            starCutouts = starCutouts[goodInds, :, :]

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

        # Transfer the background portion of the patch_model to the
        # polynomial plane model.
        bkg_model.c0_0 = PSF_model.c0_0_1
        bkg_model.c1_0 = PSF_model.c1_0_1
        bkg_model.c0_1 = PSF_model.c0_1_1

        # Subtract the planar background and renormalize the median PSF
        medianPSF -= bkg_model(xx, yy)
        medianPSF /= medianPSF.sum()

        # Return the fitted PSF values
        smajor, sminor, theta = (
            patch_model.x_stddev_0.value,
            patch_model.y_stddev_0.value,
            patch_model.theta_0.value
        )

        # Define return values and return them to the user
        PSFparams = {'smajor':smajor, 'sminor':sminor, 'theta':theta}

        return (medianPSF, PSFparams)

    ##################################
    ### END OF GETTERS             ###
    ##################################

    ##################################
    ### START OF OTHER METHODS     ###
    ##################################

    def _dictionary_to_properties(self, propDict):
        """
        Sets the instance properties from the values supplied in the propDict
        """
        # Call the parent method
        super(ReducedScience, self)._dictionary_to_properties(propDict)

        # Extend the method to include the bscale property
        if 'bscale' in propDict:
            try:
                bscale = float(propDict['bscale'])
            except:
                raise TypeError('`bscale` property must be convertible to a float')
            self.__bscale = bscale
        else:
            self.__bscale = None

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

    def crop(self, x1, x2, y1, y2):
        # TODO use the self.wcs.wcs.sub() method to recompute the right wcs
        # for a cropped image.
        """
        Crops the image to the specified pixel locations.

        Parameters
        ----------
        x1, x2, y1, y2: int
            The pixel locations for the edges of the cropped image.

        Returns
        -------
        outImg: `ReducedScience`
            A copy of the image cropped to the specified locations with updated header
            and astrometry.
        """
        for p in (x1, x2, y1, y2):
            if not issubclass(type(p), (int, np.int16, np.int32, np.int64)):
                TypeError('All arguments must be integer values')

        # Check that the crop values are reasonable
        ny, nx = self.shape
        if ((x1 < 0) or (x2 > (nx - 1)) or
            (y1 < 0) or (y2 > (ny - 1)) or
            (x2 < x1) or (y2 < y1)):
            raise ValueError('The requested crop values are outside the image.')

        # Make a copy of the array and header
        outData = self.data.copy()

        # Perform the actual croping
        outData = outData[y1:y2, x1:x2]

        # Repeat the process for the sigma array if it exists
        if self._BaseImage__fullData.uncertainty is not None:
            outUncert = self.uncertainty[y1:y2, x1:x2]
            outUncert = StdDevUncertainty(outUncert)
        else:
            outUncert = None

        outHead = self.header.copy()

        # Update the header keywords
        # First update the NAXIS keywords
        outHead['NAXIS1'] = y2 - y1
        outHead['NAXIS2'] = x2 - x1

        # Next update the CRPIX keywords
        if 'CRPIX1' in outHead:
            outHead['CRPIX1'] = outHead['CRPIX1'] - x1
        if 'CRPIX2' in outHead:
            outHead['CRPIX2'] = outHead['CRPIX2'] - y1

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
        outImg._BaseImage__header = outHead

        return outImg

    def shift(self, dx, dy, padding=0.0):
        """Shift the image dx pixels to the right and dy pixels up.

        Non-integer pixel shifts are permitted and will be properly handled by
        conserving the total flux. However, this effectively convolves the image
        with a (2x2) pixel top-hat kernel. Thus, the associated uncertainties
        will also be reduced.

        Parameters
        ----------
        dx : int or float
            Number of pixels to shift right (negative is left)
        dy :
            Number of pixels to shift up (negative is down)

        padding : scalar, optional, default: 0.0
            The value to place in the empty regions after the image has been
            shifted
        """

        # TODO
        # I can probably DRAMATICALLY improve this by simply using the
        # scipy.ndimage.shift function with 'order=1'. The results are not
        # perfectly eequal, but they're so close that it's probably worth the
        # simplification (perhaps even speedier?)

        # Store the original shape of the image array
        ny, nx = self.shape

        # Check if the X shift is an within 1 billionth of an integer value
        if round(float(dx), 12).is_integer():
            # Force the shift to an integer value
            dx = np.int(round(dx))

            # Make a copy and apply the shift.
            shiftedData = np.roll(self.data, dx, axis = 1)

            # Apply the same shifts to the sigma array if it exists
            if self._BaseImage__fullData.uncertainty is not None:
                shiftUncert = np.roll(self.uncertainty, dx, axis = 1)

        else:
            # The x-shift is non-integer...
            # Compute the two integer shiftings needed
            dxRt = np.int(np.ceil(dx))
            dxLf = dxRt - 1

            # Produce the shifted arrays
            dataRt = np.roll(self.data, dxRt, axis = 1)
            dataLf = np.roll(self.data, dxLf, axis = 1)

            # Compute the fractional contributions of each array
            fracRt = np.abs(dx - dxLf)
            fracLf = np.abs(dx - dxRt)

            # Compute the shifted array
            shiftedData = fracRt*dataRt + fracLf*dataLf

            if self._BaseImage__fullData.uncertainty is not None:
                uncertRt = np.roll(self.uncertainty, dxRt, axis = 1)
                uncertLf = np.roll(self.uncertainty, dxLf, axis = 1)

                # Compute the shifted uncertainty array
                shiftedUncert = np.sqrt(
                    (fracRt*uncertRt)**2 +
                    (fracLf*sigLf)**2
                )

        # Now fill in the shifted arrays
        fillX = np.int(np.abs(np.ceil(dx)))
        if dx > 0:
            shiftedData[:,0:fillX] = padding
        elif dx < 0:
            shiftedData[:,(nx-fillX-1):nx] = padding

        # # Place the final result in the arr attribute
        # self.arr = shiftArr

        if self._BaseImage__fullData.uncertainty is not None:
            # Now fill in the shifted arrays
            if dx > 0:
                shiftedUncert[:,0:fillX] = np.abs(padding)
            elif dx < 0:
                shiftedUncert[:,(nx-fillX-1):nx] = np.abs(padding)

            # # Place the shifted array in the sigma attribute
            # self.sigma = shiftSig

        # Check if the Y shift is an within 1 billianth of an integer value
        if round(float(dy), 12).is_integer():
            # Force the shift to an integer value
            dy = np.int(round(dy))

            # Make a copy and apply the shift.
            shiftedData = np.roll(shiftedData, dy, axis = 0)

            # Apply the same shifts to the sigma array if it exists
            if self._BaseImage__fullData.uncertainty is not None:
                shiftedUncert = np.roll(shiftedUncert, dy, axis = 0)
        else:
            # The y-shift is non-integer...
            # Compute the two integer shiftings needed
            dyTp = np.int(np.ceil(dy))
            dyBt = dyTp - 1

            # Produce the shifted arrays
            dataTp = np.roll(shiftedData, dyTp, axis = 0)
            dataBt = np.roll(shiftedData, dyBt, axis = 0)

            # Compute the fractional contributions of each array
            fracTp = np.abs(dy - dyBt)
            fracBt = np.abs(dy - dyTp)

            # Compute the shifted array
            shiftedData = fracTp*dataTp + fracBt*dataBt

            # Apply the same shifts to the sigma array if it exists
            if self._BaseImage__fullData.uncertainty is not None:
                uncertTp = np.roll(shiftedUncert, dyTp, axis = 0)
                uncertBt = np.roll(shiftedUncert, dyBt, axis = 0)

                # Compute the shifted array
                shiftSig = np.sqrt(
                    (fracTp*uncertTp)**2 +
                    (fracBt*uncertBt)**2
                )

        # Filll in the emptied pixels
        fillY = np.int(np.abs(np.ceil(dy)))
        if dy > 0:
            shiftedData[0:fillY,:] = padding
        elif dy < 0:
            shiftedData[(ny-fillY-1):ny,:] = padding

        if self._BaseImage__fullData.uncertainty is not None:
            # Now fill in the shifted arrays
            if dy > 0:
                shiftedUncert[0:fillY,:] = np.abs(padding)
            elif dy < 0:
                shiftedUncert[(ny-fillY-1):ny,:] = np.abs(padding)

            # As a final step, convert this to a StdDevUncertainty
            shiftedUncert = StdDevUncertainty(shiftedUncert)
        else:
            # If NONE of these created a shifted uncertainty, then set to None
            shiftedUncert = None

        # Make a copy of the header in case it needs to be modified
        outHead = self.header.copy()

        # Check if the header contains celestial WCS coords
        if self.has_wcs:
            if self.wcs.has_celestial:

                # Update the header astrometry
                outHead['CRPIX1'] = self.header['CRPIX1'] + dx
                outHead['CRPIX2'] = self.header['CRPIX2'] + dy

                # Make the self.wcs attribute is also updated
                outWCS = WCS(outHead)

            else:
                outWCS = None
        else:
            outWCS = None

        # Copy the image and update its data
        outImg = self.copy()

        # Store the header in the output image
        outImg._BaseImage__header = outHead

        # Store the array data, utis, and WCS for the image
        outImg._BaseImage__fullData = NDDataArray(
            shiftedData,
            uncertainty=shiftedUncert,
            unit=self.unit,
            wcs=outWCS
        )

        return outImg

    ###
    # TODO: Think through whether or not (and HOW) to implement a rotation method
    ###
    # def rotate(self, angle, reshape=True, order=3, mode='constant', cval=0.0,
    #     prefilter=True, copy=True):
    #     """This is a convenience method for accessing the scipy rotate
    #     interpolator. Future versions of this method may apply a flux
    #     conservative method (such as that apparently employed by HASTROM in
    #     the IDL astrolib).
    #     """
    #     # Apply the rotation to the array and sigma arrays
    #     outArr = ndimage.interpolation.rotate(self.arr, angle,
    #         reshape=reshape, order=order, mode=mode, cval=cval,
    #         prefilter=prefilter)
    #
    #     # I have not yet sorted out how to properly apply rotation to the
    #     # WCS information in the header
    #     warnings.warn('The WCS of this image has not been rotated', Warning)
    #
    #     # If the shape of the output array has changed, then update the header
    #     ny, nx = outArr.shape
    #     if (ny, nx) != self.arr.shape:
    #         outHead = self.header.copy()
    #         outHead['NAXIS1'] = nx
    #         outHead['NAXIS2'] = ny
    #
    #     # Rotate the uncertainty image if it exists (THIS IS PROBABLY NOT THE
    #     # PROPER WAY TO HANDLE ROTATION)
    #     hasSig = hasattr(self, 'sigma')
    #     if hasSig:
    #         outSig = ndimage.interpolation.rotate(self.sigma, angle,
    #             reshape=reshape, order=order, mode=mode, cval=cval,
    #             prefilter=prefilter)
    #
    #     # Either copy and return the image, or store it in "self"
    #     if copy == True:
    #         outImg = self.copy()
    #         outImg.arr    = outArr
    #
    #         if hasSig:
    #             outImg.sigma  = outSig
    #
    #         outImg.header = outHead
    #         outImg.wcs    = WCS(outHead)
    #
    #         return outImg
    #     else:
    #         self.arr    = outarr
    #
    #         if hasSig:
    #             self.sigma  = outSig
    #
    #         self.header = outHead
    #         self.wcs    = WCS(outHead)

    # def scale(self, quantity='flux', copy=False):
    #     """
    #     Scales the data in the arr attribute using the BSCALE and BZERO
    #     values from the header. If no such values exist, then return original
    #     array.
    #     """
    #     # Test if the quantity value supplied is an acceptable format
    #     if quantity.upper() not in ['FLUX', 'INTENSITY']:
    #         raise ValueError("'quantity' must be either 'FLUX' or 'INTENSITY'")
    #
    #     # ###############################
    #     # This first section of code will determine if the array has already
    #     # been scaled, and if it has not, then it will apply the scaling factors
    #     # stored in the image header.
    #     # ###############################
    #     # Test if the array has already been scaled
    #     if self.is_scaled:
    #         # If the array has not been set to either FLUX or INTENSITY, then
    #         # apply the scaling to the array and store the correct quantity
    #
    #         # Grab the scaling constants
    #         if 'BSCALE' in self.header.keys():
    #             if quantity.upper() == 'FLUX':
    #                 scaleConst1 = self.header['BSCALE']
    #
    #                 # Check for uncertainty in BSCALE
    #                 if 'SBSCALE' in self.header.keys():
    #                     sig_scaleConst1 = self.header['SBSCALE']
    #
    #             elif quantity.upper() == 'INTENSITY':
    #                 pixArea     = proj_plane_pixel_area(self.wcs)*(3600**2)
    #                 scaleConst1 = self.header['BSCALE']/pixArea
    #
    #                 # Check for uncertainty in BSCALE
    #                 if 'SBSCALE' in self.header.keys():
    #                     sig_scaleConst1 = self.header['SBSCALE']/pixArea
    #         else:
    #             scaleConst1 = 1
    #
    #         if 'BZERO' in self.header.keys():
    #             scaleConst0 = self.header['BZERO']
    #         else:
    #             scaleConst0 = 0
    #
    #         # Perform the actual scaling!
    #         scaledArr = scaleConst1*self.arr.copy() + scaleConst0
    #
    #         # Apply scaling uncertainty if available
    #         if hasattr(self, 'sigma'):
    #             # If there is an uncertainty in the scaling factor, then
    #             # propagate that into the uncertainty
    #             if 'SBSCALE' in self.header.keys():
    #                 # Include the uncertainty in the scaling...
    #                 sigArr = np.abs(scaledArr)*np.sqrt((self.sigma/self.arr)**2
    #                     + (sig_scaleConst1/scaleConst1)**2)
    #             else:
    #                 # Otherwise just scale up the uncertainty...
    #                 sigArr  = self.sigma.copy()
    #                 sigArr *= scaleConst1
    #
    #         # Check if a copy of the image was requested
    #         if copy:
    #             # Store the output array in a copy of this image
    #             outImg = self.copy()
    #             outImg.arr = scaledArr
    #
    #             # Set the _scaled_quantity property
    #             outImg._scaled_quantity = quantity.upper()
    #
    #             # Try to store the sigArr array in the output image
    #             try:
    #                 outImg.sigma = sigArr
    #             except:
    #                 # If no sigArr variable was found, then simply skip that
    #                 pass
    #
    #             return outImg
    #         else:
    #             # Set the _scaled_quantity property
    #             self._scaled_quantity = quantity.upper()
    #
    #             # Store the output array
    #             self.arr = scaledArr
    #
    #             # Try to store the sigArr array in the original image
    #             try:
    #                 self.sigma = sigArr
    #             except:
    #                 # If no sigArr variable was found, then simply skip that
    #                 pass
    #
    #     # ###############################
    #     # This second section of code will determine if the array has already
    #     # been scaled, and if it has, then it will reverse the the scaling
    #     # and restore the array and uncertainty to their original values.
    #     # ###############################
    #     elif not self.is_scaled:
    #         # If the array HAS been set to either FLUX or INTENSITY, then return
    #         # the array and uncertainty to their original values
    #
    #         # Grab the scaling constants
    #         if 'BSCALE' in self.header.keys():
    #             if self._scaled_quantity == 'FLUX':
    #                 scaleConst1 = self.header['BSCALE']
    #
    #                 # Check for uncertainty in BSCALE
    #                 if 'SBSCALE' in self.header.keys():
    #                     sig_scaleConst1 = self.header['SBSCALE']
    #
    #             elif self._scaled_quantity == 'INTENSITY':
    #                 pixArea     = proj_plane_pixel_area(self.wcs)*(3600**2)
    #                 scaleConst1 = self.header['BSCALE']/pixArea
    #
    #                 # Check for uncertainty in BSCALE
    #                 if 'SBSCALE' in self.header.keys():
    #                     sig_scaleConst1 = self.header['SBSCALE']/pixArea
    #         else:
    #             scaleConst1 = 1
    #
    #         if 'BZERO' in self.header.keys():
    #             scaleConst0 = self.header['BZERO']
    #         else:
    #             scaleConst0 = 0
    #
    #         # Perform the actual scaling!
    #         unScaledArr = (self.arr.copy() - scaleConst0)/scaleConst1
    #
    #         # Apply scaling uncertainty if available
    #         if hasattr(self, 'sigma'):
    #             # If there is an uncertainty in the scaling factor, then
    #             # propagate that into the uncertainty
    #             if 'SBSCALE' in self.header.keys():
    #                 # Include the uncertainty in the scaling...
    #                 sigArr = np.abs(unScaledArr)*np.sqrt((self.sigma/self.arr)**2
    #                     - (sig_scaleConst1/scaleConst1)**2)
    #             else:
    #                 # Otherwise just scale up the uncertainty...
    #                 sigArr  = self.sigma.copy()
    #                 sigArr /= scaleConst1
    #
    #         # Check if a copy of the image was requested
    #         if copy:
    #             # Store the output array in a copy of this image
    #             outImg = self.copy()
    #             outImg.arr = unScaledArr
    #
    #             # Set the _scaled_quantity property
    #             outImg._scaled_quantity = quantity.upper()
    #
    #             # Try to store the sigArr array in the output image
    #             try:
    #                 outImg.sigma = sigArr
    #             except:
    #                 # If no sigArr variable was found, then simply skip that section
    #                 pass
    #
    #             return outImg
    #         else:
    #             # Set the _scaled_quantity property to None
    #             self._scaled_quantity = None
    #
    #             # Store the output array
    #             self.arr = unScaledArr
    #
    #             # Try to store the sigArr array in the original image
    #             try:
    #                 self.sigma = sigArr
    #             except:
    #                 # If no sigArr variable was found, then simply skip that section
    #                 pass

    def rebin(self, nx, ny, total=False):
        # Extend the rebin method to update the WCS
        # Start by applying the basic rebin method
        outImg = super(ReducedScience, self).rebin(nx, ny, total=total)

        # Extract the shape and rebinning properties
        ny1, nx1 = self.shape
        dxdy     = np.array([nx1/nx, ny1/ny])

        if self.has_wcs:
            # Now treat the WCS
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
                uncertainty=outImg.uncertainty,
                unit=outImg.unit,
                wcs=WCS(outImg.header)
            )

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

    def gradient(self, kernel='sobel'):
        """
        Computes the gradient (Gx, Gy) of the image.

        Parameters
        ----------
        kernel : str ('sobel' or 'prewitt'), optional, default: 'sobel'
            The kernel to use for computing the gradient.

            sobel: https://en.wikipedia.org/wiki/Sobel_operator
            prewitt: https://en.wikipedia.org/wiki/Prewitt_operator

        Returns
        -------
        Gx, Gy : numpy.ndarray
            A tuple of numpy.ndarray instances containing the gradient along the
            horizontal axis (Gx) and vertical axis (Gy). The total magnitude of
            the graident can be computed as sqrt(Gx**2 + Gy**2)
        """
        if type(kernel) is not str:
            raise TypeError('`kernel` must be a string specifying the graident operator to use')

        if kernel.upper() == 'SOBEL':
            Gx = ndimage.sobel(self.data, axis=1)
            Gy = ndimage.sobel(self.data, axis=0)
        elif kernel.upper() == 'PREWITT':
            Gx = ndimage.prewitt(self.data, axis=1)
            Gy = ndimage.prewitt(self.data, axis=0)
        else:
            raise ValueError('`kernel` must be "SOBEL" or "PREWITT"')

        return (Gx, Gy)

    def in_image(self, coords, edge=0):
        """
        Tests which (RA, Dec) coordinates lie within the image frame.

        Parameters
        ----------
        coords: scalar or array_like
            The (RA, Dec) coordinates to examine if they are locawed within the
            image frame. Elements of an array must be tuple of convertible
            (RA, Dec) pairs or astropy.coordinates.SkyCoord instances.

        edge: int or float
            Specifies the amount of image to ignore (arcsec). If the specified
            point is within the image but is less than `edge` arcsec from the
            edge of the image, then a False value is returned.

        Returns:
        out: scalar or array
            A boolean True value for those elements of `coord` which are located
            within the image footprint and False for elements located outside
            the image footprint.
        """
        if not self.has_wcs:
            raise AttributeError('Image does not have an astrometric solution defined.')

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
        ny, nx = self.shape

        # Check which coordinates fall within the image
        xGood = np.logical_and(x > edge, x < (nx - edge - 1))
        yGood = np.logical_and(y > edge, y < (ny - edge - 1))
        allGood = np.logical_and(xGood, yGood)

        return allGood

    def correct_airmass(self, atmExtCoeff=None):
        """
        Corrects for atmospheric extinction using the airmass of this image.

        Parameters
        ----------
        atmExtCoeff : float
            The atmospheric extinction coefficient for the observing site at the
            time of the observation. This is usually determined from a series of
            observations of standard stars throughout the course of the night.

        Returns
        -------
        outImg : ReducedScience
            A copy of the original image with an muliplicative correction for
            the airmass exctinction applied to it.
        """
        # Check if an atmospheric extniction coefficient was set
        if not issubclass(
            type(atmExtCoeff),
            (float, np.float, np.float16, np.float32, np.float64)):
            raise TypeError('Must provide a floating point value for required argument `atmExtCoeff`')

        # Check if the airmass has been set
        if self.airmass is None:
            raise ValueError('This image has no estimate of the airmass.')

        # Compute the airmass corrected intensity
        correctionFactor = (10.0**(0.4*atmExtCoeff*self.airmass))
        outData          = self.data*correctionFactor

        if self._BaseImage__fullData.uncertainty is not None:
            outUncert = self.uncertainty*correctionFactor
            outUncert = StdDevUncertainty(outUncert)
        else:
            outUncert = None

        # Store this data in the __fullData attribute
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(
            outData,
            uncertainty=outUncert,
            unit=self.unit,
            wcs=self.wcs
        )

        # Update the airmas info in the header
        headerKeywordDictKeys = self.headerKeywordDict.keys()
        if 'AIRMASS' in self.headerKeywordDict:
            outHead    = self.header.copy()
            airmassKey = self.headerKeywordDict['AIRMASS']
            outHead[airmassKey] = 0.0

            outImg._BaseImage__header = outHead

        # Update the airmass property value to be zero
        outImg._BaseImage__airmass = 0.0

        return outImg

    # def fix_astrometry(self):
    #     """This ensures that the CDELT values and PC matrix are properly set."""
    #
    #     # Check if there is a header in this image
    #     if self.has_wcs:
    #         pix_scales = self.pixel_scales
    #         if len(self.header['CDELT*']) > 0:
    #             # If there are CDELT values,
    #             if ((pix_scales[0] != self.header['CDELT1']) or
    #                 (pix_scales[1] != self.header['CDELT2'])):
    #                 # and if they are not ACTUALLY set to the plate scales,
    #                 # then update the astrometry keyword values
    #                 CDELT1p = pix_scales[0]
    #                 CDELT2p = pix_scales[1]
    #
    #                 # Update the header values
    #                 self.header['CDELT1'] = CDELT1p
    #                 self.header['CDELT2'] = CDELT2p
    #                 self.header['PC1_1']  = self.header['PC1_1']/CDELT1p
    #                 self.header['PC1_2']  = self.header['PC1_2']/CDELT1p
    #                 self.header['PC2_1']  = self.header['PC2_1']/CDELT2p
    #                 self.header['PC2_2']  = self.header['PC2_2']/CDELT2p
    #     else:
    #         raise ValueError('No header in this imagae')

    def clear_astrometry(self):
        """Delete the header values pertaining to the astrometry."""

        # Define a FULL list of things to delete from the header
        wcsKeywords = [
            'WCSAXES',
            'CRPIX1', 'CRPIX2',
            'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
            'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
            'CDELT1', 'CDELT2',
            'CUNIT1', 'CUNIT2',
            'CTYPE1', 'CTYPE2',
            'CRVAL1', 'CRVAL2',
            'LONPOLE',
            'LATPOLE',
            'RADESYS',
            'EQUINOX',
            'A_1_1',  'A_0_2', 'A_2_0', 'A_ORDER',
            'AP_0_1', 'AP_1_0', 'AP_1_1', 'AP_0_2', 'AP_2_0', 'AP_ORDER',
            'B_1_1', 'B_0_2', 'B_2_0', 'B_ORDER',
            'BP_1_0', 'BP_1_1', 'BP_0_1', 'BP_0_2', 'BP_2_0', 'BP_ORDER'
        ]

        # Loop through and delete any present keywords
        for key in wcsKeywords:
            if key in self.header:
                del self.header[key]

        # Also force clear any stored WCS object from the __fullData attribute
        self._BaseImage__fullData = NDDataArray(
            self.data,
            uncertainty=StdDevUncertainty(self.uncertainty),
            unit=self.unit,
            wcs=None
        )

    def astrometry_to_header(self, wcs):
        """
        Places the astrometric information from a WCS object into the header.

        Parameters
        ----------
        wcs : astropy.wcs.wcs.WCS
            A WCS object containing the astrometric information to be placed in
            the header.

        Returns
        -------
        out : None
        """
        # Start by clearing out the old astrometric information
        self.clear_astrometry()

        # Update the image center coordinates
        yc, xc  = 0.5*np.array(self.shape)
        ra, dec = wcs.all_pix2world([yc], [xc], 0)
        coord = SkyCoord(
            ra=ra,
            dec=dec,
            unit=(u.degree, u.degree)
        )
        self._BaseImage__centerCoord = coord

        # Update the header image center coordinates
        if self.ra is not None:
            try:
                raKey = self.headerKeywordDict['RA']
                self.__header[raKey] = self.ra.to_string(unit=u.hour, sep=':')
            except: pass
        else:
            try:
                raKey = self.headerKeywordDict['RA']
                del self.__header[raKey]
            except: pass

        if self.dec is not None:
            try:
                decKey = self.headerKeywordDict['DEC']
                self.__header[decKey] = self.dec.to_string(unit=u.degree, sep=':')
            except: pass
        else:
            try:
                decKey = self.headerKeywordDict['DEC']
                del self.__header[deckey]
            except: pass

        # Convert the wcs to a header
        wcsHeader = wcs.to_header(relax=True)

        # Delete the PC and CDELT keywords because we use CDELT for something else....
        if len(wcsHeader['PC*']) > 0:
            del wcsHeader['PC*']

        if len(wcsHeader['CDELT*']) > 0:
            del wcsHeader['CDELT*']

        if wcs.wcs.has_cd():
            # Grab the cd matrix
            cd = wcs.wcs.cd
        elif wcs.wcs.has_pc():
            # Convert the pc matrix into a cd matrix
            cd = wcs.wcs.cdelt*wcs.wcs.pc
        else:
            raise ValueError('`wcs` does not include proper astrometry')

        # Loop through the CD values and replace them with updated values
        for i, row in enumerate(cd):
            for j, cdij in enumerate(row):
                key = 'CD' + '_'.join([str(i+1), str(j+1)])
                wcsHeader[key] = cdij

        # Update the header
        self._BaseImage__header.update(wcsHeader)

        return None

    def oplot_sources(self, satLimit=16e3, crowdLimit=0.0,
        s=100, marker='o', edgecolor='red', facecolor='none', **kwargs):
        """
        Overplot detected sources with markers.

        Parameters
        ----------
        satLimit : int or float, optional, default: 16e3
            Sources which contain any pixels with more than this number of
            counts will be discarded from the returned list of sources on
            account of being saturated.

        crowdLimit : int or float, optional, default: 0
            Sources with a neighest neighbor closer than this distance (in
            pixels) will be discarded from the returned list of sources on
            account of being too crowded.

        s : int or float
            size in points^2. Default is rcParams['lines.markersize'] ** 2

        marker : MarkerStyle, optional, default: 'o'
            See markers for more information on the different styles of markers
            scatter supports. marker can be either an instance of the class or
            the text shorthand for a particular marker.

        edgecolor : , optional, default: 'red'

        facecolor : , optional, default: 'none'

        Other Paramaters
        ----------------
        Accepts any other keywords permissible by the matplotlib.pyplot.scatter
        method.
        """
        # Grab the sources using the get_sources() method
        xs, ys = self.get_sources(satLimit=satLimit, crowdLimit=crowdLimit)

        # The following line makes it so that the zoom level no longer changes,
        # otherwise Matplotlib has a tendency to zoom out when adding overlays.
        ax = self.axes
        ax.set_autoscale_on(False)

        # Overplot the sources
        ax.scatter(xs, ys, s=s, edgecolor=edgecolor, facecolor=facecolor, **kwargs)

        # Force a redraw of the figure
        # Force a redraw of the canvas
        fig = self.figure.canvas.draw()
