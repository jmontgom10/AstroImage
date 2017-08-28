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
from astropy.io import fits
from astropy.nddata import NDDataArray, StdDevUncertainty
from astropy.modeling import models, fitting
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, FK4, FK5
from astropy.wcs.utils import proj_plane_pixel_scales, proj_plane_pixel_area
from astropy import units as u
from astropy.stats import sigma_clip, sigma_clipped_stats
from photutils import DAOStarFinder, data_properties

# Matplotlib imports
import matplotlib as mpl
import matplotlib.colors as mcol
import matplotlib.pyplot as plt

# AstroImage imports
from .baseimage import BaseImage, ClassProperty
from .mixins import NumericsMixin, ResizingMixin

# Define which functions, classes, objects, etc... will be imported via the command
# >>> from .reducedimage import *
__all__ = ['MasterBias', 'MasterDark', 'MasterFlat', 'ReducedScience']

class ReducedImage(BaseImage):
    """
    The base class for reduced calibration and science data.

    Defines additional mathematical operations (e.g., trig and log functions)

    Properties
    ----------
    airmass         The airmass of the observation
    axes            The Axes instance storing the plotted image (if plotted)
    data            The actual 2D numpy array of the image in the fits file
    binning         The binning of the image as a tuple, returned as binning
                    along the height axis, then width axis
    date            The UTC date and time of the observation in the format
                    YYYY-MM-DD   HH:MM:SS.SS
    dec             The declination of the observation in the format
                    +DD:(AM)(AM):(AS)(AS).(AS)(AS)
    dtype           The data type of the image array (see `numpy.ndarray`)
    expTime         The total exposure time, in seconds
    figure          The Figure instance storing the plotted axes (if plotted)
    filename        The image filename on disk
    filter          The filter through which the image was obtained
    header          The header info for the associated fits file.
    height          The height of the image, in pixels.
    image           The AxesImage storing the plottted data (if plotted)
    instrument      The instrument from which the image was obtained
    ra              The right ascension of the observation in the format
                    HH:MM:SS.SS
    shape           Dimensions of the image as a tuple, returned as height, then
                    width, in pixels, in keeping with the behavior of the
                    `numpy.ndarray` size attribute.
    uncertainty     The array of uncertainties associated with `data`
    units           The units of the numpy 2D array stored in `data`
    width           The width of the image, in pixels

    Class Methods
    -------------
    set_headerKeywordDict
    read

    Methods
    -------
    set_arr
    set_uncertainty
    set_header
    copy
    write
    rebin
    show

    Examples
    --------
    Read in calibrated files
    >>> from astroimage import ReducedScience
    >>> img1 = ReducedScience.read('img1.fits')
    >>> img2 = ReducedScience.read('img2.fits')

    Check that the images are the same dimensions
    >>> img1.shape, img2.shape
    ((500, 500), (500, 500))

    Now compute the mean of those two images
    >>> img3 = 0.5*(img1 + img2)

    Rebin the resulting image
    >>> img3.rebin(250, 250)

    Check that the new image has the correct dimensions
    >>> img3.shape
    (250, 250)

    Display the resultant average, rebinned image
    >>> fig, ax, axImg = img3.show()
    """

    ##################################
    ### START OF CLASS VARIABLES   ###
    ##################################

    # Extend the list of acceptable properties for this class
    __properties = copy.deepcopy(BaseImage.properties)
    __properties.extend([
        'uncertainty' # This property and below are for the ReducedImage class
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
        """
        Constructs a `ReducedImage` instance from provided arguments.

        More properly implemented by subclasses
        Parameters
        ----------
        data : `numpy.ndarray`, optional
            The array of values to be stored for this image

        uncertainty : `numpy.ndarray`, optional
            The uncertainty of the values to be stored for this image

        header : `astropy.io.fits.header.Header`, optional
            The header to be associated with the `arr` attribute

        properties : `dict`, optional
            A dictionary of properties to be set for this image
            (e.g. {'unit': u.adu, 'ra': 132.323, 'dec': 32.987})

        Returns
        -------
        outImg : `ReducedImage` (or subclass)
            A new instance containing the supplied data, header, and
            properties
        """
        # Start by instantiating the basic BaseImage type information
        super(ReducedImage, self).__init__(*args, **kwargs)

    ##################################
    ### START OF PROPERTIES        ###
    ##################################

    @property
    def has_uncertainty(self):
        """Boolean flag if the `uncertainty` property exists"""
        hasUncert = False
        try:
            hasUncert = self._BaseImage__fullData.uncertainty is not None
        except:
            pass
        try:
            hasUncert = self._BaseImage__fullData.uncertainty.array is not None
        except:
            pass

        return hasUncert

    @property
    def uncertainty(self):
        """The uncertainties associated with the `data` values"""
        if self.has_uncertainty:
            return self._BaseImage__fullData.uncertainty.array
        else:
            return None

    @uncertainty.setter
    def uncertainty(self, uncert):
        """
        Used to replace the private `uncertainty` attribute.

        Parameters
        ----------
        uncert : numpy.ndarray
            An array containing the array to be placed in the private
            `uncertainty` property

        Returns
        -------
        out : None
        """
        # Test if arr is a numpy array
        if not isinstance(uncert, np.ndarray):
            raise TypeError('`uncert` must be an instance of numpy.ndarray')

        # Test if the replacement array matches the previous array's shape
        if uncert.shape != self.shape:
            raise ValueError('`uncert` must have shape ({0}x{1})'.format(
                *self.shape))

        # Update the image uncertainty
        self._BaseImage__fullData = NDDataArray(
            self.data,
            uncertainty=StdDevUncertainty(uncert),
            unit=self._BaseImage__fullData.unit,
            wcs=self._BaseImage__fullData.wcs
        )

    @property
    def snr(self):
        """The signal-to-noise ratio of the data and uncertainty in this image"""
        if self.has_uncertainty:
            return self.data/self.uncertainty
        else:
            return None

    ##################################
    ### END OF PROPERTIES        ###
    ##################################

    ##################################
    ### START OF OTHER METHODS     ###
    ##################################

    def _build_HDUs(self):
        # Invoke the parent method to build the basic HDU
        HDUs = super(ReducedImage, self)._build_HDUs()

        if self.uncertainty is not None:
             # Bulid a secondary HDU
            sigmaHDU = fits.ImageHDU(data = self.uncertainty.astype(self.dtype),
                                     name = 'UNCERTAINTY',
                                     do_not_scale_image_data=True)
            HDUs.append(sigmaHDU)

        return HDUs

    def divide_by_expTime(self):
        """Divides the image by its own exposure time and sets expTime to 1"""
        # Divide by the exposure time
        outImg = self/self.expTime

        # Modify the expTime value
        outImg._BaseImage__expTime = 1.0

        # Make sure the header is updated, too
        outImg._properties_to_header()

        return outImg

    def show_uncertainty(self, axes=None, cmap='viridis', vmin=None, vmax=None,
            origin='lower', interpolation='nearest', noShow=False,
            stretch='linear', **kwargs):
        """Displays the array stored in the `uncertainty` property."""
        # Check if there is an uncertainty to display
        if not self.has_uncertainty:
            raise ValueError('No `uncertainty` array in this instance to display.')

        # Build the appropriate axes for this object
        axes = self._build_axes(axes=axes)

        # TODO: Figure out whether or not to store this AxesImage instance
        # Display the signal-to-noise ratio
        image = self._show_array(self.uncertainty, axes=axes, cmap=cmap,
            vmin=vmin, vmax=vmax, origin=origin, interpolation=interpolation,
            noShow=noShow, stretch=stretch, **kwargs)

    def show_snr(self, axes=None, cmap='viridis', vmin=None, vmax=None,
            origin='lower', interpolation='nearest', noShow=False,
            stretch='linear', **kwargs):
        """Displays the signal-to-noise ratio for this image."""
        # Check if there is an uncertainty to compute SNR
        if not self.has_uncertainty:
            raise ValueError('No `uncertainty` array in this instance to compute SNR.')

        # Build the appropriate axes for this object
        axes = self._build_axes(axes=axes)

        # TODO: Figure out whether or not to store this AxesImage instance
        # Display the signal-to-noise ratio
        image = self._show_array(self.snr,  axes=axes, cmap=cmap, vmin=vmin,
            vmax=vmax, origin=origin, interpolation=interpolation,
            noShow=noShow, stretch=stretch, **kwargs)

        return image

    ##################################
    ### END OF OTHER METHODS       ###
    ##################################

##################################
### START OF SUBCLASSES        ###
##################################

class MasterBias(ReducedImage):
    """A class for reading in reduced master bias frames."""

    def __init__(self, *args, **kwargs):
        super(MasterBias, self).__init__(*args, **kwargs)

        if self.obsType != 'BIAS':
            raise IOError('Cannot instantiate a RawBias with a {0} type image.'.format(
                self.obsType
            ))


class MasterDark(ReducedImage):
    """A class for reading in reduced master dark frames."""

    def __init__(self, *args, **kwargs):
        super(MasterDark, self).__init__(*args, **kwargs)

        if self.obsType != 'DARK':
            raise IOError('Cannot instantiate a RawDark with a {0} type image.'.format(
                self.obsType
            ))

    ##################################
    ### START OF PROPERTIES        ###
    ##################################

    @property
    @lru_cache()
    def is_significant(self):
        """Boolean flag if the dark current is greater than 2x(read noise)"""
        return (np.median(self.data)/np.std(self.data)) > 2.0

    ##################################
    ### END OF PROPERTIES        ###
    ##################################

class MasterFlat(ReducedImage):
    """A class for reading in reduced master flat frames."""

    def __init__(self, *args, **kwargs):
        super(MasterFlat, self).__init__(*args, **kwargs)

        if self.obsType != 'FLAT':
            raise IOError('Cannot instantiate a RawFlat with a {0} type image.'.format(
                self.obsType
            ))

class ReducedScience(ResizingMixin, NumericsMixin, ReducedImage):
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
                return u.Quantity(rotAng, u.degree)
            else:
                # No rotation was found, so just return a zero
                return u.Quantity(0.0, u.degree)

        else:
            raise AttributeError('This image does not have a wcs defined')

    # TODO: handling physical units via the `units` property, so this may not
    # be a relevant property to keep.
    @property
    def is_scaled(self):
        """Boolean flag of whether `data` is in scaled units or ADU"""
        return self.__is_scaled

    ##################################
    ### END OF PROPERTIES          ###
    ##################################

    ##################################
    ### START OF GETTERS           ###
    ##################################
    # TODO: add "make_source_mask" convenience method to handle masking and
    # cache the result
    @lru_cache()
    def get_sources(self, FWHMguess=3.0, minimumSNR=7.0, satLimit=16e3,
        crowdLimit=0, edgeLimit=50):
        """Implements the daofind algorithm to extract source positions.

        Parameters
        ----------
        FWHMguess : int or float, optional, default: 3.0
            An estimate of the star full-width-at-half-maximum to be used in the
            convolution kernel for searching for stars.

        minimumSNR : int or float, optional, default: 5.0
            The minimum signal-to-noise ratio to consider a source "detected"

        satLimit : int or float, optional, default: 16e3
            Sources which contain any pixels with more than this number of
            counts will be discarded from the returned list of sources on
            account of being saturated.

        crowdLimit : int or float, optional, default: 0
            Sources with a neighest neighbor closer than this distance (in
            pixels) will be discarded from the returned list of sources on
            account of being too crowded.

        edgeLimit : int, or float, optional, default: 80
            Sources detected within this distance (in pixels) of the image edge
            will be discarded from the returned list of sources on account of
            being too close to the image edge. The default value of 80 should
            sufficiently cull any false positives from edge-effects.

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
        daofind = DAOStarFinder(fwhm=FWHMguess, threshold=minimumSNR*std)

        # Use that object to find the stars in the image
        sources = daofind(tmpData - median)

        # Grab the image shape for later use
        ny, nx = tmpData.shape

        # Cut out edge stars if requested
        if edgeLimit > 0:
            xStars, yStars = sources['xcentroid'], sources['ycentroid']
            nonEdgeStars = xStars > edgeLimit
            nonEdgeStars = np.logical_and(nonEdgeStars,
                xStars < nx - edgeLimit - 1)
            nonEdgeStars = np.logical_and(nonEdgeStars,
                yStars > edgeLimit)
            nonEdgeStars = np.logical_and(nonEdgeStars,
                yStars < ny - edgeLimit - 1)

            # Cull the sources list to only include non-edge stars
            if np.sum(nonEdgeStars) > 0:
                nonEdgeInds = np.where(nonEdgeStars)
                sources     = sources[nonEdgeInds]
            else:
                warnings.warn('There are no non-edge stars')
                return np.array([None]), np.array([None])
                # raise IndexError('There are no non-edge stars')


        # Perform the saturation test
        notSaturated = []

        # Initalize a circular mask for the star patches
        xx, yy = np.mgrid[-15:16, -15:16]
        radialDist = np.sqrt(xx**2 + yy**2)
        circularMask = (radialDist <= 15.1).astype(np.float)

        for xStar, yStar in zip(sources['xcentroid'], sources['ycentroid']):
            # Compute the boundaries of a small cutout for this star
            bt = np.int(np.floor(yStar - 15))
            tp = bt + 31
            lf = np.int(np.floor(xStar - 15))
            rt = lf + 31

            # Grab the star cutout for this star.
            starStamp = tmpData[bt:tp, lf:rt]*circularMask

            # Test if the maximum of the star region is a saturated value
            notSaturated.append(starStamp.max() < satLimit)

        # Cull the sources list to ONLY include non-saturated sources
        if np.sum(notSaturated) > 0:
            notSaturatedInds = np.where(notSaturated)
            sources          = sources[notSaturatedInds]
        else:
            warnings.warn('No sources passed the saturation test')
            return np.array([None]), np.array([None])
            # raise IndexError('No sources passed the saturation test')

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
                reasonableSize = sizeParam < 21.0

                # Check if the source is a reasonable SHAPE
                shapeParam = props.semimajor_axis_sigma.value / props.semiminor_axis_sigma.value
                reasonableShape = shapeParam < 1.2

                # Compute the distance from the cutout center and check if the
                # centroid is located within 5 pixels of the center. If it,s
                # not, then this source is on an INCREDIBLY steep gradient.
                # TODO: Remove gradient using the planeFit function.
                cutoutSize = tpCut - btCut
                distanceFromCenter = np.sqrt(
                    (props.xcentroid.value - 0.5*cutoutSize)**2 +
                    (props.ycentroid.value - 0.5*cutoutSize)**2
                )
                reasonablePosition = distanceFromCenter < 5.0

                # # Directly examine the data and see what's going wrong.
                # from photutils import EllipticalAperture
                # position = (props.xcentroid.value, props.ycentroid.value)
                # r = 1.0 # approximate isophotal extent
                # aSem = props.semimajor_axis_sigma.value * 1
                # bSem = props.semiminor_axis_sigma.value * 1
                # theta = props.orientation.value
                # apertures = EllipticalAperture(position, aSem, bSem, theta=theta)
                #
                # # Plot it up
                # plt.ion()
                # plt.imshow(patch_data, origin='lower', cmap='viridis',
                #     interpolation='nearest')
                # apertures.plot(color='#d62728')
                #
                #
                # import pdb; pdb.set_trace()
                # plt.clf()

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
                if (isolatedBool1 and isolatedBool2 and
                    reasonableSize and reasonablePosition and reasonableShape):
                    isolatedSource.append(True)
                else:
                    isolatedSource.append(False)

            # Cull the sources list to ONLY include non-crowded sources
            if np.sum(isolatedSource) > 2:
                isolatedInds = np.where(isolatedSource)
                sources = sources[isolatedInds]
            else:
                warnings.warn('No sources passed the crowding test')
                return np.array([None]), np.array([None])
                # raise IndexError('No sources passed the crowding test')

        # Grab the x, y positions of the sources and return as arrays
        xs, ys = sources['xcentroid'].data, sources['ycentroid'].data

        # TODO: return semiminor and semimajor axes for each source???
        return xs, ys

    @lru_cache()
    def get_sources_at_coords(self, searchCoords, pointingTolerance=2.5,
        **kwargs):
        """
        Finds any sources near to the specified coordinates.

        If no star meeting the detection critera can be found at a
        given coordinate, then a NaN is returned for that star.

        Parameters
        ----------
        searchCoords : astropy.coordinates.SkyCoord
            The coordinates at which to look for stars. If this object contains
            multiple entries, the pixel coordinates of the nearest star meeting
            the detection criteria will be returned, or a None value if no stars
            near that point meet the detection criteria.

        pointingTolerance : int or float, optional, default: 2.5
            The maximum difference (in arcsec) between the expected location and
            the detected location.

        Other Parameters
        ----------------
        Also takes any of the keyword arguments for the `find_sources` method,
        and passes them to that method as it searches for stars in the image.

        Returns
        -------
        xStars, yStars : numpy.ndarray
            The pixel positions of the detected stars.
        """
        # Find all the stars in the image
        xAllStars, yAllStars = self.get_sources(**kwargs)

        # Convert the star positions to celestial coordinates
        allRAs, allDecs = self.wcs.wcs_pix2world(xAllStars, yAllStars, 0, ra_dec_order=True)
        starCoords = SkyCoord(
            ra=allRAs,
            dec=allDecs,
            unit=u.degree,
            frame=FK5
        )

        # Match the two lists of coordinates
        idx, d2d, _ = searchCoords.match_to_catalog_sky(starCoords)

        # Cull the detected positions to only include positively matched values
        xStars, yStars = xAllStars[idx], yAllStars[idx]

        # Test if the matches are within the tolerance.
        toleranceQuantity = u.Quantity(pointingTolerance, u.arcsec).to(u.degree)
        badMatches        = d2d >  toleranceQuantity
        badInds           = np.where(badMatches)

        # Replace the matches stars outside of the tolerance with NaN value
        xStars[badInds] = np.NaN
        yStars[badInds] = np.NaN

        return xStars, yStars

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
        # Test for a "NO STARS" signal
        if ((xStars.size == 1 and xStars[0] is None) or
            (yStars.size == 1 and yStars[0] is None)):
            # Return a list with NO star cutouts
            return [None]

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

            # Grab the set of points to which to fit a plane
            xyzPts = np.array(xyPts + (starCutout[xyPts],))

            # Skip this one if there are some non-finite values
            if np.sum(np.logical_not(np.isfinite(xyzPts))) > 0: continue

            # Fit a plane to the corner samples
            # TODO: double check that this plan-fit procedure is working properly
            point, normalVec = planeFit(xyzPts)

            # Compute the value of the fited plane background
            planeVals = (
                point[2] +
                (normalVec[0]/normalVec[2])*(xx - point[0]) +
                (normalVec[1]/normalVec[2])*(yy - point[1])
            )

            # Subtract the fitted background values plane
            starCutout -= planeVals

            # Store the patch in the starCutouts
            starCutouts.append(starCutout)

        return starCutouts

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
                shiftedUncert = np.roll(self.uncertainty, dx, axis = 1)

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
                    (fracLf*uncertLf)**2
                )

        # Now fill in the shifted arrays
        fillX = np.int(np.abs(np.ceil(dx)))
        if dx > 0:
            shiftedData[:,0:fillX] = padding
        elif dx < 0:
            shiftedData[:,(nx-fillX-1):nx] = padding

        # Place the final result in the fullData attribute
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

        # Transform coordinates to pixel positions (I was hoping to use the SIP
        # polynomials at this point, but that seems to be causing errors!)
        x, y = self.wcs.wcs_world2pix(coords.ra, coords.dec, 0)

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

    def clear_sip_keys(self):
        """Delete the Simple Image Polynomial (SIP) keys from header"""

        # Define a FULL list of things to delete from the header
        wcsKeywords = [
            'A_1_1',  'A_0_2', 'A_2_0', 'A_ORDER',
            'AP_0_1', 'AP_1_0', 'AP_1_1', 'AP_0_2', 'AP_2_0', 'AP_ORDER',
            'B_1_1', 'B_0_2', 'B_2_0', 'B_ORDER',
            'BP_1_0', 'BP_1_1', 'BP_0_1', 'BP_0_2', 'BP_2_0', 'BP_ORDER'
        ]

        # Loop through and delete any present keywords
        for key in wcsKeywords:
            if key in self.header:
                del self.header[key]

        # Also force reset any stored WCS object from the __fullData attribute
        self._BaseImage__fullData = NDDataArray(
            self.data,
            uncertainty=StdDevUncertainty(self.uncertainty),
            unit=self.unit,
            wcs=WCS(self.header)
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
        # Check if the provided wcs is the right type
        if not isinstance(wcs, WCS):
            raise TypeError('`wcs` must be an astropy.wcs.wcs.WCS instance.')

        # Check if the provided wcs has actual astrometry.
        if wcs.wcs.has_cd():
            # Grab the cd matrix
            cd = wcs.wcs.cd
        elif wcs.wcs.has_pc():
            # Convert the pc matrix into a cd matrix
            cd = wcs.wcs.cdelt*wcs.wcs.pc
        else:
            raise ValueError('`wcs` does not include proper astrometry')

        # Start by clearing out the old astrometric information
        self.clear_astrometry()

        # Update the image center coordinates
        yc, xc  = 0.5*np.array(self.shape)
        ra, dec = wcs.all_pix2world([yc], [xc], 0, ra_dec_order=True)
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

        # Loop through the CD values and replace them with updated values
        for i, row in enumerate(cd):
            for j, cdij in enumerate(row):
                key = 'CD' + '_'.join([str(i+1), str(j+1)])
                wcsHeader[key] = cdij

        # Update the header
        self._BaseImage__header.update(wcsHeader)

        if self.has_uncertainty:
            outUncert = StdDevUncertainty(self.uncertainty)
        else:
            outUncert = None

        # Store WCS in the proper MASTER variable for later retrieval.
        self._BaseImage__fullData = NDDataArray(
            self.data,
            uncertainty=outUncert,
            unit=self.unit,
            wcs=wcs
        )

        return None

    def _get_ticks(self):
        """
        Builds a list of tick properties for plotting utilities using WCS.

        Parameters
        ----------
        None

        Returns
        -------
        RAspacing, DecSpacing : tuple
            The spacing interval for the RA and Dec axes

        RAformatting, DecFormatting : tuple
            The formatting to get nice tick labels for each axis

        RAminorTickFreqs, DecminorTicksFreq : tuple
            The frequency of minor ticks appropriate given the major tick
            spacing for each axis
        """
        # Check if WCS is present
        if not self.has_wcs:
            return (None, None, None)

        # First compute the image dimensions in arcsec
        ny, nx        = np.array(self.shape)
        ps_x, ps_y    = self.pixel_scales
        height, width = (ny*u.pix)*ps_y, (nx*u.pix)*ps_x

        # Setup a range of viable major tick spacing options
        spacingOptions = u.arcsec * np.array([
            0.1,          0.25,        0.5,
            1,            2,           5,             10,
            15,           20,          30,            1*60,
            2*60,         5*60,        10*60,         30*60,
            1*60*60,      2*60*60,     5*60*60
        ])

        # Setup corresponding RA and Dec tick label format strings
        RAformatters = np.array([
            'hh:mm:ss.s', 'hh:mm:ss.s', 'hh:mm:ss.s',
            'hh:mm:ss',   'hh:mm:ss',   'hh:mm:ss',   'hh:mm:ss',
            'hh:mm:ss',   'hh:mm:ss',   'hh:mm:ss',   'hh:mm',
            'hh:mm',      'hh:mm',      'hh:mm',      'hh:mm',
            'hh',         'hh',         'hh'
        ])
        DecFormatters = np.array([
            'dd:mm:ss.s', 'dd:mm:ss.s', 'dd:mm:ss.s',
            'dd:mm:ss',   'dd:mm:ss',   'dd:mm:ss',   'dd:mm:ss',
            'dd:mm:ss',   'dd:mm:ss',   'dd:mm:ss',   'dd:mm',
            'dd:mm',      'dd:mm',      'dd:mm',      'dd:mm',
            'dd',         'dd',         'dd'
        ])

        # Define a set of minor tick frequencies associated with each
        # major tick spacing
        minorTicksFreqs = np.array([
            10,           4,            5,
            10,           4,            5,            10,
            3,            10,           6,            10,
            4,            5,            10,           6,
            10,           4,            5
        ])

        # Figure out which major tick spacing provides the FEWEST ticks
        # but greater than 3
        y_cen, x_cen    = 0.5*ny, 0.5*nx
        RA_cen, Dec_cen = self.wcs.all_pix2world(
            [x_cen],
            [y_cen],
            0,
            ra_dec_order=True
        )

        # Compute which spacing, format, minorTicksFreq to select for RA axis
        hours2degrees = 15.0
        RAcompressionFactor = np.cos(np.deg2rad(Dec_cen))
        compressedRAspacingOptions = hours2degrees*spacingOptions*RAcompressionFactor
        numberOfRAchunks = (width/compressedRAspacingOptions).decompose().value
        atLeast2RAchunks = np.floor(numberOfRAchunks) >= 2
        RAspacingInd  = np.max(np.where(atLeast2RAchunks))

        # Compute which spacing, format, minorTicksFreq to select for Dec axis
        numberOfDecChunks = (height/spacingOptions).decompose().value
        atLeast3DecChunks = np.floor(numberOfDecChunks) >= 3
        DecSpacingInd = np.max(np.where(atLeast3DecChunks))

        # Select the actual RA and Dec spacing
        RAspacing  = hours2degrees*spacingOptions[RAspacingInd]
        DecSpacing = spacingOptions[DecSpacingInd]

        # Select the specific formatting for this tick interval
        RAformatter  = RAformatters[RAspacingInd]
        DecFormatter = DecFormatters[DecSpacingInd]

        # And now select the minor tick frequency
        RAminorTicksFreq  = minorTicksFreqs[RAspacingInd]
        DecMinorTicksFreq = minorTicksFreqs[DecSpacingInd]

        return (
            (RAspacing, DecSpacing),
            (RAformatter, DecFormatter),
            (RAminorTicksFreq, DecMinorTicksFreq)
        )

    def _build_axes(self, axes=None):
        """
        Constructs the appropriate set of axes for this image.

        If a preconstructed axes instance is supplied, then that instance is
        simply returned otherwise a new axis instance is constructed. If the
        object includes a viable WCS, then a WCSaxes instance is constructed,
        otherwise a regular matplotlib.axes.Axes instance is constructed.

        Paramaters
        ----------
        axes : None or axesInstance
            The axes in which te place the image output.

        Returns
        -------
        axes : WCSaxes or matplotlib.axes.Axes
            The Axes instance in which the data will be displayed.
        """
        # TODO: test if this section of code can be executed via "super"
        # If a set of axes was provided, then simply extract the current figure,
        # and return that information to the user
        if axes is not None:
            try:
                # Get the parent figure instance
                fig = axes.figure

                return axes
            except:
                raise TypeError('`axes` must be a `matplotlib.axes.Axes` or `astropy.visualization.wcsaxes.core.WCSAxes` instance.')

        # If there is no WCS, then simply return a regular old set of axes.
        if self._BaseImage__fullData.wcs is None:
            # Build a regular matplotlib Axes instanc.
            fig = plt.figure(figsize = (8,8))
            axes = fig.add_subplot(1,1,1)

            return axes

        # If there is a valid WCS in this image, then build the axes using the
        # WCS for the projection
        fig = plt.figure(figsize = (8,8))
        axes = fig.add_subplot(1, 1, 1, projection=self.wcs)

        # Set the axes linewidth
        axes.coords.frame.set_linewidth(2)

        # Label the axes establish minor ticks.
        RA_ax  = axes.coords[0]
        Dec_ax = axes.coords[1]
        RA_ax.set_axislabel(
            'RA [J2000]',
            fontsize=12,
            fontweight='bold'
        )
        Dec_ax.set_axislabel(
            'Dec [J2000]',
            fontsize=12,
            fontweight='bold',
            minpad=-0.4
        )

        # Retrieve the apropriate spacing and formats
        spacing, formatter, minorTicksFreq = self._get_ticks()

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

        return axes

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

        # Force a redraw of the figure canvas
        fig = self.figure.canvas.draw()

##################################
### END OF SUBCLASSES          ###
##################################
