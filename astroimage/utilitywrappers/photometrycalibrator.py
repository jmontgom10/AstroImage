# This tells Python 2.x to handle imports, division, printing, and unicode the
# way that `future` versions (i.e. Python 3.x) handles those things.
from __future__ import absolute_import, division, print_function, unicode_literals

# Core library imports

# Scipy imports
import numpy as np
from scipy import odr
from skimage import measure, morphology

# Astropy imports
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils import (DAOStarFinder, data_properties,
    CircularAperture, CircularAnnulus, aperture_photometry)
from astroquery.vizier import Vizier

# AstroImage imports
from ..reduced import ReducedScience
from .imagestack import ImageStack
from .photometryanalyzer import PhotometryAnalyzer

class PhotometryCalibrator(object):
    """
    Provides methods to calibrate the images to match catalog photometry

    Images obtained using any of the following filters can be calibrated. To
    calibrate an image, simply instantiate a PhotometricCalibration object using
    a dictionary with the corresponding entries in the `Abbreviation` column as
    the keys and ReducedScience image instances as the values.

    Full Name       Abbreviation    Wavelength      f0
                                    [micron]        [Jy]
    =======================================================
    Johnson-U*      U               0.366           1790.0
    Johnson-B       B               0.438           4063.0
    Johnson-V       V               0.545           3636.0
    Cousins-R**     R               0.641           3064.0
    Cousins-I**     I               0.798           2416.0

    Sloan-g'***     g               0.4770          3631.0
    Sloan-r'***     r               0.6222          3631.0
    Sloan-i'***     i               0.7632          3631.0

    2MASS-J         J               1.235           1594.0
    2MASS-H         H               1.662           1024.0
    2MASS-Ks        Ks              2.159            666.8

    Notes
    =====
    *   Not yet supported
    **  Calibrating magnitudes are computed using Sloan-g, -r, and -i data using
        the transformations provided by Jordi et al. (2006)
        http://adsabs.harvard.edu/abs/2006A%26A...460..339J
    *** The Sloan photometric system is reported in AB magnitudes, which are
        defined to *always* have a zero-point flux of 3631.0 Jy.

    References
    ==========
    Photometric Systems
    -------------------
    Johsnon-Cousins:
    Bessell, Castelli & Plez (1998)
        http://www.adsabs.harvard.edu/abs/1998A&A...333..231B

    Sloan:
    Oke & Gunn (1983)
        http://adsabs.harvard.edu/abs/1983ApJ...266..713O
    Fukugita et al. (1996)
        http://www.adsabs.harvard.edu/abs/1996AJ....111.1748F
    Smith et al. (2002)
        http://www.adsabs.harvard.edu/abs/2002AJ....123.2121S

    2MASS:
    Cohen et al. (2003)
        http://adsabs.harvard.edu/abs/2003AJ....126.1090C

    Reference Star Catalogs
    -----------------------
    APASS: Henden & Munari (2014)
        http://adsabs.harvard.edu/abs/2014CoSka..43..518H
    2MASS: Skrutskie et al. (2006
        http://adsabs.harvard.edu/abs/2006AJ....131.1163S
    """

    # NOTE: I don't think I need these, but in case I want to match to
    # SDSS reported magnitudes (rather than the USNO defined `prime` system)
    #
    # To convert from u'g'r'i'z' to 2.5m ugriz:
    #   u(2.5m) = u' - b25(u)*( (u'-g')-(u'-g')_zp ) + zpOffset25(u)
    #   g(2.5m) = g' - b25(g)*( (g'-r')-(g'-r')_zp ) + zpOffset25(g)
    #   r(2.5m) = r' - b25(r)*( (r'-i')-(r'-i')_zp ) + zpOffset25(r)
    #   i(2.5m) = i' - b25(i)*( (r'-i')-(r'-i')_zp ) + zpOffset25(i)
    #   z(2.5m) = z' - b25(z)*( (i'-z')-(i'-z')_zp ) + zpOffset25(z)
    #
    # where
    #
    #    b25(u) =  0.000
    #    b25(g) = -0.060
    #    b25(r) = -0.035
    #    b25(i) = -0.041
    #    b25(z) =  0.030
    #
    #   (u'-g')_zp = 1.39  |
    #   (g'-r')_zp = 0.53  | these have the same values
    #   (r'-i')_zp = 0.21  |      as in the previous eqn.
    #   (i'-z')_zp = 0.09  |
    #
    # Fortunately, all the zpOffset25's are
    #
    #   zpOffset25(u) = 0.000
    #   zpOffset25(g) = 0.000
    #   zpOffset25(r) = 0.000
    #   zpOffset25(i) = 0.000
    #   zpOffset25(z) = 0.000

    ##################################
    ### START OF CLASS VARIABLES   ###
    ##################################

    # Bands and zero point flux [in Jy = 10^(-26) W /(m^2 Hz)]
    # Following table from Bessll, Castelli & Plez (1998)
    # Passband  Effective wavelength (microns)  Zero point (Jy)
    # U	        0.366                           1790
    # B         0.438                           4063
    # V         0.545                           3636
    # R         0.641                           3064
    # I         0.798                           2416
    # J         1.22                            1589
    # H         1.63                            1021
    # K         2.19                            640

    # Similarly, for the NIR
    # Flux of 0-mag source in 2MASS (Jy)
    # Taken from Cohen et al. (2003) -- http://adsabs.harvard.edu/abs/2003AJ....126.1090C
    zeroFlux = dict(zip(
        np.array([
        'B',    'V',    'R' ,
        'g',     'r',   'i',
        'J',     'H',   'Ks']),
        np.array([
        4063.0, 3636.0, 3064.0,
        3631.0, 3631.0, 3631.0,
        1594.0, 1024.0, 666.8])*u.Jy
    ))

    wavelength = dict(zip(
        np.array([
        'B',    'V',    'R',
        'g',    'r',    'i',
        'J',    'H',    'Ks']),
        np.array([
        0.438,  0.545,  0.641,
        0.4770, 0.6222, 0.7632,
        1.235,  1.662,  2.159])*u.um
    ))

    ##################################
    ### END OF CLASS VARIABLES     ###
    ##################################

    ##################################
    ### START OF STATIC METHODS    ###
    ##################################

    @staticmethod
    def _APASS_Vgr_to_R(apassPhotTable):
        """
        Converts the provided APASS magnitudes to the Johsnon-Cousins system

        Full set of transformations from Jordi et al. (2006)
        (Taken from http://classic.sdss.org/dr7/algorithms/sdssUBVRITransform.html)

        U-B   =     (0.79 ± 0.02)*(u-g)    - (0.93 ± 0.02)
        U-B   =     (0.52 ± 0.06)*(u-g)    + (0.53 ± 0.09)*(g-r) - (0.82 ± 0.04)
        B-g   =     (0.175 ± 0.002)*(u-g)  + (0.150 ± 0.003)
        B-g   =     (0.313 ± 0.003)*(g-r)  + (0.219 ± 0.002)
        V-g   =     (-0.565 ± 0.001)*(g-r) - (0.016 ± 0.001)
        V-I   =     (0.675 ± 0.002)*(g-i)  + (0.364 ± 0.002) if  g-i <= 2.1
        V-I   =     (1.11 ± 0.02)*(g-i)    - (0.52 ± 0.05)   if  g-i >  2.1
        R-r   =     (-0.153 ± 0.003)*(r-i) - (0.117 ± 0.003)
        R-I   =     (0.930 ± 0.005)*(r-i)  + (0.259 ± 0.002)
        I-i   =     (-0.386 ± 0.004)*(i-z) - (0.397 ± 0.001)

        NOTE: Some of the above transformations cannot be reproduced using the
        data in the tabels from Jordi et al. (2006). Thus, the following
        transformation was developed instead.

        R = V + 0.608*(g - r) + 0.844
        """
        # Grab the Sloan r' and i' magnitudes
        V    = apassPhotTable['Vmag']
        e_V  = apassPhotTable['e_Vmag']
        g    = apassPhotTable['g_mag']
        e_g  = apassPhotTable['e_r_mag']
        r    = apassPhotTable['r_mag']
        e_r  = apassPhotTable['e_r_mag']
        gr   = g - r
        e_gr = np.sqrt(e_g**2 + e_r**2)

        # Store the original transformation coefficients from Jordi et al. (2006)
        a7   = +1.646
        e_a7 = +0.008
        b7   = -0.138
        e_b7 = +0.004

        # Compute the Cousns-R magnitude
        R = V + (g - r - b7)/a7

        # Compute the uncertainty in the R-band magnitude
        numerator     = (g - r - b7)
        v_numerator   = (e_gr**2 + e_b7**2)
        v_colorTerm   = ((v_numerator/numerator) + (e_a7/a7)**2)
        e_R           = np.sqrt(e_V**2 + v_colorTerm)

        return R, e_R

    @staticmethod
    def _APASS_gr_to_R(apassPhotTable):
        """
        Computes the photometric transformation from SDSS g', r' to Johnson-R

        Uses the results of the MCMC regression I personally performed using
        *every* APASS-Landolt star match I could find.
        """
        # Setup the regression values and uncertainties for the relation
        # R(Landolt) - r(APASS) = cc*(r(APASS) - g(APASS)) + zp
        cc, zp = -1.112865, -0.143307
        s_cc, s_zp, rhos_ccs_zp = 3.93E-05, 1.79E-05, -1.97E-05

        # Grab the APSS g' and r', values
        g    = apassPhotTable['g_mag']
        e_g  = apassPhotTable['e_r_mag']
        r    = apassPhotTable['r_mag']
        e_r  = apassPhotTable['e_r_mag']
        gr   = g - r
        e_gr = np.sqrt(e_g**2 + e_r**2)

        # Compute the Johnson-R value from these
        R = r + cc*gr + zp

        # Compute the uncertainty in the Johnson-R value
        variance_colorTerm = (s_cc/cc)**2 + (e_gr/gr)**2
        e_R = np.sqrt(
            e_r**2 +
            variance_colorTerm +
            s_zp**2
        )

        return R, e_R

    @classmethod
    def _parse_APASS(cls, apassPhotTable):
        # Initalize an empty table
        outputTable = Table()

        # Grab the RA and Dec values
        outputTable['_RAJ2000'] = apassPhotTable['_RAJ2000']
        outputTable['_DEJ2000'] = apassPhotTable['_DEJ2000']

        # Grab the B-band photometry
        outputTable['B']   = apassPhotTable['Bmag']
        outputTable['e_B'] = apassPhotTable['e_Bmag']

        # Grab the V-band photometry
        outputTable['V']   = apassPhotTable['Vmag']
        outputTable['e_V'] = apassPhotTable['e_Vmag']

        # Compute the R-band photometry
        R, e_R = cls._APASS_Vgr_to_R(apassPhotTable)
        outputTable['R']   = R
        outputTable['e_R'] = e_R

        # Grab the Sloan photometry
        outputTable['g']   = apassPhotTable['g_mag']
        outputTable['e_g'] = apassPhotTable['e_g_mag']
        outputTable['r']   = apassPhotTable['r_mag']
        outputTable['e_r'] = apassPhotTable['e_r_mag']
        outputTable['i']   = apassPhotTable['i_mag']
        outputTable['e_i'] = apassPhotTable['e_i_mag']

        return outputTable

    @staticmethod
    def _parse_2MASS(tmassPhotTable):
        # Initalize an empty table
        outputTable = Table()

        # Only keep the high quality data
        goodRows      = np.where(tmassPhotTable['Qflg'].data.data == b'AAA')
        goodPhotTable = tmassPhotTable[goodRows]

        # Grab the RA and Dec values
        outputTable['_RAJ2000'] = goodPhotTable['_RAJ2000']
        outputTable['_DEJ2000'] = goodPhotTable['_DEJ2000']

        # Grab the J, H, K photometry
        outputTable['J']    = goodPhotTable['Jmag']
        outputTable['e_J']  = goodPhotTable['e_Jmag']
        outputTable['H']    = goodPhotTable['Hmag']
        outputTable['e_H']  = goodPhotTable['e_Hmag']
        outputTable['Ks']   = goodPhotTable['Kmag']
        outputTable['e_Ks'] = goodPhotTable['e_Kmag']

        return outputTable

    @staticmethod
    def _match_catalogs(catalogList):
        """Uses astrometric matching to combine multiple catalogs"""
        if len(catalogList) == 1:
            return catalogList[0]
        else:
            raise NotImplemented('Catalog matching is not yet implemented')

    @staticmethod
    def _repairMagnitudePairDictUncertainties(magnitudePairDict):
        """
        Ensures that no catalog magnitudes have *zero* uncertainty
        """
        # Make a copy to return to the user
        outputMagnitudePairDict = magnitudePairDict.copy()

        # Ensure that there are not a bunch of "zero uncertainty" entries
        for waveband in magnitudePairDict.keys():
            # Grab the uncertainty values for this waveband
            wavebandUncert = magnitudePairDict[waveband]['catalog']['uncertainty']

            # Locate any zero-value uncertainties
            zeroUncert = (wavebandUncert == 0)

            if np.sum(zeroUncert) == len(zeroUncert):
                # If all the uncertainties are zero, then force them all to 0.075
                minRealUncertainty = 0.075
            else:
                # Otherwise, grab the minumum *non-zero* uncertainty
                minRealUncertainty = np.min(
                    wavebandUncert[np.where(np.logical_not(zeroUncert))]
                )

            # Replace the zero-valued uncertainties with a better value
            wavebandUncert[np.where(zeroUncert)] = minRealUncertainty

            # Replace the uncertainty column with its repaired value
            outputMagnitudePairDict[waveband]['catalog']['uncertainty'] = wavebandUncert

        return outputMagnitudePairDict

    ##################################
    ### END OF STATIC METHODS      ###
    ##################################

    def __init__(self, imageDict):
        """
        Constructs a PhotometryCalibrator instance from the provided images

        Parameters
        ----------
        imageDict : dict
            A dictionary containing the waveband names as keys and ReducedScience
            images as the values.
        """
        # Check that the provided images have astrometry
        if not np.all([img.has_wcs for img in imageDict.values()]):
            raise ValueError('All images in `imageDict` must have astrometric solutions')

        # Check that at least *one* waveband was provided
        if len(imageDict) < 1:
            raise ValueError('`imageDict` must contain at least one image')

        # Briefly dissociate the images from the keys for image alignment
        imageKeys = []
        imageList = []
        for key, value in imageDict.items():
            imageKeys.append(key)
            imageList.append(value)

        # Construct an ImageStack instance to test alignment
        imageStack = ImageStack(imageList)
        if not imageStack.aligned:
            print('Performing necessary image alignment')
            imageStack.align_images_with_cross_correlation()

            # Reconstruct the image dictionary
            imageDict1 = dict(zip(imageKeys, imageStack.imageList))
        else:
            imageDict1 = imageDict

        # Construct the necessary star masks for later use
        starMaskList = imageStack._produce_individual_star_masks()
        starMaskDict = dict(zip(imageKeys, starMaskList))

        # Grab the wavelengths available for calibration.
        availableWavebands = self.__class__.zeroFlux.keys()

        # Get wavelength sorted arrays of wavebands and wavelengths
        wavebands   = []
        wavelengths = []
        for waveband in imageDict1.keys():
            # Check if the waveband is in the list of known wavebands
            if waveband not in availableWavebands:
                raise ValueError('{} is not in the list of available wavebands'.format(waveband))

            # If the waveband is available for calibration, then store it
            wavebands.append(waveband)
            wavelengths.append(self.__class__.wavelength[waveband])

        # Convert the waveband and wavelength lists to arrays
        wavebands   = np.array(wavebands)
        wavelengths = np.array([wl.value for wl in wavelengths])*wavelengths[0].unit

        # Sort in order of increasing wavelength
        sortInds    = wavelengths.argsort()
        wavebands   = wavebands[sortInds]
        wavelengths = wavelengths[sortInds]

        # Store the sorted versions of the wavelengths for later use
        self.imageDict    = imageDict1
        self.starMaskDict = starMaskDict
        self.wavebands    = wavebands
        self.wavelengths  = wavelengths

    def _retrieve_catalog_photometry(self):
        """
        Downloads the relevant catalog data for calibration
        """
        # TODO: generalize this to be able to handle unaligned images
        # NOTE: (i.e., determine the footprint of **all** the images)
        # IS THIS *NOT* WHAT I'M ALREADY DOING?

        # Construct a small dictionary to specify which wavebands are available
        # for each catalog
        catalogWavebands = {
            'APASS': ['B', 'V', 'R', 'g', 'r', 'i'],
            '2MASS': ['J', 'H', 'Ks']
        }

        # Compute the image width and height of the image
        maxBt = +91
        maxTp = -91
        maxLf = -1
        maxRt = +361
        for img in self.imageDict.values():
            ny, nx = img.shape
            lfrt, bttp = img.wcs.wcs_pix2world([0, ny], [0, nx], 0)
            lf, rt = lfrt
            bt, tp = bttp

            if bt < maxBt: maxBt = bt
            if tp > maxTp: maxTp = tp
            if lf > maxLf: maxLf = lf
            if rt < maxRt: maxRt = rt

        # Grab the maximum width and the median (RA, Dec)
        RAcen, DecCen = 0.5*(maxLf + maxRt), 0.5*(maxBt + maxTp)
        height = (maxTp - maxBt)*u.deg
        width  = (maxLf - maxRt)*np.cos(np.deg2rad(DecCen))*u.deg

        # TODO: DETERMINE IF OPTICAL BANDS ARE PRESENT --> DOWNLOAD APASS
        # TODO: DETERMINE IF NIR BANDS ARE PRESENT --> DOWNLOAD 2MASS
        catalogList = []
        Vizier.ROW_LIMIT = -1
        for catalog, providedWavebands in catalogWavebands.items():
            # Determine if any of the stored images requrire this catalog
            downloadCatalog = False
            for waveband in self.wavebands:
                if waveband in providedWavebands: downloadCatalog = True

            if downloadCatalog:
                # Download the requested catalog
                catalogs = Vizier.query_region(
                    SkyCoord(
                        ra=RAcen, dec=DecCen,
                        unit=(u.deg, u.deg),
                        frame='fk5'
                    ),
                    width=width,
                    height=height,
                    catalog=catalog
                )

                # Parse the catalog using the appropriate parser
                catalogParser = getattr(self.__class__, '_parse_'+catalog)
                # TODO: use the specific Vizier catalog name rather than
                # simply assuming the 0th element is the correct one.
                parsedCatalog = catalogParser(catalogs[0])

                # Store the parsed catalog in a list for later matching
                catalogList.append(parsedCatalog)

        # Apply the catalog parser to the downloaded catalog and store
        self.catalog = self.__class__._match_catalogs(catalogList)

    def _compute_magnitude_pairs(self, wavebandList):
        """
        Computes aperture corrected magnitudes, paired with catalog entries

        Loops through each element of the waveband list and returns a nested
        dictionary where the keys to the root level are the wavebands provided
        in the `wavebandList` argments

        Parameters
        ----------
        wavebandList : array_like
            The names of the wavebands for which to compute magnitude pairs

        Returns
        -------
        magnitudePairDict : dict
            A dictionary containing the aperture corrected instrumental
            instrumental magnitudes and the corresponding catalog magnitudes
        """
        # Before proceeding to measure photometry, build a complete list of
        # rows with masked data
        maskedRows = False
        for waveband in wavebandList:
            maskedRows = np.logical_or(maskedRows, self.catalog[waveband].mask)

        # Loop through each waveband and measure photometry
        magnitudePairDict = {}
        for waveband in wavebandList:
            # Approx catalog stars in the (only) image!
            thisImg = self.imageDict[waveband]

            # Construct a SkyCoord instance from the star coordinates
            starCoords = SkyCoord(
                self.catalog['_RAJ2000'], self.catalog['_DEJ2000']
            )

            # Locate these stars within the image
            xs, ys = thisImg.get_sources_at_coords(starCoords)

            # Grab the corresponding rows of the photometric catalog
            starCatalog = self.catalog

            # Construct a PhotometryAnalyzer instance for this image
            photAnalyzer = PhotometryAnalyzer(thisImg)

            # Determine the aperture at which the signal-to-noise ratio reaches a
            # maximum for each of these stars.
            starAprs = photAnalyzer.get_maximum_SNR_apertures(xs, ys)

            # Compute the curve of growth parameters for this image
            xCOG, yCOG = photAnalyzer.get_COG_stars()
            kingParams = photAnalyzer.get_curve_of_growth(xCOG, yCOG)

            # TODO: Determine some much better radius to use for aperture photometry
            skyAprIn  = 10*kingParams['Ri']
            skyAprOut = skyAprIn + 2.5

            # Perform basic aperture photometry using the estimate maximum SNR radii
            instrumentalFluxes = []
            fluxUncerts        = []
            for iStar, starInfo in enumerate(zip(xs, ys, starAprs)):
                # Break apart the star information
                xs1, ys1, starApr = starInfo

                # Fill unmatched stars with NaNs and continue
                if not np.isfinite(xs1) or not np.isfinite(ys1):
                    instrumentalFluxes.append(np.NaN)
                    fluxUncerts.append(np.NaN)
                    continue

                # Measure the rudimentary aperture photometry for these stars
                thisFlux, thisFluxUncert = photAnalyzer.aperture_photometry(
                    xs1, ys1, starApr, skyAprIn, skyAprOut
                )

                # Make sure that we only have scalars to deal with
                try:
                    testLen  = len(thisFlux)
                    thisFlux = thisFlux[0]
                except:
                    pass

                try:
                    testLen        = len(thisFluxUncert)
                    thisFluxUncert = thisFluxUncert[0]
                except:
                    pass

                # Store the photometry
                instrumentalFluxes.append(thisFlux)
                fluxUncerts.append(thisFluxUncert)

            # Convert back to numpy arrays
            instrumentalFluxes = np.array(instrumentalFluxes).flatten()
            fluxUncerts        = np.array(fluxUncerts).flatten()

            # Mark any rows with bad data for deletion
            maskedRows = np.logical_or(
                maskedRows,
                np.logical_not(np.isfinite(instrumentalFluxes))
            )
            maskedRows = np.logical_or(
                maskedRows,
                np.logical_not(np.isfinite(fluxUncerts))
            )

            # To avoid warnings, with NaNs, simply set any masked rows to 1e-6
            maskRowInds = np.where(maskedRows)
            instrumentalFluxes[maskRowInds] = 1e-6

            # Convert the instrumental fluxes into instrumental (Pogson) magnitudes
            instrumentalMagnitudes = -2.5*np.log10(instrumentalFluxes)

            # Compute the uncertainty if they exist
            if np.all([fu is not None for fu in fluxUncerts]):
                instrumentalMagUncerts = 2.5*fluxUncerts/(instrumentalFluxes*np.log(10))
            else:
                instrumentalMagUncerts = None

            # Compute the amount of aperture correction for the used starApertures
            # NOTE: Assume that aperture correction introduces zero uncertainty
            apertureCorrections = photAnalyzer.compute_aperture_corrections(
                starAprs, kingParams
            )

            # Apply the aperture correction to the instrumentalMagnitudes
            correctedMagnitudes = instrumentalMagnitudes - apertureCorrections

            # Pair the measured photometry against the catalog photometry
            wavebandMagDict = {
                'catalog': {
                    'data': starCatalog[waveband],
                    'uncertainty': starCatalog['e_' + waveband]
                },
                'instrument': {
                    'data': correctedMagnitudes,
                    'uncertainty': instrumentalMagUncerts
                }
            }

            # Store the data for this waveband in another dictionary
            magnitudePairDict[waveband] = wavebandMagDict

        # Now that *all* the magnitudes have been computed, go through each
        # waveband and eliminate any fluxes which make no logical sense
        goodRowInds = np.where(np.logical_not(maskedRows))
        for waveband in wavebandList:
            # Replace each entry with its own data, sans the masked rows
            wavebandMagDict = {
                'catalog': {
                    'data': magnitudePairDict[waveband]['catalog']['data'][goodRowInds].data.data,
                    'uncertainty': magnitudePairDict[waveband]['catalog']['uncertainty'][goodRowInds].data.data
                },
                'instrument': {
                    'data': magnitudePairDict[waveband]['instrument']['data'][goodRowInds],
                    'uncertainty': magnitudePairDict[waveband]['instrument']['uncertainty'][goodRowInds]
                }
            }

            # Restore this data in the magnitudePairDict
            magnitudePairDict[waveband] = wavebandMagDict

        # Make sure the uncertainties in the magnitudePairDict make sense
        magnitudePairDict = self.__class__._repairMagnitudePairDictUncertainties(magnitudePairDict)

        return magnitudePairDict

    def _singleband_calibration(self):
        """Computes the calibrated image using just a single waveband matching"""
        # Grab the instrumental and catalog magnitudes
        thisWaveband      = self.wavebands[0]
        magnitudePairDict = self._compute_magnitude_pairs(self.wavebands)

        # Compute the differences between instrumental and calibrated magnitudes
        # and propagate the errors.
        magnitudeDifferences = (
            magnitudePairDict[thisWaveband]['catalog']['data'] -
            magnitudePairDict[thisWaveband]['instrument']['data']
        )
        magnitudeDiffUncert = np.sqrt(
            magnitudePairDict[thisWaveband]['catalog']['uncertainty']**2 +
            magnitudePairDict[thisWaveband]['instrument']['uncertainty']**2
        )

        # Compute a single-band zero-point magnitude
        weights      = 1.0/magnitudeDiffUncert**2
        zeroPointMag = np.sum(weights*magnitudeDifferences)/np.sum(weights)

        # Use the zero-point magnitude and zero-point flux to compute a
        # calibrated version of the input image.
        # Start with the zeroPointMag correction factor (CF)
        imageUnits       = self.imageDict[thisWaveband].unit
        pixelSideLengths = self.imageDict[thisWaveband].pixel_scales
        pixelArea        = pixelSideLengths[0]*pixelSideLengths[1]
        pixelArea       *= u.pix**2
        pixelArea        = pixelArea.to(u.arcsec**2)
        zeroPointFlux    = self.__class__.zeroFlux[thisWaveband].to(u.uJy)
        zeroPointMagCF   = (10**(-0.4*zeroPointMag))
        zeroPointMagCF  /= imageUnits
        zeroPointMagCF  /= pixelArea

        # The final calibrated image is equal to...
        # flux_cal = (f0 * zeroPointMagCF) * f_inst
        calIntenImg = (
            self.imageDict[thisWaveband] *
            (zeroPointFlux * zeroPointMagCF)
        )

        # Store the image in a dictionary and return to user
        calImgDict = {
            thisWaveband: calIntenImg
        }

        return calImgDict

    def _multiband_calibration(self):
        """Computes the calibrated images using multiple wavebands and color"""
        def linearFunc(B, x):
            '''Linear function y = m*x + b'''
            # B is a vector of the parameters.
            # x is an array of the current x values.
            # x is in the same format as the x passed to Data or RealData.
            #
            # Return an array in the same format as y passed to Data or RealData.
            return B[0]*x + B[1]

        # Create an odr.Model instance from the linear function
        linearModel = odr.Model(linearFunc)

        # Loop through each waveband and compute the photometry for *that* image
        # and the subsequent (in wavelength) image. Then, compute the color
        # image based on the aperture photometry from the two images
        for iWave, waveband in enumerate(self.wavebands[0:-1]):
            # Grab magnitude pairs to do 2-band photometry
            magnitudePairDict = self._compute_magnitude_pairs(
                self.wavebands[iWave:iWave+2]
            )

            # Grab the wavebands for convenience
            waveband1 = waveband
            waveband2 = self.wavebands[iWave+1]

            ####################################################################
            # Waveband #1
            ####################################################################
            magnitudeDifference1 = (
                magnitudePairDict[waveband1]['catalog']['data'] -
                magnitudePairDict[waveband1]['instrument']['data']
            )
            magnitudeDiffUncert1 = np.sqrt(
                magnitudePairDict[waveband1]['catalog']['uncertainty']**2 +
                magnitudePairDict[waveband1]['instrument']['uncertainty']**2
            )
            instrumentalColor    = (
                magnitudePairDict[waveband1]['instrument']['data'] -
                magnitudePairDict[waveband2]['instrument']['data']
            )
            instrumentalColorUncert = np.sqrt(
                magnitudePairDict[waveband1]['instrument']['uncertainty']**2 +
                magnitudePairDict[waveband2]['instrument']['uncertainty']**2
            )

            # Perform an ODR regression (assume there are no outliers!)
            # Build an instance to store the data
            wavebandData1 = odr.RealData(
                instrumentalColor, magnitudeDifference1,
                sx=instrumentalColorUncert, sy=magnitudeDiffUncert1
            )

            # Build a regression object
            wavebandODR1 = odr.ODR(wavebandData1, linearModel, beta0=[0.0, 23.0])

            # Run the regression
            wavebandRegression1 = wavebandODR1.run()

            # Compute the calibrated waveband1 image
            # Start with the zeroPointMag correction factor (CF)
            imageUnits1       = self.imageDict[waveband1].unit
            pixelSideLengths1 = self.imageDict[waveband1].pixel_scales
            pixelArea1        = pixelSideLengths1[0]*pixelSideLengths1[1]
            pixelArea1       *= u.pix**2
            pixelArea1        = pixelArea1.to(u.arcsec**2)
            zeroPointFlux1    = self.__class__.zeroFlux[waveband1].to(u.uJy)
            zeroPointMagCF1   = (10**(-0.4*wavebandRegression1.beta[1]))
            zeroPointMagCF1  /= imageUnits1
            zeroPointMagCF1  /= pixelArea1

            # Compute linear-space color correction factor and fluxRatio image
            colorCF1      = 10**wavebandRegression1.beta[0]

            # Use a simple flux ratio
            fluxRatio = self.imageDict[waveband1]/self.imageDict[waveband2]
            goodSNR = (np.abs(fluxRatio.snr) > 3.0)

            # Construct the star mask for this pair
            starMask = np.logical_or(
                self.starMaskDict[waveband1],
                self.starMaskDict[waveband2]
            )

            # Don't consider high SNR pixels *inside* stars, as the principal
            # interest in the calibration of *intensity* in extended emission
            # objects such as galaxies or nebulae
            goodSNR = np.logical_and(
                goodSNR,
                np.logical_not(starMask)
            )

            # Locate the regions with consistently good SNR
            all_labels  = measure.label(goodSNR)
            all_labels1 = morphology.remove_small_objects(
                all_labels, min_size=0.001*all_labels.size
            )

            # Grab the central region of the image
            ny, nx = all_labels1.shape
            cy, cx = np.int(ny//2), np.int(nx//2)
            lf, rt = cx - np.int(0.1*nx), cx + np.int(0.1*nx)
            bt, tp = cy - np.int(0.1*ny), cy + np.int(0.1*nx)
            centerRegion = all_labels1[bt:tp, lf:rt]
            label_hist, labels = np.histogram(centerRegion, np.unique(centerRegion))
            label_mode  = labels[label_hist.argmax()]
            goodSNR     = (all_labels1 == label_mode)

            # Smooth out the boundaries of the "goodSNR" region
            fluxRatio1 = np.median(fluxRatio.data[goodSNR])

            # The final calibrated image is equal to...
            # flux_cal = (f0 * zeroPointMagCF) * f_inst * (colorCF * fluxRatio)
            calIntenImg1 = (
                self.imageDict[waveband1] *
                (zeroPointFlux1 * zeroPointMagCF1) *
                (colorCF1 * fluxRatio1)
            )

            ####################################################################
            # Waveband #1
            ####################################################################
            magnitudeDifference2 = (
                magnitudePairDict[waveband2]['catalog']['data'] -
                magnitudePairDict[waveband2]['instrument']['data']
            )
            magnitudeDiffUncert2 = np.sqrt(
                magnitudePairDict[waveband2]['catalog']['uncertainty']**2 +
                magnitudePairDict[waveband2]['instrument']['uncertainty']**2
            )

            # Perform an ODR regression (assume there are no outliers!)
            # Build an instance to store the data
            wavebandData2 = odr.RealData(
                instrumentalColor, magnitudeDifference2,
                sx=instrumentalColorUncert, sy=magnitudeDiffUncert2
            )

            # Build a regression object
            wavebandODR2 = odr.ODR(wavebandData2, linearModel, beta0=[0.0, 23.0])

            # Run the regression
            wavebandRegression2 = wavebandODR2.run()

            # Compute the calibrated waveband2 image
            # Start with the zeroPointMag correction factor (CF)
            imageUnits2       = self.imageDict[waveband2].unit
            pixelSideLengths2 = self.imageDict[waveband2].pixel_scales
            pixelArea2        = pixelSideLengths2[0]*pixelSideLengths2[1]
            pixelArea2       *= u.pix**2
            pixelArea2        = pixelArea2.to(u.arcsec**2)
            zeroPointFlux2    = self.__class__.zeroFlux[waveband2].to(u.uJy)
            zeroPointMagCF2   = (10**(-0.4*wavebandRegression2.beta[1]))
            zeroPointMagCF2  /= imageUnits2
            zeroPointMagCF2  /= pixelArea2

            # Compute linear-space color correction factor
            colorCF2  = 10**wavebandRegression2.beta[0]

            # The final calibrated image is equal to...
            # flux_cal = (f0 * zeroPointMagCF) * f_inst * (colorCF * fluxRatio)
            calIntenImg2 = (
                self.imageDict[waveband2] *
                (zeroPointFlux2 * zeroPointMagCF2) *
                (colorCF2 * fluxRatio1)
            )

            ####################################################################
            # (Waveband #1 - Waveband #2) color
            ####################################################################
            catalogColor = (
                magnitudePairDict[waveband1]['catalog']['data'] -
                magnitudePairDict[waveband2]['catalog']['data']
            )
            catalogColorUncert = np.sqrt(
                magnitudePairDict[waveband1]['catalog']['uncertainty']**2 +
                magnitudePairDict[waveband2]['catalog']['uncertainty']**2
            )

            # Perform an ODR regression (assume there are no outliers!)
            # Build an instance to store the data
            colorData = odr.RealData(
                instrumentalColor, catalogColor,
                sx=instrumentalColorUncert, sy=catalogColorUncert
            )

            # Build a regression object
            colorODR = odr.ODR(colorData, linearModel, beta0=[0.0, 23.0])

            # Run the regression
            colorRegression = colorODR.run()

            # Compute the instrumental color image
            instColor = -2.5*np.log10(fluxRatio)

            # Compute the calibrated color image from the regression coefficients
            calColorImg = colorRegression.beta[1] + colorRegression.beta[0]*instColor

            # Build an image to store these calibrated images
            colorKey     = waveband1 + waveband2
            waveband1Key = '_'.join([waveband1, colorKey])
            waveband2Key = '_'.join([waveband2, colorKey])
            calImgDict = {
                waveband1Key: calIntenImg1,
                waveband2Key: calIntenImg2,
                colorKey: calColorImg
            }

            return calImgDict

    def calibrate_photometry(self):
        """
        Matches image photometry to the catalog photometry entries
        """
        # Start by downloading and parsing the requested catalog
        self._retrieve_catalog_photometry()

        # If there is more than band to work with, then do multi-band calibration
        if self.wavelengths.size > 1:
            calibratedImgDict = self._multiband_calibration()
        # If there is only one band to work with, then do single-band calibration
        else:
            calibratedImgDict = self._singleband_calibration()

        return calibratedImgDict
