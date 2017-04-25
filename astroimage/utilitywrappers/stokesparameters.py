# Core imports
import copy

# Scipy imports
import numpy as np

# Astropy imports
from astropy.table import Table
from astropy.coordinates import SkyCoord, FK5
import astropy.units as u

# AstroImage imports
from ..baseimage import ClassProperty
from ..reduced import ReducedScience
from .inpainter import Inpainter
from .astrometrysolver import AstrometrySolver

# Define which functions, classes, objects, etc... will be imported via the command
# >>> from stokesparameters import *
__all__ = ['StokesParameters']

class StokesParameters(object):
    """
    A container for holding Stokes Parameter images.

    Takes in a set of images to either construct the Stokes parameters or a set
    of prepared Stokes parameters.

    If a set of polaroid (or HWP) rotation angle images are provided, they must
    be in the following (key, value) pairs:

    'I_0'   : astroimage.ReducedScience
    'I_45'  : astroimage.ReducedScience
    'I_90'  : astroimage.ReducedScience
    'I_135' : astroimage.ReducedScience

    If a set of Stokes parameters are provided, they must be in the the
    following (key, value) pairs:

    'I' : astroimage.ReducedScience
    'Q' : astroimage.ReducedScience
    'U' : astroimage.ReducedScience
    'V' : astroimage.ReducedScience, optional, default: None

    Provides a method to convert the Stokes parameters into images of
    polarization percentage and positon angle via several predefined estimators.
    """

    # Initalize polarization calibration constant variable. The values here are
    # those for an ideal polarimeter with perfect efficiency and perfect
    # equatorial alignment.
    __polarimeterCalibrationConstants = {
        'PE': 1.0,
        's_PE': 0.0,
        'PAsign': +1,
        'D_PA': u.Quantity(0.0, u.degree),
        's_D_PA': u.Quantity(0.0, u.degree)
    }

    @ClassProperty
    @classmethod
    def baseClass(cls):
        """Quick reference to the BaseImage class"""
        thisClass_methodResolutionOrder = cls.mro()
        baseClass = thisClass_methodResolutionOrder[-2]
        return baseClass

    # TODO: decide whether or not it's a good idea to use this method.
    @classmethod
    def read_calibration_constants(cls, filename, **kwargs):
        """
        Reads the polarimeter constants from disk and stores.

        The file at `filename` must be readable by the astropy.table.Table.read
        method. Any keyword arguments provided to this method will be passed to
        the astropy.table.Table.read method.
        """
        pass

    @classmethod
    def set_calibration_constants(cls, constantDict):
        """
        Sets the calibration constants according to the dictionary provided.

        The dictionary accepts the following (key, value) pairs.

        'PE' : int or float
            The polarimetric efficiency of the polarimeter.

        's_PE' : int or float
            The uncertainty in the 'PE' value.

        'PAsign' : int (+1 or -1)
            The direction of the correlation between increasing polaroid
            rotation angle and increasing polarization position angle.

        'D_PA' : astropy.units.quantity.Quantity
            The offset between the instrument rotation angle frame and the
            equatorial coordinate frame. This quantity must have angular units
            specified.

        's_D_PA' : astropy.units.quantity.Quantity
            The uncertainty in the 'D_PA' value. This must also be have angular
            units specified.
        """
        # Initalize a dictionary to store the finalized values
        storageDict = {}

        # Test if the PE constant wast provided
        if 'PE' in constantDict:
            storageDict['PE'] = constantDict['PE']
        else:
            storageDict['PE'] = 1.0

        # Test if the s_PE constant was provided
        if 's_PE' in constantDict:
            storageDict['s_PE'] = constantDict['s_PE']
        else:
            storageDict['s_PE'] = 0.0

        # Test if the PAsign constant was provided
        if 'PAsign' in constantDict:
            # Test if PAsign is an integer value
            try:
                assert float(constantDict['PAsign']).is_integer()
            except:
                raise TypeError('`PAsign` value must be an integer or a float')

            # Test if PAsign is +1 or -1
            try:
                assert np.abs(constantDict['PAsign']) == 1
            except:
                raise ValueError('`PAsign` value must be +1 or -1')

            # Store the value if it passes the tests
            storageDict['PAsign'] = constantDict['PAsign']
        else:
            storageDict['PAsign'] = +1

        # Test if the D_PA constant was provided
        if 'D_PA' is constantDict:
            # Test if provided value has angular units
            if not isinstance(constantDict['D_PA'], u.Quantity):
                raise TypeError('`D_PA` value must be an astropy.units.quantity.Quantity instance')
            if not constantDict['D_PA'].unit.is_equivalent(u.rad):
                raise TypeError('`D_PA` value must be an angular quantity')

            storageDict['D_PA'] = constantDict['D_PA']

        else:
            storageDict['D_PA'] = u.Quantity(0.0, u.degree)

        # Test if the s_D_PA constant was provided
        if 's_D_PA' is constantDict:
            # Test if provided value has angular units
            if not isinstance(constantDict['s_D_PA'], u.Quantity):
                raise TypeError('`s_D_PA` value must be an astropy.units.quantity.Quantity instance')
            if not constantDict['s_D_PA'].unit.is_equivalent(u.rad):
                raise TypeError('`s_D_PA` value must be an angular quantity')

            storageDict['s_D_PA'] = constantDict['s_D_PA'].to(constantDict['D_PA'].unit)

        else:
            storageDict['s_D_PA'] = u.Quantity(0.0, u.degree)

        # Now store the dictionary in the class variable
        cls.__polarimeterCalibrationConstants = storageDict

    @ClassProperty
    @classmethod
    def PE(cls):
        """The polarimetric efficiency calibration constant"""
        return cls.baseClass.__polarimeterCalibrationConstants['PE']

    @ClassProperty
    @classmethod
    def s_PE(cls):
        """The uncertainty in the polarimetric efficiency calibration constant"""
        return cls.baseClass.__polarimeterCalibrationConstants['PE']

    @ClassProperty
    @classmethod
    def PAsign(cls):
        """The PAsign calibration constant"""
        return cls.baseClass.__polarimeterCalibrationConstants['PAsign']

    @ClassProperty
    @classmethod
    def D_PA(cls):
        """The PA offset calibration constant"""
        return cls.baseClass.__polarimeterCalibrationConstants['D_PA']

    @ClassProperty
    @classmethod
    def s_D_PA(cls):
        """The uncertainty in the PA offset calibration constant"""
        return cls.baseClass.__polarimeterCalibrationConstants['s_D_PA']

    def __init__(self, imageSet):
        """
        Constructs a StokesParameters instance.

        Takes in a set of images to either construct the Stokes parameters or
        a set of prepared Stokes parameters.

        If a set of polaroid (or HWP) rotation angle images are provided, they
        must be in the following (key, value) pairs:

        'I_0'   : astroimage.ReducedScience
        'I_45'  : astroimage.ReducedScience
        'I_90'  : astroimage.ReducedScience
        'I_135' : astroimage.ReducedScience

        If a set of Stokes parameters are provided, they must be in the
        the following (key, value) pairs:

        'I' : astroimage.ReducedScience
        'Q' : astroimage.ReducedScience
        'U' : astroimage.ReducedScience
        'V' : astroimage.ReducedScience, optional, default: None

        Parameters
        ----------
        imageSet : dict
            A dictionary containing an aligned set of images required to
            construct Stokes Parameters.

        Returns
        -------
        stokesParameters : astroimage.StokesParameters
            An instance containing all available Stokes Parameters, which are
            accessible via instance properties.
        """
        # Test if all the values provided are ReducedScience objects
        allImagesAreReduced = all(
            [isinstance(img, ReducedScience) for img in imageSet.values()]
        )
        if not allImagesAreReduced:
            raise TypeError('All images must be `ReducedScience` instances')

        # Test if all the images have the same shape
        imageShapes = [img.shape for img in imageSet.values()]
        imagesHaveSameShape = all([s == imageShapes[0] for s in imageShapes[1:]])
        if not imagesHaveSameShape:
            raise ValueError('All images must have the same shape')

        # Test if all the images have astrometry for alignment test
        imagesHaveAstrometry = all([img.has_wcs for img in imageSet.values()])
        if not imagesHaveAstrometry:
            raise ValueError('All images must have astrometric solutions to check for alignment')

        # Test if the provided images are aligned
        imagesAreAligned = self._test_alignment(imageSet)
        if not imagesAreAligned:
            raise ValueError('All images must be aligned')

        # Test if a set of IPPA images were provided
        ippaKeys = ['I_0', 'I_45', 'I_90', 'I_135']
        ippaImagesProvided = all([key in imageSet for key in ippaKeys])

        if ippaImagesProvided:
            self.__ippaImages = imageSet
        else:
            self.__ippaImages = None

        # Test if a set of Stokes Images were provided
        stokesKeys = ['I', 'Q', 'U']
        stokesImagesProvided = all([key in imageSet for key in stokesKeys])

        # Now store the stokesImages in a hidden attribute
        if stokesImagesProvided:
            # Double check if a Stokes V was provided, and if not, then add 'None'
            if 'V' not in imageSet:
                imageSet['V'] = 0

            self.__stokesImages = imageSet
        else:
            stokesKeys.append('V')
            self.__stokesImages = None

    @staticmethod
    def _test_alignment(imageSet):
        """Tests if the provided image set is astrometrically aligned"""
        # Grab the pixel coordinates of the middle of the image
        imageList = list(imageSet.values())
        cy, cx    = 0.5*np.array(imageList[0].shape)

        # Convert the pixel coordinates to celestial coordinates
        centerCoords = [img.wcs.all_pix2world(cx, cy, 0, ra_dec_order=True) for img in imageList]
        centerCoords = np.array(centerCoords)
        centerCoords = SkyCoord(
            ra=centerCoords[:,0],
            dec=centerCoords[:,1],
            unit=u.degree,
            frame=FK5
        )

        # Compute the offsets between the images
        imageOffsets = [centerCoords[0].separation(c) for c in centerCoords]

        # Test if these are all within ~2 pixel alignment
        pixScale = imageList[0].pixel_scales
        pixScale = np.sqrt(pixScale.value.prod())*pixScale.unit
        offsetTolerance = (1.5*u.pix)*pixScale
        alignmentTestResult = all(
            [imgOff < offsetTolerance for imgOff in imageOffsets]
        )

        return alignmentTestResult

    ##################################
    ### START OF PROPERTIES        ###
    ##################################

    @property
    def I(self):
        return self._StokesParameters__stokesImages['I']

    @property
    def Q(self):
        return self._StokesParameters__stokesImages['Q']

    @property
    def U(self):
        return self._StokesParameters__stokesImages['U']

    @property
    def V(self):
        return self._StokesParameters__stokesImages['V']

    ##################################
    ### END OF PROPERTIES          ###
    ##################################

    ##################################
    ### START OF IMAGE METHODS     ###
    ##################################

    def crop(self, x1, x2, y1, y2):
        # TODO use the self.wcs.wcs.sub() method to recompute the right wcs
        # for a cropped image.
        """
        Crops the images in this instance to the specified pixel locations.

        Parameters
        ----------
        x1, x2, y1, y2: int
            The pixel locations for the edges of the cropped IPPA and Stokes
            images.

        Returns
        -------
        outImg: `ReducedScience`
            The cropped StokesParameters instance
        """
        # Make a copy of the instance for storing the output values
        outStokes = copy.deepcopy(self)

        # Attempt to apply the rebinning to each image
        try:
            if self._StokesParameters__ippaImages is not None:
                rebinnedI_0 = self._StokesParameters__ippaImages['I_0'].crop(x1, x2, y1, y2)
                rebinnedI_45 = self._StokesParameters__ippaImages['I_45'].crop(x1, x2, y1, y2)
                rebinnedI_90 = self._StokesParameters__ippaImages['I_90'].crop(x1, x2, y1, y2)
                rebinnedI_135 = self._StokesParameters__ippaImages['I_135'].crop(x1, x2, y1, y2)

                # Construct the IPPA image dictionary
                keyList  = ['I_0', 'I_45', 'I_90', 'I_135']
                imgList  = [rebinnedI_0, rebinned_I45, rebinnedI_90, rebinnedI_135]
                ippaDict = dict(zip(keyList, imgList))

                # Store the IPPA image dictionary in the output instance
                outStokes._StokesParameters__ippaImages   = ippaDict


            rebinnedI = self._StokesParameters__stokesImages['I'].crop(x1, x2, y1, y2)
            rebinnedQ = self._StokesParameters__stokesImages['Q'].crop(x1, x2, y1, y2)
            rebinnedU = self._StokesParameters__stokesImages['U'].crop(x1, x2, y1, y2)

            # Construct the Stokes parameters image dictionary
            keyList    = ['I', 'Q', 'U', 'V']
            imgList    = [rebinnedI, rebinnedQ, rebinnedU, None]
            stokesDict = dict(zip(keyList, imgList))

            # Store the Stokes parameters image dictionary in the output
            outStokes._StokesParameters__stokesImages = stokesDict

        except:
            raise

        return outStokes

    def rebin(self, nx, ny):
        """
        Rebins the images in this instance to have a specified shape.

        Parameters
        ----------
        nx, ny: int
            The target number of pixels along the horizontal (nx) and vertical
            (ny) axes for the IPPA and Stokes images.

        Returns
        -------
        outStokes : astroimage.stokesparameters.StokesParameters
            The rebinned StokesParameters instance
        """
        # Make a copy of the instance for storing the output values
        outStokes = copy.deepcopy(self)

        # Attempt to apply the rebinning to each image
        try:
            if self._StokesParameters__ippaImages is not None:
                rebinnedI_0 = self._StokesParameters__ippaImages['I_0'].rebin(nx, ny)
                rebinnedI_45 = self._StokesParameters__ippaImages['I_45'].rebin(nx, ny)
                rebinnedI_90 = self._StokesParameters__ippaImages['I_90'].rebin(nx, ny)
                rebinnedI_135 = self._StokesParameters__ippaImages['I_135'].rebin(nx, ny)

                # Construct the IPPA image dictionary
                keyList  = ['I_0', 'I_45', 'I_90', 'I_135']
                imgList  = [rebinnedI_0, rebinned_I45, rebinnedI_90, rebinnedI_135]
                ippaDict = dict(zip(keyList, imgList))

                # Store the IPPA image dictionary in the output instance
                outStokes._StokesParameters__ippaImages   = ippaDict


            rebinnedI = self._StokesParameters__stokesImages['I'].rebin(nx, ny)
            rebinnedQ = self._StokesParameters__stokesImages['Q'].rebin(nx, ny)
            rebinnedU = self._StokesParameters__stokesImages['U'].rebin(nx, ny)

            # Construct the Stokes parameters image dictionary
            keyList    = ['I', 'Q', 'U', 'V']
            imgList    = [rebinnedI, rebinnedQ, rebinnedU, None]
            stokesDict = dict(zip(keyList, imgList))

            # Store the Stokes parameters image dictionary in the output
            outStokes._StokesParameters__stokesImages = stokesDict

        except:
            raise

        return outStokes

    ##################################
    ### END OF IMAGE METHODS       ###
    ##################################

    ##################################
    ### START OF COMPUTING METHODS ###
    ##################################

    def compute_stokes_parameters(self):
        """
        Converts the provided IPPA images to Stokes parameter images

        The IPPA images required for this computation should have been provided
        during instantiation. The resulting Stokes parameter images will be
        stored in an instance attribute, and the parameters will be accessible
        via the instance 'I', 'Q', 'U',  and 'V' properties.

        The computed Stokes U and Q parameters are scaled up by the provided
        polarimetric efficiency (PE) and rotated by the offset (D_PA) between
        the polarimeter frame and the equatorial frame. If the polarimeter
        calibration constants were not set using the `set_calibration_constants`
        class method, then an ideal polarimeter with perfect efficiency and
        perfect equatorial alignment is assumed.

        The uncertainty of the Stokes U and Q parameters is set to be the
        average of the independently computed uncertainty in each parameter.
        This means that Stokes U and Q have the same uncertainty in each pixel.
        This is the canonical approach and is justified by the fact that the
        distributions of the independent uncertainties have similar means and
        dispersions. Thus, a single estimate of the uncertainty is used.

        This has the fortunate side effect of making the uncertainty invariant
        under a coordinate frame rotation. So the frame rotation applied by
        the D_PA calibration constant has no effect on the uncertainty in Stokes
        U and Q parameters.
        """

        # Test if the IPPA images are present
        if self._StokesParameters__ippaImages is None:
            raise ValueError('No IPPA images were found')

        # Start by inpainting any IPPA images
        for key, img in self._StokesParameters__ippaImages.items():
            # Construct an Inpainter instance
            inpainter = Inpainter(img)

            # Inpaint NaNs
            inpaintedImg = inpainter.inpaint_nans()

            # Restore the image to the dictionary
            self._StokesParameters__ippaImages[key] = inpaintedImg

        #**********************************************************************
        # Stokes I
        #**********************************************************************
        # Average the images to get Stokes I
        stokesI = 0.5*(
            self._StokesParameters__ippaImages['I_0']  +
            self._StokesParameters__ippaImages['I_45'] +
            self._StokesParameters__ippaImages['I_90'] +
            self._StokesParameters__ippaImages['I_135']
        )

        # Trigger a re-solving of the image astrometry. Start by clearing data
        # from the header.
        stokesI.clear_astrometry()
        tmpHeader = stokesI.header
        del tmpHeader['POLPOS']
        stokesI.header = tmpHeader

        # Create an AstrometrySolver instance and run it.
        print('Solving astrometry for Stokes I image')
        astroSolver = AstrometrySolver(stokesI)
        stokesI, success = astroSolver.run(clobber=True)

        # Check if astrometric solution succeeded
        if not success:
            raise RuntimeError('Failed to solve astrometry of Stokes I image.')

        #**********************************************************************
        # Stokes Q
        #**********************************************************************
        # Subtract the I_0 and I_ 90images to get Stokes Q
        A = (
            self._StokesParameters__ippaImages['I_0'] -
            self._StokesParameters__ippaImages['I_90']
        )
        B = (
            self._StokesParameters__ippaImages['I_0'] +
            self._StokesParameters__ippaImages['I_90']
        )

        # Divide the difference images
        stokesQ = A/B

        #**********************************************************************
        # Stokes U
        #**********************************************************************
        # Subtact the I_45 and I_135 images to get Stokes Q
        A = (
            self._StokesParameters__ippaImages['I_45'] -
            self._StokesParameters__ippaImages['I_135']
        )
        B = (
            self._StokesParameters__ippaImages['I_45'] +
            self._StokesParameters__ippaImages['I_135']
        )

        # Divide the difference images
        stokesU = A/B

        #**********************************************************************
        # Apply calibration constants
        #**********************************************************************
        # Divide by the polarimetric uncertainty. This is not an elegant
        # solution, but the easiest way to do that is to create a ReducedScience
        # instance containing the value and its uncertainty.
        PE_image = ReducedScience(
            np.array([[self.PE]]),
            uncertainty=np.array([[self.s_PE]])
        )

        D_PA_image = ReducedScience(
            np.array([[self.D_PA.value]]),
            uncertainty=np.array([[self.s_D_PA.value]]),
            properties={'unit':self.D_PA.unit}
        )

        # Divide the Stokes Q and U images by the polarimetric efficiency, and
        # correct for any PAsign inversion.
        stokesQcor1 = 1.0         * stokesQ/PE_image
        stokesUcor1 = self.PAsign * stokesU/PE_image

        # Pause to compute the uncertainty is Stokes U and Q before rotating.
        uncertInStokesQU = 0.5*(stokesQcor1.uncertainty + stokesUcor1.uncertainty)

        # # Propagate any uncertainty in D_PA into the uncertInStokesQU variable
        #
        # TODO: Think through a better way to incorporate uncertainty in D_PA
        #
        # uncertInStokesQU = np.sqrt(
        #     uncertInStokesQU**2 +
        #     self.s_D_PA**2
        # )

        # Rotate the Q and U images by the PA offset
        stokesQcor2 = (
            np.cos(2*self.D_PA).value*stokesQcor1 -
            np.sin(2*self.D_PA).value*stokesUcor1
        )
        stokesUcor2 = (
            np.sin(2*self.D_PA).value*stokesQcor1 +
            np.cos(2*self.D_PA).value*stokesUcor1
        )

        # Construct a new image force to share the stokesI header
        stokesQ = ReducedScience(
            stokesQcor2.data,
            uncertainty=uncertInStokesQU,
            header=stokesI.header,
            properties={'unit': u.dimensionless_unscaled}
        )

        # Construct a new image force to share the stokesI header
        stokesU = ReducedScience(
            stokesUcor2.data,
            uncertainty=uncertInStokesQU,
            header=stokesI.header,
            properties={'unit': u.dimensionless_unscaled}
        )

        # Store the results in a dictionary and store the dictionary in instance
        stokesKeysList   = ['I', 'Q', 'U', 'V']
        stokesImagesList = [stokesI, stokesQ, stokesU, None]
        stokesImagesDict = dict(zip(stokesKeysList, stokesImagesList))
        self._StokesParameters__stokesImages = stokesImagesDict

    def _compute_polarization_percentage(self, P_estimator='NAIVE',
        minimum_SNR=1.0):
        """
        Computes the polarization image using the specified estimator.
        """
        # TODO: Modularize this up the wazoooo!
        if P_estimator.upper() == 'NAIVE':
            #####
            # Compute a raw estimation of the polarization map
            #####
            P_image  = 100*np.sqrt(self.Q**2 + self.U**2)

            # The sigma which determines the Rice distribution properties is the
            # width of the Stokes Parameter distribution, so we will simply
            # compute an average uncertainty and assign it to the P_image.
            P_image.uncertainty = 100*0.5*(self.Q.uncertainty + self.U.uncertainty)

        elif P_estimator.upper() == 'WARDLE_KRONBERG':
            #####
            # Handle the ricean correction using Wardle and Kronberg (1974)
            #####
            # Quickly build the P map
            P_image = np.sqrt(self.Q.data**2 + self.U.data**2)

            # Apply the bias correction. The sigma which determines the Rice
            # distribution properties is the width of the Stokes Parameter
            # distribution, so we will simply compute an average uncertainty and
            # assign it to the P_image.
            smap = 0.5*(self.Q.uncertainty + self.U.uncertainty)

            import matplotlib.pyplot as plt
            plt.ion()
            plt.imshow(P_image.data/smap, vmin=0, vmax=0.01)

            # WHY ISN'T THIS SHOWING ANYTHING?
            import pdb; pdb.set_trace()

            # This is the old correction we were using before I reread the
            # Wardle and Kronberg paper... it is even MORE aggresive than the
            # original recomendation

            # Catch any stupid values (nan, inf, etc...)
            badVals = np.logical_not(np.logical_and(
                np.isfinite(P_image.data),
                np.isfinite(smap)))

            # Replace the stupid values with zeros
            if np.sum(badVals) > 0:
                badInds               = np.where(badVals)
                P_image.data[badInds] = 0.0
                smap[badInds]         = 1.0

            # Check which measurements don't match the minimum SNR
            zeroVals = P_image/smap <= minimum_SNR
            numZero  = np.sum(zeroVals.astype(int))
            if numZero > 0:
                # Make sure the square-root does not produce NaNs
                zeroInds               = np.where(zeroVals)
                P_image.data[zeroInds] = 2*smap[zeroInds]

            # Compute the "debiased" polarization map
            tmpP =  P_image.data*np.sqrt(1.0 - (smap/P_image.data)**2)

            if numZero > 0:
                # Set all insignificant detections to zero
                tmpP[zeroInds] = 0.0

            # Now we can safely take the sqare root of the debiased values
            P_image             = np.sqrt(self.Q**2 + self.U**2)
            P_image.data        = 100*tmpP
            P_image.uncertainty = 100*smap

        elif P_estimator.upper() == 'MAIER_PIECEWISE':
            # The following is taken from Maier et al. (2014) (s = sigma_q ==
            # sigma_u) and the quantity of interest is the SNR value  p = P/s

            # Compute the raw polarization and uncertainty
            P_image = np.sqrt(self.Q.data**2 + self.U.data**2)
            smap    = 0.5*(self.Q.uncertainty + self.U.uncertainty)

            # Compute the SNR map (called "p" in most papers)
            pmap = P_image/smap

            # Find those values which don't meet the minimum_SNR requirement
            zeroInds = np.where(P_image/smap <= minimum_SNR)
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
            P_image             = np.sqrt(self.Q**2 + self.U**2)
            P_image.data        = 100*p1map*smap
            P_image.uncertainty = 100*smap

        elif P_estimator.upper() == 'MODIFIED_ASYMPTOTIC':
            # The following is taken from Plaszczynski et al. (2015). This makes
            # the additional assumption/simplicication that (s = sigma_q ==
            # sigma_u).

            # Compute the raw polarization and uncertainty
            P_image = np.sqrt(self.Q.data**2 + self.U.data**2)
            smap    = 0.5*(self.Q.uncertainty + self.U.uncertainty)

            # Apply the asymptotic debiasing effect.
            P1map = P_image - (smap**2 * ((1 - np.exp(-(P_image/smap)**2)/(2*P_image))))

            # Locate any null values
            zeroInds = np.where(P_image/smap <= minimum_SNR)
            if len(zeroInds[0]) > 0:
                # Set all insignificant detections to zero
                P1map[zeroInds] = 0.0

            # Now compute an AstroImage object and store the corrected P1map
            P_image             = np.sqrt(self.Q**2 + self.U**2)
            P_image.data        = 100*P1map
            P_image.uncertainty = 100*smap

        else:
            raise ValueError("Did not recognize 'P_estimator' keyword value")

        return P_image

    def _compute_position_angle(self, PA_estimator='NAIVE'):
        """
        Computes the polarization image using the specified estimator.
        """

        # TODO: this should be placed into the matrix rotation when executing
        # 'compute_stokes_parameters'.
        #
        # # Grab the rotation of these images with respect to celestial north
        # Qrot = self.Q.rotation
        # Urot = self.U.rotation
        #
        # # Test if both Q and U have astrometry
        # if Qrot == Urot:
        #     # Add rotation angle to final deltaPA
        #     self.D_PA += Qrot
        # else:
        #     raise ValueError('The astrometry in U and Q do not seem to match.')

        # PA estimation
        ############################################################################
        # Estimate of the polarization position angle using the requested method.
        ############################################################################
        if PA_estimator.upper() == 'NAIVE':
            # Build the PA map and add the uncertaies in quadrature
            PA_image = ((
                np.rad2deg(0.5*np.arctan2(self.U, self.Q)) +
                (720.0*u.degree)) %
                (180.0*u.degree)
            )

            # # TODO: implement the MAIER_ET_AL PA uncertainty estimation.
            #
            # if s_DPA > 0.0:
            #     PA_image.uncertainty = np.sqrt(PA_image.uncertainty**2 + s_DPA**2)

        elif PA_estimator.upper() == 'MAX_LIKELIHOOD_1D':
            raise NotImplementedError()
        elif PA_estimator.upper() == 'MAX_LIKELIHOOD_2D':
            raise NotImplementedError()
        else:
            raise ValueError("Did not recognize 'PA_estimator' keyword value")

        return PA_image

    def compute_polarization_images(self, P_estimator='NAIVE', minimum_SNR=3.0,
        PA_estimator='NAIVE'):
        """
        Computes the polarization images from the Stokes parameter images.

        Parameters
        ----------
        P_estimator : str, optional, default: 'WARDLE_KRONBERG'
            The estimator to use when computing polarization percentage.

            'NAIVE' :

            'WARDLE_KRONBERG' :

            'MODIFIED_ASYMPTOTIC' :

            'MAIER_PIECEWISE' :

        minimum_SNR : int or float, optional, default: 3.0
            The minimum detected signal-to-noise ratio to require before
            rejecting the null-hypothesis: "There is no polarization here."

        PA_estimator : str, optional, default: 'NAIVE'
            The estimator to use when computing position angle.

            'NAIVE' :

        Returns
        -------
        P_image, PA_image : astroimage.BaseImage subclass
            Images containing the polarization percentage and position angle,
            respectively.
        """

        # Test if the Stokes parameter images are present
        if self._StokesParameters__stokesImages is None:
            raise ValueError('No Stokes parameter images were found')

        # Test if the calibration values were defined
        if self.__polarimeterCalibrationConstants is None:
            raise RuntimeError('Calibration constants must be provided')

        # Compute polarization percentage
        P_image = self._compute_polarization_percentage(P_estimator=P_estimator,
            minimum_SNR=minimum_SNR)

        # Compute the polarization position angle
        PA_image = self._compute_position_angle(PA_estimator=PA_estimator)

        return P_image, PA_image

    ##################################
    ### END OF COMPUTING METHODS   ###
    ##################################
