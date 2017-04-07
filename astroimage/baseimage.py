# TODO
#
# ***NEVER*** IMPORT CLASSES, ALWAYS IMPORT MODULES, e.g.
#
# >>> import astropy.wcs as wcs
# >>> testWCS = wcs.WCS()
#
# Rather than
#
# >>> from astropy.wcs import WCS
# >>> testWCS = WCS()
#
# This seems to be more proper programing behavior

# This tells Python 2.x to handle imports, division, printing, and unicode the
# way that `future` versions (i.e. Python 3.x) handles those things.
from __future__ import absolute_import, division, print_function, unicode_literals

# Core library imports
import os
import copy
import re
import warnings
import inspect
from datetime import datetime
from functools import lru_cache

# Scipy imports
import numpy as np

# Astropy imports
from astropy.io import fits
from astropy.nddata import NDDataArray, StdDevUncertainty
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import (ImageNormalize,  # Import the ImageNormalize
    ManualInterval, MinMaxInterval,                 # Import intervals
    ZScaleInterval, AsymmetricPercentileInterval,
    LinearStretch, LogStretch, AsinhStretch)        # Import stretchers
from astropy.stats import sigma_clipped_stats

# Matplotlib imports
import matplotlib as mpl
import matplotlib.colors as mcol
import matplotlib.pyplot as plt

# Define which functions, classes, objects, etc... will be imported via the command
# >>> from .baseimage import *
__all__ = ['BaseImage']

# Define a short helper class to create a class property
class ClassProperty(property):
    """A decotator class for class methods to turn them into properties"""
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()


# Now define the BaseImage class
class BaseImage(object):
    """
    Base class for all image subclasses in this package.

    Most of the arithmetic operations on images are defined in this base class.

    Should not be instantiated by users directly.
    """

    ##################################
    ### START OF CLASS VARIABLES   ###
    ##################################
    # Set a default header keyword dictionary. This dictionary enables header
    # keywords to be translated into BaseImage property values (and vice versa).
    # The name of the BaseImage property is stored in the dictionary keys as a
    # string (e.g. 'FILTER'), and the corresponding header keyword is stored in
    # the associated value (e.g. 'FILTNME3'). This tells the BaseImage class
    # that the `filter` property is stored in the fits header under the
    # 'FILTNME3' keyword. Both keys and values are case insensitive, as the code
    # converts everything to upper case to handle uniformly.
    # Thd default translation dictionary was defined to work for PRISM-Perkins.

    # TODO: Add the ability to provide an "Initalization function" which will
    # make modifications to the header (and data?) before proceeding to Generate
    # the image instance.
    #
    # I would use this to move the binning data from 'CRDELT1' and 'CRDELT2' to
    # the correct locations: 'ADELX_01' and 'ADELY_01'.
    #
    # By moving this header data, I can proceed to use the expected binning
    # entry:
    #
    # 'BINNING': ('ADELX_01', 'ADELY_01')
    #
    # and it would work!!! Then I dcan stop pre-treating "CRDELT*" keywords
    # every time I handle astrometry.
    __headerKeywordDict = {
        'AIRMASS': 'AIRMASS',
        # 'BINNING': ('CRDELT1', 'CRDELT2'),
        'BINNING': ('ADELX_01', 'ADELY_01'),
        'INSTRUMENT': 'INSTRUME',
        'FILTER': 'FILTNME3',
        'PRESCANWIDTH': 'PRESCAN',
        'OVERSCANWIDTH': 'POSTSCAN',
        'RA': 'TELRA',
        'DEC': 'TELDEC',
        # 'FRAME': 'RADESYS', # TODO: Eventually find a way to include this?
        'EXPTIME': 'EXPTIME',
        'DATE': 'DATE-OBS',
        'OBSTYPE': 'OBSTYPE',
        'UNIT': 'BUNIT',
        'SCALEFACTOR': 'BSCALE',
        'GAIN': 'AGAIN_01'
    }

    # Store a list of acceptable properties for this class
    __properties =[
        'airmass',
        'binning',
        'data',
        'date',
        'dec',
        'dtype',
        'expTime',
        'filename',
        'filter',
        'gain',
        'header',
        'height',
        'instrument',
        'obsType',
        'ra',
        'shape',
        'unit',
        'width'
    ]

    ##################################
    ### END OF CLASS VARIABLES     ###
    ##################################

    ##################################
    ### START OF STATIC METHODS    ###
    ##################################

    @staticmethod
    def _combine_headers(header1, header2):
        """
        Combines the relevant information of two constituent images into a final
        output header. This is useful for producing a single header to write to
        disk when two instances have been combined with a binary operation
        (e.g. __add__, __sub__, __mul__, __div__, etc...)

        Parameters
        ----------
        header1 : `astropy.io.fits.header.Header`
            Header containing information from one of the images to be combined

        header2 : `astropy.io.fits.header.Header`
            Header containing information from one of the images to be combined

        Returns
        -------
        outHead : `astropy.io.fits.header.Header`
            A new header instance which contains the relevant information from
            each of the images to be combined.
        """
        # TODO: evaluate whether or not I really need this method.
        # What should even happen in it if I do implement it?
        raise NotImplementedError

    @staticmethod
    def _dtype_to_bitpix(dtype):
        """
        Converts a numpy data type (e.g., float, or np.int32) into the
        corresponding FITS header BITPIX value (e.g., -32, 32).

        Parameters
        ----------
        self. numpy.dtype or <class 'type'>
            The data type to be converted into the corresponding FITS
            header BITPIX value.

        Returns
        -------
        bitpix : int
            The corresponding BITPIX value
        """
        # BITPIX lookup table for FITS standards
        # define BYTE_IMG      8  /*  8-bit unsigned integers */
        # define SHORT_IMG    16  /* 16-bit   signed integers */
        # define LONG_IMG     32  /* 32-bit   signed integers */
        # define LONGLONG_IMG 64  /* 64-bit   signed integers */
        # define FLOAT_IMG   -32  /* 32-bit single precision floating point */
        # define DOUBLE_IMG  -64  /* 64-bit double precision floating point */

        # Check which data type has been provided
        dtypes = (
            np.dtype(np.byte),
            np.dtype(np.int16),
            np.dtype(np.int32),
            np.dtype(np.int64),
            np.dtype(np.float32),
            np.dtype(np.float64)
        )
        bitpixes = (8, 16, 32, 64, -32, -64)
        bitpixDict = dict(zip(dtypes, bitpixes))

        return bitpixDict[dtype]

    ##################################
    ### END OF STATIC METHODS      ###
    ##################################

    ##################################
    ### START OF CLASS METHODS     ###
    ##################################

    @ClassProperty
    @classmethod
    def baseClass(cls):
        """Quick reference to the BaseImage class"""
        thisClass_methodResolutionOrder = cls.mro()
        baseClass = thisClass_methodResolutionOrder[-2]
        return baseClass

    @classmethod
    def _header_handler(cls, header):
        """Modifies header in a way to be specified by the user"""
        # By default, do not modify the header at all.
        return header

    @classmethod
    def set_header_handler(cls, handlerFunction):
        """
        Sets the `_header_handler` helper method for the class

        The `handlerFunction` will be applied to input headers as part of the
        __init__ method. The purpose of this is to provide the user with a means
        of making some minor changes to image headers as they are read in. For
        example, if the image binning were stored in the `CDELT1` and `CDELT2`
        keywords, those values can be moved elsewhere for safekeeping.

        Parameters
        ----------
        handlerFunction : function
            The handlerFunction must be a predefined function which takes an
            astropy.io.fits.header.Header object as its only argument, Modifies
            that header in some way, and then returns the modified header.
        """
        cls.baseClass._header_handler = handlerFunction

    @ClassProperty
    @classmethod
    def headerKeywordDict(cls):
        """The translation dictionary from header keywords to image properties"""
        return cls.baseClass.__headerKeywordDict

    @classmethod
    def set_headerKeywordDict(cls, translation):
        """
        Used to set the translation dictionary for translating the header
        keywords into image properties. Without this dictionary, it is
        impossible to accurately decipher the fits image headers.

        Parameters
        ----------
        translation : dict
            A dictionary containing *Image property names as keys and header
            keywords as values.

        Returns
        -------
        out : None

        Examples
        --------
        First set the translation dictionary, then read in images.

        >>> from astroimage import BaseImage
        >>> translation = {\\
            'airmass': 'AIRMASS', \\
            'binning': ('CRDELT1', 'CRDELT2'), \\
            'prescan': 'PRESCAN', \\
            'overscan': 'POSTSCAN', \\
            'ra': 'TELRA', \\
            'dec': 'TELDEC', \\
            'expTime': 'EXPTIME', \\
            'date': 'DATE-OBS' \\
            }
        >>> BaseImage.set_headerKeywordDict(translation)
        >>> img = BaseImage('img.fits')
        >>> print(img.airmass)
        1.43
        >>> print((img.ra, img.dec))
        (92.434234, 32.985404)

        You can also set the translation dictionary using any of the subclasses
        of the BaseImage class.

        >>> from astroimage import BaseImage
        >>> from astroimage import ScienceImage
        >>> ScienceImage.set_headerKeywordDict(translation)
        >>> BaseImage('img.fits')
        >>> print(img.airmass)
        1.43
        """
        # Check if a dictionary was provided
        if issubclass(type(translation), dict):
            raise TypeError("`translation` must be a dictionary")

        # Initalize a new dictionary to store upper cased keys and values
        reformattedTranslation = {}

        # Loop through each key
        for k, v in translation.items():
            # Check that we can make it upper case
            if issubclass(type(k), str):
                reformattedKey = k.upper()
            else:
                raise TypeError("Keys in `translation` must be strings")

            # Add the new key value pair to the new dictionary
            reformattedTranslation[reformattedKey] = v

        # Grab the baseclass and store the dictionary as a class variable
        cls.baseClass._BaseImage__headerKeywordDict = reformattedTranslation

    @ClassProperty
    @classmethod
    def properties(cls):
        return cls.__properties

    @classmethod
    def _parse_header_and_properties(cls, header, properties):
        """
        Extracts values from the header and properties dictionary.

        Parameters
        ----------
        header : astropy.io.fits.Header
            The header containing some (or all) of the image properties.

        properties: dict
            A dictionary to provide property values for this image. Values in
            this dictionary will take precedence over header values so that the
            user can force properties to be a specified value.

        Returns
        -------
        outDict : dict

        """
        # Initalize an empty dictionary to store all the extracted properties
        outDict = {}

        # Grab the keys to the headerKeywordDict
        headerKeywordDictKeys = cls.headerKeywordDict.keys()

        # Since the class which invokes this method will contain the correct
        # list of `properties` for that respective class, we can simply loop
        # through that list and test if either the HEADER or the properties
        # dictionary provide a value.
        for prop in cls.properties:
            # Build a upper case version of property name to remove ambiguity
            upperProp = prop.upper()

            # Test if this property might be read from the header
            if upperProp in headerKeywordDictKeys:
                # Grab the headerKeywordDict entry
                headerKeyword = cls.headerKeywordDict[upperProp]

                # Test if the keyword has multiple elements (e.g. binning)
                if isinstance(headerKeyword, (tuple, list)):
                    thisProperty = []
                    for headerSubKeyword in headerKeyword:
                        if headerSubKeyword in header:
                            thisProperty.append(header[headerSubKeyword])

                    # Add this property to the outDict if something was found
                    if len(thisProperty) > 0:
                        outDict[prop] = tuple(thisProperty)

                # Otherwise, test if the property is contained in the header
                elif headerKeyword in header:
                    outDict[prop] = header[headerKeyword]

            # Test if property is contained in the properties dictionary. This
            # will overwrite any values taken from the header.
            if prop in properties:
                # Now test if the property was previously set and issue a
                # warning before overwriting header property values
                if prop in outDict:
                    warningString = '\n\tProperty {0} = {1} from the header will be overwritten\n\twith {0} = {2} from the properties keyword.'
                    warningString =  warningString.format(
                        prop, outDict[prop], properties[prop])
                    warnings.warn(warningString)

                # Now update the output dictionary to have the new value
                outDict[prop] = properties[prop]

        # Return the new "properties" dictionary to the user
        return outDict

    @classmethod
    def read(cls, filename, properties={}):
        """
        Constructs an image instance by reading in a fits file from disk.

        Parameters
        ----------
        filename : str
            The path to the fits file to be read in from disk.

        properties : dict, optional, default: {}
            A dictionary containing properties for this image. These values may
            overwrite any values in the header, so be careful!

        Returns
        -------
        outImg : `RawImage` (or subclass)
            A new instance containing the data from the fits file specified or
            the provided array and header values.
        """
        # Check if filename is a string
        if not issubclass(type(filename), str):
            raise TypeError('`filename` must be a string path')

        # Check if filename is a valid file path
        if not os.path.isfile(filename):
            raise FileNotFoundError('File {0} not found'.format(filename))

        # Replace leading dot notation
        if filename[0:2] == ('.' + os.path.sep):
            filename = os.path.join(os.getcwd(), filename[2:])

        # Attempt to read in the file
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                HDUlist = fits.open(filename, do_not_scale_image_data=False)
        except:
            raise FileNotFoundError('File {0} could not be read.'.format(filename))

        # Initalize keyword argument dictionary for __init__ call
        initKwargs = {}

        # Grab the header from HDUlist set BZERO and BSCALE to trivial values
        thisHeader = HDUlist[0].header
        thisHeader['BZERO'] = 0
        thisHeader['BSCALE'] = 1

        # Store the header in the dictionary of arguments to pass to __init__
        initKwargs['header'] = thisHeader

        # Parse the number of bits used for each pixel
        floatFlag = thisHeader['BITPIX'] < 0
        numBits   = np.abs(thisHeader['BITPIX'])

        # TODO: treat the BITPIX header value more correctly!
        # See the following wedsite:
        # http://docs.astropy.org/en/stable/io/fits/
        #
        # Determine the appropriate data type for the array.
        if floatFlag:
            if numBits >= 64:
                dataType = np.dtype(np.float64)
            else:
                dataType = np.dtype(np.float32)
        else:
            if numBits >= 64:
                dataType = np.dtype(np.int64)
            elif numBits >= 32:
                dataType = np.dtype(np.int32)
            else:
                dataType = np.dtype(np.int16)

        # Store the data as the correct data type
        thisData = HDUlist[0].data.astype(dataType)

        # Loop through the HDUlist and check for an 'UNCERTAINTY' HDU
        for HDU in HDUlist:
            if HDU.name.upper() == 'UNCERTAINTY':
                initKwargs['uncertainty'] = HDU.data

        # Cleanup and close the HDUlist object
        HDUlist.close()

        # Pass the properties along to the __init__ method
        initKwargs['properties'] = properties

        # Now build the image instance using the correct __init__ method
        outImg = cls(thisData, **initKwargs)

        # Store the filename in the constructed instance
        outImg.__filename = filename

        # Return the image to the user
        return outImg

    ##################################
    ### END OF CLASS METHODS       ###
    ##################################

    def __init__(self, *args, **kwargs):
        """
        Constructs a `BaseImage` instance from provided arguments.

        Parameters
        ----------
        data : `numpy.ndarray`
            The array of values to be stored for this image

        header : `astropy.io.fits.header.Header`, optional
            The header to be associated with the `data` attribute

        properties : `dict`, optional
            A dictionary of properties to be set for this image
            (e.g. {'expTime': 25.0, ra': 132.323, 'dec': 32.987})

        Returns
        -------
        outImg : `BaseImage` (or subclass)
            A new instance containing the supplied data and header.
        """
        # Test that at least one argument was provided
        if len(args) > 1:
            raise TypeError('Required argument `data` (pos 1) not found')

        # Make a copy of the array
        thisData = copy.deepcopy(args[0])

        # Test if the first (and ONLY non-keyword) argument was an array
        if not issubclass(type(args[0]), np.ndarray):
            raise TypeError('`data` argument must be a numpy.ndarray')

        # Initalize a dictionary of keword arguments to pass to the NDDataArray
        kwargsForNDData = {}

        # Check if this CAN have and DOSE have uncertainty
        if 'uncertainty' in kwargs:
            # An array was provided, check it and store it
            thisUncertainty = copy.deepcopy(kwargs['uncertainty'])
            if not issubclass(type(thisUncertainty),
                (np.ndarray, StdDevUncertainty)):
                raise TypeError('`uncertainty` must be a numpy.ndarray or an astropy.nddata.StdDevUncertainty')

            kwargsForNDData['uncertainty'] = (
                StdDevUncertainty(thisUncertainty))

        if 'header' in kwargs:
            # A header was provided, so check it and store it
            thisHeader = copy.deepcopy(kwargs['header'])
            if not issubclass(type(thisHeader), fits.Header):
                raise TypeError('`header` must be an astropy.io.fits.header.Header instance')

            # If a header WAS found, then apply potential modifications to it
            thisHeader = BaseImage._header_handler(thisHeader)
        else:
            # If no header was provided, then generate a null header dictionary
            thisHeader = {}

        # Test if a fully defined WCS exists in the header
        thisWCS = WCS(thisHeader)
        if thisWCS.has_celestial:
            kwargsForNDData['wcs'] = thisWCS

        # TODO:
        # Build a method to parse which properties have been provided by the
        # image header and which properties are provided by the "properties"
        # keyword argument.
        #
        # TODO:
        # Ignore any properties not included in the current class variable
        # 'properties'.

        # Check if a 'properties' keyword argument was provided
        if 'properties' in kwargs:
            thisProperties = kwargs['properties']
            if type(thisProperties) is not dict:
                raise TypeError('`properties` keyword must be a dictionary like object')
        else:
            # If no properties dictionary was defined, then create a blank one
            thisProperties = {}

        # TODO: pass both header AND properties KEYWORD into a static method,
        # which will then return a single dictionary containing all the
        # properties to be uploaded to the "self" (PRECEDENCE IS GIVEN TO THE
        # PROPERTIES KWARG SO THAT THE USER CAN OVERWRITE THE HEADER information
        # IF THEY WANT TO.)

        # Parse the header and properties dictionary for their values
        newPropertiesDict = self._parse_header_and_properties(
            thisHeader,
            thisProperties
        )

        ####
        # Units are the only quantity that is specified in the header or the
        # properties dictionary, but which is stored in the NDDataArray object,
        # so it must be parsed separately.
        ###
        # Test a unit was defined in the header or properties dictionary
        if 'unit' in newPropertiesDict:
            # Grab the provided unit and test if it's the right type
            thisUnit = newPropertiesDict['unit']

            # TODO: Make this MUCH MUCH more robust
            if thisUnit == 'ADU': thisUnit = 'adu'

            if issubclass(type(thisUnit), (u.UnitBase, str)):
                try:
                    # Store the unit in the keyword arguements for the NDData object
                    kwargsForNDData['unit'] = u.Unit(thisUnit)
                except:
                    raise #TypeError('`unit` property must be convertible to an astropy.units unit')
            else:
                raise TypeError('`unit` property must be a string or an astropy.units.Unit')
        else:
            # If no unit was provided, then force it to be dimensionless
            kwargsForNDData['unit'] = u.dimensionless_unscaled

        # Now that we have everything we need, let's BUILD the NDDataArray,
        # and store it in the hidden "__fullData" attribute.
        self.__fullData = NDDataArray(thisData, **kwargsForNDData)

        if len(thisHeader) == 0:
            # No header was provided for this image, so build one.
            thisHeader  = fits.Header()
            thisHeader.append(('SIMPLE', True, 'Written by Python AstroImage package'))
            thisHeader.append(('BITPIX', self._dtype_to_bitpix(self.dtype), 'Number of bits per data pixel'))
            thisHeader.append(('NAXIS', 2, 'Number of data axes'))
            thisHeader.append(('NAXIS1', self.width, ))
            thisHeader.append(('NAXIS2', self.height, ))
            thisHeader.append(('EXTEND', True, 'FITS data may contain extensions'))
            thisHeader.add_comment("FITS (Flexible Image Transport System) format is defined in 'Astronomy")
            thisHeader.add_comment("and Astrophysics', volume 376, page 359; bibcode 2001A&A...376..359H")

        # Store the header
        self.__header = thisHeader

        # Import the properties to their respective attributes
        self._dictionary_to_properties(newPropertiesDict)

        # Update any missing header keywords with the stored property values
        self._properties_to_header()

        # Initalize the image axIM attritube to be None
        self.__image = None

    ##################################
    ### START OF PROPERTIES        ###
    ##################################

    @property
    def airmass(self):
        """The airmass at which this frame was observed"""
        return self.__airmass

    @property
    def data(self):
        """The array containing the image values"""
        return self.__fullData.data

    @data.setter
    def data(self, data):
        """
        Used to replace the private `data` attribute.

        Parameters
        ----------
        data : numpy.ndarray
            An array containing the array to be placed in the private `data`
            property

        Returns
        -------
        out : None
        """
        # Test if arr is a numpy array
        if not isinstance(data, np.ndarray):
            raise TypeError('`data` must be an instance of numpy.ndarray')

        # Test if the replacement array matches the previous array's shape
        newShape = self.shape != data.shape
        if newShape:
            raise ValueError('`data` must have shape ({0}x{1})'.format(
                *self.shape))

        # Update the image array
        self.__fullData = NDDataArray(
            data,
            uncertainty=self.__fullData.uncertainty,
            unit=self.__fullData.unit,
            wcs=self.__fullData.wcs
        )

        # Update the dtype and header values to match the new array
        self.__header['BITPIX'] = self._dtype_to_bitpix(self.dtype)

        return None

    @property
    def binning(self):
        """The binning of the array read from the sensor"""
        return self.__binning

    @property
    def date(self):
        """The date of the observation"""
        return self.__date

    @property
    def dec(self):
        """The declination of the object"""
        if self.__centerCoord is None:
            return None
        return self.__centerCoord.dec

    @property
    def dtype(self):
        """The data type of the array stored in the `data` attribute"""
        return self.__fullData.dtype

    @property
    def expTime(self):
        """The exposure time of the image"""
        # !!!ASSUME THAT EXPOSURE TIME IS IN SECONDS!!!
        return self.__expTime * u.second

    @property
    def filename(self):
        """The filename (on disk) of the image"""
        return self.__filename

    @filename.setter
    def filename(self, name):
        """Sets the provide `filename` property"""
        if type(name) is not str:
            raise TypeError('`name` must be a string object')

        self.__filename = name

    @property
    def filter(self):
        """The filter through which this image was obtained."""
        return self.__filter

    @property
    def gain(self):
        """The gain of the detector."""
        return self.__gain

    @property
    def header(self):
        """The header associated with this fits image"""
        return self.__header

    @header.setter
    def header(self, head):
        """
        Used to replace the private `header` attribute.

        Parameters
        ----------
        head : astropy.io.fits.header.Header
            An astropy Header instance to be placed in the `header` attribute.

        Returns
        -------
        out : None
        """
        # Test if head is a fits.Header object
        if not isinstance(head, fits.Header):
            raise TypeError('`head` must be an instance of astropy.io.fits.header.Header.')

        self.__header = head

        return None

    @property
    def height(self):
        """The height, in pixels, of the array stored in the `data` attribute"""
        return self.__fullData.shape[0]

    @property
    def instrument(self):
        """The instrument from which this image was obtained."""
        return self.__instrument

    @property
    def ra(self):
        """The right ascension of the observation"""
        if self.__centerCoord is None:
            return None
        return self.__centerCoord.ra

    @property
    def shape(self):
        """The dimnesnional shape of the array stored in the `data` attribute"""
        return self.__fullData.shape

    @property
    def unit(self):
        """The units of the data stored in the `data` attribute"""
        return self.__fullData.unit

    @property
    def has_angle_units(self):
        """A boolean flag for if the unit can be considered an angle"""
        return self.unit.is_equivalent(u.rad)

    @property
    def has_dimensionless_units(self):
        """A boolean flag for if the unit can be considered dimensionless"""
        # Check if this is an angle
        return self.unit.is_equivalent(u.dimensionless_unscaled)

    @property
    def width(self):
        """The width, in pixels, of the array stored in the `data` attribute"""
        return self.__fullData.shape[1]

    @property
    def obsType(self):
        """The observation type of the image"""
        return self.__obsType

    # @obsType.setter
    # def obsType(self, obs):
    #     """Sets the private `obsType` property"""
    #     if type(obs) is not str:
    #         raise TypeError('`obs` must be a string object')
    #
    #     self.__obsType = obs

    @property
    @lru_cache()
    def mode(self):
        """An estimate of the statistical mode of this image"""
        # Compute the number of bins that will be needed to find mode
        numBins = np.int(np.ceil(0.1*(np.max(self.data) - np.min(self.data))))

        # Loop through larger and larger binning until find unique solution
        foundMode = False
        while not foundMode:
            # Generate a histogram of the flat field
            hist, flatBins = np.histogram(self.data.flatten(), numBins)

            # Locate the histogram maximum
            maxInds = (np.where(hist == np.max(hist)))[0]
            if maxInds.size == 1:
                # Grab the index of the maximum value and shrink
                maxInd = maxInds[0]
                foundMode = True
            else:
                # Shrink the NUMBER of bins to help find a unqiue maximum
                numBins *= 0.9

        # Estimate flatMode from histogram maximum
        flatMode = np.mean(flatBins[maxInd:maxInd+2])*self.unit

        return flatMode

    @ property
    def image(self):
        """The AxesImage instance in which the plotted imgage is shown."""
        if self.__image is not None:
            return self.__image

        return None

    @property
    def axes(self):
        """The Axes instance in which the plotted image is stored"""
        if self.image is not None:
            return self.image.axes

        return None

    @property
    def figure(self):
        """The Figure instance in which the plotted axes is stored."""
        if self.axes is not None:
            return self.axes.figure

        return None

    ##################################
    ### END OF PROPERTIES          ###
    ##################################

    ##################################
    ### START OF MAGIC METHODS     ###
    ##################################

    ### Binary operator magic methods ###
    def __pos__(self):
        """
        Implements the unary `+` operation for the values in this image

        Parameters
        ----------
        No parameters

        Returns
        -------
        self : `BaseImage` (or subclass)
            The same instance as the input instance.
        """
        return self

    def __neg__(self):
        """
        Implements the unary `-` operation for the values in this image

        Parameters
        ----------
        No parameters

        Returns
        -------
        outImg : `BaseImage` (or subclass)
            A copy of the instance with the self.data attribute replaced with
            its negative value.
        """
        # Implements behavior for negation (e.g. -some_object)
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(
            -1*self.data,
            uncertainty=self._BaseImage__fullData.uncertainty,
            unit=self._BaseImage__fullData.unit,
            wcs=self._BaseImage__fullData.wcs
        )

        return outImg

    def __abs__(self):
        """
        Computes the absolute value of the values in this image

        Parameters
        ----------
        No parameters

        Returns
        -------
        outImg :  `BaseImage` (or subclass)
            A copy of the instance with the self.data property replaced with its
            absolute value.
        """
        outImg            = self.copy()
        outImg.__fullData = NDDataArray(
            np.abs(self.data),
            uncertainty=self._BaseImage__fullData.uncertainty,
            unit=self._BaseImage__fullData.unit,
            wcs=self._BaseImage__fullData.wcs
        )

        return outImg

    def __add__(self, other):
        """
        Computes the some of this image and the supplied value(s)

        Parameters
        ----------
        other : integer, float, `BaseImage` (or subclass)
            Either a constant or array of values to be added to the array of
            this `BaseImage` (or subclass) instance.

        Returns
        -------
        outImg : `BaseImage` (or subclass)
            A new instance which is the sum of this instance and the input value
            or instance. If other is a `BaseImage` (or subclass), the new header
            will contain a combination of the relevant information from the two
            input headers.
        """
        # Grab the data if posible
        if issubclass(type(other), BaseImage):
            # Add another astroimage instance
            otherData = NDDataArray(other._BaseImage__fullData)
        elif issubclass(type(other), u.Quantity):
            # Add a Quantity instance
            otherData = NDDataArray(other)
        elif (self.has_dimensionless_units
            and issubclass(type(other),
            (int, np.int8, np.int16, np.int32, np.int64,
            float, np.float16, np.float32, np.float64))):
            # Add a unitless scalar quantity (if image is unitless)
            otherData = other
        else:
            # Incompatible types and/or units
            raise TypeError('Cannot add {0} with {1} units and {2}'.format(
                type(self).__name__, str(self.unit), type(other).__name__))

        # Attempt the addition
        try:
            outImg   = self.copy()
            selfData = NDDataArray(self._BaseImage__fullData)
            outImg._BaseImage__fullData = selfData.add(otherData)
        except:
            raise

        # Return the added image
        return outImg

    def __radd__(self, other):
        """
        Computes the sum of this image and the supplied value(s)

        Parameters
        ----------
        other : integer, float, BaseImage or subclass
            Either a constant or array of values to be added to the array of
            this `BaseImage` (or subclass) instance.

        Returns
        -------
        outImg : `BaseImage` (or subclass)
            A new instance which is the sum of this instance and the input
            instance. If other is a `BaseImage` (or subclass), the new header
            will contain a combination of the relevant information from the two
            input headers.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Computes some other value(s) subtracted from this image

        Parameters
        ----------
        other : integer, float, BaseImage or subclass
            Either a constant or array of values to be subtracted from the array
            of this `BaseImage` (or subclass) instance.

        Returns
        -------
        outImg : `BaseImage` (or subclass)
            A new instance which is the difference of this instance and the
            input value or instance. If other is a `BaseImage` (or subclass),
            the new header will contain a combination of the relevant
            information from the two input headers.
        """
        # Grab the data if posible
        if issubclass(type(other), BaseImage):
            # Subtract another astroimage instance
            otherData = NDDataArray(other._BaseImage__fullData)
        elif issubclass(type(other), u.Quantity):
            # Subtract a Quantity instance
            otherData = NDDataArray(other)
        elif (self.has_dimensionless_units
            and issubclass(type(other),
            (int, np.int8, np.int16, np.int32, np.int64,
            float, np.float16, np.float32, np.float64))):
            # Subtract a unitless scalar quantity (if image is unitless)
            otherData = other
        else:
            # Incompatible types
            raise TypeError('Cannot subtract {0} and {1}'.format(
                type(self).__name__, type(other).__name__))

        # Attempt the subtraction
        try:
            outImg   = self.copy()
            selfData = NDDataArray(self._BaseImage__fullData)
            outImg._BaseImage__fullData = selfData.subtract(otherData)
        except:
            raise

        # Return the subtracted image
        return outImg

    def __rsub__(self, other):
        """
        Computes the this image subtracted from some other value(s)

        Parameters
        ----------
        other : integer, float, BaseImage or subclass
            Either a constant or array of values from which to subtract the
            array of this `BaseImage` (or subclass) instance.

        Returns
        -------
        outImg : `BaseImage` (or subclass)
            A new instance which is the difference between this instance and the
            input value or instance. If other is a `BaseImage` (or subclass),
            the new header will contain a combination of the relevant
            information from the two input headers.
        """
        return -self.__sub__(other)

    def __mul__(self, other):
        """
        Computes the product of this image multiplied by some other value(s)

        Parameters
        ----------
        other : integer, float, BaseImage or subclass
            Either a constant or array of values by which to multiply the array
            of this `BaseImage` (or subclass) instance.

        Returns
        -------
        outImg : `BaseImage` (or subclass)
            A new instance which is the product of this instance and the input
            value or instance. If other is a `BaseImage` (or subclass), the new
            header will contain a combination of the relevant information from
            the two input headers.
        """
        # Grab the data if posible
        if issubclass(type(other), BaseImage):
            # Multiply another astroimage instance
            otherData = NDDataArray(other._BaseImage__fullData)
        elif issubclass(type(other), u.Quantity):
            # Multiply a Quantity instance
            otherData = NDDataArray(other)
        elif (issubclass(type(other),
            (int, np.int8, np.int16, np.int32, np.int64,
            float, np.float16, np.float32, np.float64))):
            # Multiply a unitless scalar quantity regardless of image units
            otherData = other
        else:
            # Incompatible types
            raise TypeError('Cannot multiply {0} and {1}'.format(
                type(self).__name__, type(other).__name__))

        # Attempt the multiplication
        try:
            outImg   = self.copy()
            selfData = NDDataArray(self._BaseImage__fullData)
            outImg._BaseImage__fullData = selfData.multiply(otherData)
        except:
            raise

        # Return the multiplied image
        return outImg

    def __rmul__(self, other):
        """
        Computes the product of the supplied value(s) multiplied by this image

        Parameters
        ----------
        other : integer, float, BaseImage or subclass
            Either a constant or array of values by which to multiply the array
            of this `BaseImage` (or subclass) instance.

        Returns
        -------
        outImg : `BaseImage` (or subclass)
            A new instance which is the product between this instance and the
            input value or instance. If other is a `BaseImage` (or subclass),
            the new header will contain a combination of the relevant
            information from the two input headers.
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Computes this image divided by the supplied value(s)

        Parameters
        ----------
        other : integer, float, BaseImage or subclass
            Either a constant or array of values by which to divide the array of
            this `BaseImage` (or subclass) instance.

        Returns
        -------
        outImg : `BaseImage` (or subclass)
            A new instance which is the quotient of this instance and the input
            value or instance. If other is a `BaseImage` (or subclass), the new
            header will contain a combination of the relevant information from
            the two input headers.
        """
        # Grab the data if posible
        if issubclass(type(other), BaseImage):
            # Divide another astroimage instance
            otherData = NDDataArray(other._BaseImage__fullData)
        elif issubclass(type(other), u.Quantity):
            # Divide a Quantity instance
            otherData = NDDataArray(other)
        elif (issubclass(type(other),
            (int, np.int8, np.int16, np.int32, np.int64,
            float, np.float16, np.float32, np.float64))):
            # Divide a unitless scalar quantity regardless of image units
            otherData = other
        else:
            # Incompatible types
            raise TypeError('Cannot divide {0} and {1}'.format(
                type(self).__name__, type(other).__name__))

        # Attempt the division
        try:
            outImg   = self.copy()
            selfData = NDDataArray(self._BaseImage__fullData)
            outImg._BaseImage__fullData = selfData.divide(otherData)
        except:
            raise

        # Return the divided image
        return outImg

    def __rtruediv__(self, other):
        """
        Computes some other value(s) divided by this image

        Parameters
        ----------
        other : integer, float, BaseImage or subclass
            Either a constant or array of values which will be divided by the
            array of this `BaseImage` (or subclass) instance.

        Returns
        -------
        outImg : `BaseImage` (or subclass)
            A new instance which is the quotient of this instance and the input
            value or instance. If other is a `BaseImage` (or subclass), the new
            header will contain a combination of the relevant information from
            the two input headers.
        """
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(np.ones(self.shape))
        outImg = outImg.__mul__(other)
        outImg = outImg.__truediv__(self)

        return outImg

    def __pow__(self, other):
        """
        Computes this image raised to some other power

        Parameters
        ----------
        other : integer, float, BaseImage or subclass
            Either a constant or array of values which will be divided by the
            array of this `BaseImage` (or subclass) instance.

        Returns
        -------
        outImg : `BaseImage` (or subclass)
            A new instance which is the quotient of this instance and the input
            value or instance. If other is a `BaseImage` (or subclass), the new
            header will contain a combination of the relevant information from
            the two input headers.
        """
        # Grab the data if posible
        if (issubclass(type(other),
            (int, np.int8, np.int16, np.int32, np.int64,
            float, np.float16, np.float32, np.float64))):
            # Divide a unitless scalar quantity regardless of image units
            otherData   = other
            otherUncert = 0.0
        else:
            # Incompatible types
            raise TypeError('Cannot exponentiate {0} and {1}'.format(
                type(self).__name__, type(other).__name__))

        # Attempt the exponentiation
        try:
            # Grab the self Data
            selfData = self.data.copy()

            # Compute the output
            outData   = selfData**otherData

            # Check if error propagation is required
            if self._BaseImage__fullData.uncertainty is not None:
                # Grab the image uncertainty
                selfUncert = self.uncertainty

                # Compute and convert the output uncertainty
                outUncert = np.abs(outData) * np.sqrt(
                    (selfUncert*otherData/selfData)**2
                )
                outUncert = StdDevUncertainty(outUncert)
            else:
                outUncert = None

            # Try to exponentiate the units, or else set them to None
            try:
                outUnit = (self.unit**otherData)
            except:
                outUnit = None

            # Make a copy of the image and store the output in it
            outImg  = self.copy()
            outImg._BaseImage__fullData = NDDataArray(
                outData,
                uncertainty=outUncert,
                unit=outUnit,
                wcs=self._BaseImage__fullData.wcs
            )
        except:
            raise

        # Return the divided image
        return outImg

    # There are, of course, other operators we could define, but I don't think
    # we will need them, so we don't bother defining them just yet.

    def __str__(self):
        """
        Returns a string representation of the content stored in the image.

        Parameters
        ----------
        No parameters

        Returns
        -------
        description : str
            A string description of the BaseImage instance
        """

        # Build the description string from the instance properties
        description = 'SUMMARY FOR     ' + str(self.filename) + '\n' + \
        'Image Type:     ' + str(type(self).__name__) + '\n' + \
        'Instrument:     ' + str(self.instrument) + '\n' + \
        'Filter:         ' + str(self.filter) + '\n' + \
        '------------------------------------\n' + \
        'Airmass:        ' + str(self.airmass) + '\n' + \
        'Binning:        ' + str(self.binning[0]) + ' x ' + str(self.binning[1]) + '\n' + \
        'UTC Obs Time:   ' + str(self.date) + '\n' + \
        'RA/DEC:         ' + str(self.ra) + '   ' + str(self.dec) + '\n' + \
        'Exposure Time:  ' + str(self.expTime) + ' seconds\n' + \
        'Image Size:     ' + str(self.width) + ' x ' + str(self.height) + '\n' + \
        'Units:          ' + str(self.unit)

        return description

    def __repr__(self):
        """
        Simply invokes the `str` method to produce a string representation of
        the content stored in this `BaseImage` (or subclass) instance.

        Parameters
        ----------
        No parameters

        Returns
        -------
        out : str
            A string description of the BaseImage instance
        """

        return self.__str__()

    ##################################
    ### END OF MAGIC METHODS       ###
    ##################################

    ##################################
    ### START OF CUSTOM SETTERS    ###
    ##################################

    # TODO: Figure out whether I should use an explicit "set_data" method or
    # simply migrate this code over to a @arr.setter decorated method.
    #
    # PRO @arr.setter
    # Never worry about changing the interface to set/get the arr attribute
    #
    # CON @arr.setter
    # Cannot use any keyword arguments to FORCE the array shape to change size.
    #
    # Possible workaround:
    # Use keyword arguments in the __init__ method to allow a new BaseImage
    # instance to be constructed using predefined arrays and header. E.g.
    #
    # >>> from astroimage import BaseImage
    # >>> from astropy.io import fits
    # >>> arr1    = np.arange(100).reshape((10,10))
    # >>> sigma1  = np.sqrt(arr1)
    # >>> header1 = fits.Header({'NAXIS1':10, 'NAXIS2':10})
    # >>> img = BaseImage(arr=arr, uncertainty=sigma1, header=header1)
    # >>> print(
    # ... ((img.data == arr1).all(),
    # ... (img.uncertainty == sigma1).all(),
    # ... (img.header == header1))
    # ... )
    # (True, True, True)

    def _dictionary_to_properties(self, propDict):
        """
        Sets the instance properties from the values supplied in the propDict
        """
        if 'binning' in propDict:
            try:
                binning = tuple(propDict['binning'])
                assert len(binning) == 2
                self.__binning = binning
            except:
                raise TypeError('`binning` property must be convertible to a two elemnt tuple')
        else:
            self.__binning = (1, 1)

        if 'instrument' in propDict:
            inst = propDict['instrument']
            if type(inst) is not str:
                raise TypeError('`instrument` property must be a string')
            self.__instrument = inst
        else:
            self.__instrument = None

        if 'filename' in propDict:
            filename = propDict['instrument']
            if type(filename) is not str:
                raise TypeError('`filename` property must be a string')
            self.__filename = filename
        else:
            self.__filename = None

        if 'filter' in propDict:
            filt = propDict['filter']
            if type(filt) is not str:
                raise TypeError('`filter` property must be a string')
            self.__filter = filt
        else:
            self.__filter = None

        if 'gain' in propDict:
            try:
                gain = np.float(propDict['gain'])
            except:
                raise TypeError('`airmass` property must be convertible to a float')
            self.__gain = gain
        else:
            self.__gain = None

        if 'obsType' in propDict:
            obsType = propDict['obsType']
            if type(obsType) is not str:
                raise TypeError('`obsType` property must be a string')
            self.__obsType = obsType.strip().upper()
        else:
            self.__obsType = None

        if 'date' in propDict:
            # TODO: use regular expressions (re module) to extract date/time
            date = propDict['date']
            if type(date) is not str:
                raise TypeError('`date` property must be a string')
            self.__date = date
        else:
            # Set the current date-time as the date value
            nowDateStr = str(datetime.utcnow())
            # Trim off the fractions of a second
            nowDateStr = (nowDateStr.split('.'))[0]
            # Reformat to match what would be provided by a FITS header
            nowDateStr = nowDateStr.replace(' ', '  ')
            nowDateStr = nowDateStr.replace('T', '  ')
            self.__date = nowDateStr

        if 'airmass' in propDict:
            try:
                airmass = float(propDict['airmass'])
            except:
                raise TypeError('`airmass` property must be convertible to a float')
            self.__airmass = airmass
        else:
            self.__airmass = None

        # TODO:
        # Include the "RADESYSa" to set a "radecsys" @property
        # Use this property to help set "frame=self.radecsys" in the SkyCoord
        # below.
        if 'ra' in propDict and 'dec' in propDict:
            try:
                coord = SkyCoord(
                    ra=propDict['ra'],
                    dec=propDict['dec'],
                    unit=(u.hour, u.degree)
                )
            except:
                raise

            self.__centerCoord = coord
        else:
            self.__centerCoord  = None

        # if 'ra' in propDict:
        #     try:
        #         ra = float(propDict['ra'])
        #     except:
        #         raise TypeError('`ra` propety must be convertible to a float')
        #     self.__ra = ra
        # else:
        #     self.__ra = None
        #
        # if 'dec' in propDict:
        #     try:
        #         dec = float(propDict['dec'])
        #     except:
        #         TypeError('`dec` property must be convertible to a float')
        #     self.__dec = dec
        # else:
        #     self.__dec = None

        if 'expTime' in propDict:
            try:
                expTime = float(propDict['expTime'])
            except:
                TypeError('`expTime` property must be convertible to a float')
            self.__expTime = expTime
        else:
            self.__expTime = None

    def _properties_to_header(self):
        """
        Forces attribute values back into the instance header

        Uses the dictionary stored in the BaseImage.headerKeywordDict to decide
        what header keywords to use for storing the attribute values.
        """
        # Grab the translation dictionary keys so we can check which
        # properties have been assigned to header keywords
        headerKeywordDictKeys = self.__headerKeywordDict.keys()

        # Check for a translation dictionary entry so that each property
        # will be set using the correct header keyword.

        # TODO: write a method to "remove_empty_properties_from_header",
        # which should LOOP through all the listed properties, test if its
        # empty and if it is, then attempt to delete it.

        if self.binning is not None:
            # If binning has been specified by the instrument, then set it...
            try:
                # Look for a provided binning translator and set it
                binKeys = self.headerKeywordDict['BINNING']
                for k, di in zip(binKeys, self.binning):
                    self.__header[k] = di
            except: pass
        else:
            # Otherwise delete binning data from the header
            try:
                # Look for a provided binning translator and set it
                binKeys = self.headerKeywordDict['BINNING']
                for k in binKeys:
                    del self.__header[k]
            except: pass

        # If no units were provided, then it should be dimensionless
        if self.has_dimensionless_units:
            # Recast the data as a dimensionless quantity
            self._BaseImage__fullData = NDDataArray(
                self.data,
                uncertainty=self._BaseImage__fullData.uncertainty,
                unit=u.dimensionless_unscaled,
                wcs=self._BaseImage__fullData.wcs
            )

        # Attempt to add the units to the header
        try:
            unitKey = self.headerKeywordDict['UNIT']
            self.__header[unitKey] = '{0:FITS}'.format(self.unit)
        except: pass

        # Continue with all the other properties
        if self.instrument is not None:
            try:
                instrumentKey = self.headerKeywordDict['INSTRUMENT']
                self.__header[instrumentKey] = self.instrument
            except: pass
        else:
            try:
                instrumentKey = self.headerKeywordDict['INSTRUMENT']
                del self.__header[instrumentKey]
            except: pass

        if self.filter is not None:
            try:
                filterKey = self.headerKeywordDict['FILTER']
                self.__header[filterKey] = self.filter
            except: pass
        else:
            try:
                filterKey = self.headerKeywordDict['FILTER']
                del self.__header[filterKey]
            except: pass

        if self.obsType is not None:
            try:
                obsTypeKey = self.headerKeywordDict['OBSTYPE']
                self.__header[obsTypeKey] = self.obsType
            except: pass
        else:
            try:
                obsTypeKey = self.headerKeywordDict['OBSTYPE']
                del self.__header[obsTypeKey]
            except: pass

        if self.date is not None:
            try:
                dateKey = self.headerKeywordDict['DATE']
                # TODO: Perform the reverse translation
                # TODO: use regular expressions (re module) to extract date/time
                self.__header[dateKey] = str(self.date).replace('  ', 'T')
            except: pass
        else:
            try:
                dateKey = self.headerKeywordDict['DATE']
                del self.__header[dateKey]
            except: pass

        if self.airmass is not None:
            try:
                airmassKey = self.headerKeywordDict['AIRMASS']
                self.__header[airmassKey] = self.airmass
            except: pass
        else:
            try:
                airmassKey = self.headerKeywordDict['AIRMASS']
                del self.__header[airmassKey]
            except: pass

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

        if self.expTime is not None:
            try:
                expTimeKey = self.headerKeywordDict['EXPTIME']
                self.__header[expTimeKey] = self.expTime
            except: pass
        else:
            try:
                expTimeKey = self.headerKeywordDict['EXPTIME']
                del self.__header[expTimeKey]
            except: pass

        if self.gain is not None:
            try:
                gainKey = self.headerKeywordDict['GAIN']
                self.__header[gainKey] = self.gain
            except: pass
        else:
            try:
                gainKey = self.headerKeywordDict['GAIN']
                del self.__header[gainKey]
            except: pass

    def _build_HDUs(self):
        """
        Extracts the stored data and returns a list of HDU objects
        """
        # Copy the output array and header
        outArr  = self.data.copy()
        outHead = self.header.copy()

        # Update the header bitpix value
        outHead['BITPIX'] =  self._dtype_to_bitpix(self.dtype)

        # Build a new HDU object to store the data
        arrHDU = fits.PrimaryHDU(data = outArr,
                                 do_not_scale_image_data = True)

        # Replace the original header (since some cards may have been stripped)
        arrHDU.header = self.header

        # Construct the HDU list
        HDUs = [arrHDU]

        return HDUs

    ##################################
    ### END OF CUSTOM SETTERS      ###
    ##################################

    ##################################
    ### START OF OTHER METHODS     ###
    ##################################

    def astype(self, dtype, copy=True):
        """
        Converts the data stored in the `data` to the requested datatype and
        updates the header to reflect that. If `copy` is set to True, then a
        copy of the retyped image is returned to the user.

        Parameters
        ----------
        self : numpy.dtype or <class 'type'>
            The data type to convert the image to.

        dtype : str or dtype
            Typecode or data-type to which the image is cast

        copy : bool, optional, defualt: True
            By default, astype always returns a newly allocated image. If this
            set set to False, the the input image is returned instead of a copy.

        Returns
        -------
        outImg : BaseImage (or subclass)
            Unless copy is False, outImg is a new image with the same data cast
            into the specified data type.
        """
        # Catch a bad dtype error
        try:
            dtype1 = np.dtype(dtype)
        except:
            raise TypeError('data type not understood')

        # Construct the output image
        if copy or (dtype1 is not self.dtype):
            outImg = self.copy()
        else:
            outImg = self

        # Construct a new, recast, data structure
        outImg._BaseImage__fullData = NDDataArray(
            self.data.astype(dtype1),
            uncertainty=self._BaseImage__fullData.uncertainty,
            unit=self._BaseImage__fullData.unit,
            wcs=self._BaseImage__fullData.wcs
        )

        # Update the output header
        outImg.__header['BITPIX'] = self._dtype_to_bitpix(dtype1)

        return outImg

    def convert_units_to(self, unit):
        """
        Converts the image to the specified unit.

        Parameters
        ----------
        unit : astropy.units.core.UnitBase instance, str
            An object that represents the unit to convert to. Must be an
            UnitBase object or a string parseable by the astropy.units package.

        Returns
        -------
        out : BaseImage (or subclass)
            Unless copy is False, outImg is a new image with the same data cast
            into the specified units.
        """
        # Attempt the unit conversion for the
        try:
            selfUnitData = self.data*self.unit
            outData      = selfUnitData.to(unit)
            outData      = outData.value
        except:
            raise

        # If there is an uncertainty, then convert those units, too
        if self._BaseImage__fullData.uncertainty is not None:
            selfUnitUncert = self.uncertainty*self.unit
            outUncert      = selfUnitUncert.to(unit)
            outUncert      = StdDevUncertainty(outUncert.value)
        else:
            outUncert = None

        # Construct the output image
        outImg = self.copy()
        outImg._BaseImage__fullData = NDDataArray(
            outData,
            uncertainty=outUncert,
            unit=unit,
            wcs=self._BaseImage__fullData.wcs
        )

        return outImg

    def copy(self):
        """
        A convenience method for the copy.deepcopy function.

        Returns a copy of the `BaseImage` (or subclass) object.

        Parameters
        ----------
        No parameters

        Returns
        -------
        outImg : `BaseImage` (or subclass)
            An exact replicate of the `BaseImage` (or subclass) on which the
            copy method was invoked. The returned will have a different memory
            address, but it will contain the exact same information.
        """
        # Check if axIm attribute is occupied and busy
        if self.image is not None:
            # If it is then save it, clear it out, and prepare to put it back...
            axIm = self.image
            self.__image = None
            replaceImage = True
        else:
            replaceImage = False

        # Make a copy of the image
        outImg = copy.deepcopy(self)

        if replaceImage:
            self.__image = axIm

        return outImg

    def write(self, filename = None, dtype = None, clobber = False):
        """
        Writes the `BaseImage` (or subclass) to disk.

        Parameters
        ----------
        filename : str, optional
            Supplies the fully qualified or relative path to the location on
            disk where the `BaseImage` (or subclass) will be written. If no string
            is provided, then the image will be written to whatever location
            is currently stored in the `filename` attribute.

        dtype : numpy.dtype or <class 'type'>, optional
            The desired data-type for the output image.  If not given, then the
            image will be saved as the dtype currently in the `data` attribute.
            (This is sometimes a 64-bit integer or float, which is generally
            much more precision than required for most scientific purposes. If
            you don't need this much precision and are saving many files to
            disk, then consider downcasting to a numpy.int32 or numpy.float32
            type.)

        clobber : bool, optional, default: False
            If True, then any file on disk with the same name is overwritten.

        Returns
        -------
        out : None
        """
        # Test if a filename was provided. If not, then grab the
        if filename is None:
            # If no filename was provided, check if one was stored in this instance.
            if self.filename is None:
                raise ValueError('`filename` must be either explicitly provided or stored in this instance')

            # If a filename was stored in this instance, then use that filename
            filename = self.filename

        # Check if the explicitly provided filename is a string
        if type(filename) is not str:
            raise TypeError('`filename` must be a string')

        # If a data type was specified, recast output data into that format
        if dtype is None: dtype = self.dtype

        # First convert the array data
        try:
            outImg = self.astype(dtype)
        except:
            raise ValueError('`dtype` value not recognized')

        # Now force all the header values to be updated
        outImg._properties_to_header()

        # Extract the HDU data
        HDUs = outImg._build_HDUs()

        # Build the final output HDUlist
        HDUlist = fits.HDUList(HDUs)

        # Write file to disk
        HDUlist.writeto(filename, clobber=clobber)

        return None

    def rebin(self, nx, ny, total=False):
        """
        Rebins the image to have a specified shape.

        Parameters
        ----------
        nx, ny: int
            The target number of pixels along the horizontal (nx) and vertical
            (ny) axes.

        total : bool, optional, default: False
            If set to true, then returned array is total of the binned pixels
            rather than the average.

        Returns
        -------
        outImg : `BaseImage` (or subclass) or None
            If copy was set to True, then a rebinned copy of the original image
            is returned. Otherwise None is returned and the original image is
            rebinned in place.
        """
        # Grab the shape of the initial array
        ny0, nx0 = self.shape

        # TODO: Catch the case of upsampling along one axis but downsampling
        # along the other. This should not be possible!

        # Test for improper result shape
        goodX = ((nx0 % nx) == 0) or ((nx % nx0) == 0)
        goodY = ((ny0 % ny) == 0) or ((ny % ny0) == 0)
        if not (goodX and goodY):
            raise ValueError('Result dimensions must be integer factor of original dimensions')

        # First test for the trivial case
        if (nx0 == nx) and (ny0 == ny):
            return self.copy()

        # Compute the pixel ratios of upsampling and down sampling
        xratio, yratio = np.float(nx)/np.float(nx0), np.float(ny)/np.float(ny0)
        pixRatio       = np.float(xratio*yratio)
        aspect         = yratio/xratio         #Measures change in aspect ratio.

        if ((nx0 % nx) == 0) and ((ny0 % ny) == 0):
            # Handle integer downsampling
            # Get the new shape for the array and compute the rebinning shape
            sh = (ny, ny0//ny,
                  nx, nx0//nx)

            # Computed weighted rebinning
            rebinArr = (self.data.reshape(sh).sum(-1).sum(1))

            # Check if total flux conservation was requested.
            # If not, then multiply by the pixel size ratio.
            if not total: rebinArr *= pixRatio

        elif ((nx % nx0) == 0) and ((ny % ny0) == 0):
            # Handle integer upsampling
            rebinArr = np.kron(
                self.data,
                np.ones((ny//ny0, nx//nx0))
            )

            # Check if total flux conservation was requested.
            # If it was, then divide by the pixel size ratio.
            if total: rebinArr /= pixRatio

        # Compute the output uncertainty
        if self._BaseImage__fullData.uncertainty is not None:
            selfVariance = (self.uncertainty)**2
            if ((nx0 % nx) == 0) and ((ny0 % ny) == 0):
                # Handle integer downsampling
                outVariance = selfVariance.reshape(sh).sum(-1).sum(1)

                # Check if total flux conservation was requested.
                # If not, then multiply by the pixel size ratio.
                if not total: outVariance *= pixRatio

            elif ((nx % nx0) == 0) and ((ny % ny0) == 0):
                # Handle integer upsampling
                outVariance = np.kron(
                    selfVariance,
                    np.ones((ny//ny0, nx//nx0))
                )

                # Check if total flux conservation was requested.
                # If not, then divide by the pixel size ratio.
                if total: outVariance /= pixRatio

            # Convert the uncertainty into the correct class for NDDataArray
            outUncert = StdDevUncertainty(np.sqrt(outVariance))
        else:
            # Handle the no-uncertainty situation
            outUncert = None

        # Construct the output NDDataArray
        outData = NDDataArray(
            rebinArr,
            uncertainty=outUncert,
            unit=self._BaseImage__fullData.unit,
            wcs=self._BaseImage__fullData.wcs
        )

        # Return a copy of the image with a rebinned array
        outImg = self.copy()
        outImg._BaseImage__fullData = outData

        # Update the header values
        outHead = self.header.copy()
        outHead['NAXIS1'] = nx
        outHead['NAXIS2'] = ny

        # Store the header in the output image
        outImg._BaseImage__header = outHead

        # Update the binning attribute to match the new array
        outImg._BaseImage__binning = (
            outImg.binning[0]/xratio,
            outImg.binning[1]/yratio
        )

        # Return the updated image object
        return outImg

    @lru_cache()
    def sigma_clipped_stats(self, nsamples=1000, **kwargs):
        """
        Calculate sigma-clipped statistics on the provided data.

        See astropy.stats.sigma_clipped_stats for full documentation.

        This is a convenience method, and it caches the result for faster
        performance.

        Parameters
        ----------
        nsamples : int or float, optional, default: 1000
            The number of samples to draw from the image. If there are less than
            nsamples pixels in the image, then nsamples pixels will be randomly
            drawn (with replacement) to speed up computation of the image
            statistics.

        **kwargs
            Also takes all keyword arguments and passes them to
            astropy.stats.sigma_clipped_stats

        Returns
        -------
        mean, median, stddev : float
            The mean, median, and standard deviation of the sigma-clipped data.
        """
        if nsamples < self.data.size:
            # Grab the shape of the image
            ny, nx   = self.shape

            # Compute the randomly sampled pixel indices
            idx = np.round((nx-1)*np.random.rand(nsamples)).astype(int)
            idy = np.round((ny-1)*np.random.rand(nsamples)).astype(int)

            # Extract the randomly sampled indices
            tmpArr = self.data[idy, idx]
        else:
            tmpArr = self.data.flatten()

        return sigma_clipped_stats(tmpArr, **kwargs)

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
        RA_cen, Dec_cen = self.wcs.all_pix2world([x_cen], [y_cen], 0)

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
        fig  : matplotlib.figure.Figure
            The Figure instance in which axes will be stored.

        axes : WCSaxes or matplotlib.axes.Axes
            The Axes instance in which the data will be displayed.
        """
        # Handle the trivial case
        if axes is not None:
            # Extract the figure from the axes instance and return
            fig  = axes.figure

            return (fig, axes)

        if self._BaseImage__fullData.wcs is None:
            # Handle the matplotlib case
            # Initalize a Figure and Axes instance
            fig = plt.figure(figsize = (8,8))
            axes = fig.add_subplot(1,1,1)

            return (fig, axes)

        # If there is a valid WCS in this image, then build the
        # axes using the WCS for the projection
        fig = plt.figure(figsize = (8,8))
        axes = fig.add_subplot(1, 1, 1, projection=self.wcs)

        # Set the axes linewidth
        axes.coords.frame.set_linewidth(2)

        # Label the axes establish minor ticks.
        RA_ax  = axes.coords[0]
        Dec_ax = axes.coords[1]
        RA_ax.set_axislabel('RA [J2000]',
                            fontsize=12, fontweight='bold')
        Dec_ax.set_axislabel('Dec [J2000]',
                             fontsize=12, fontweight='bold', minpad=-0.4)

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

        return (fig, axes)

    def _normalize_and_stretch_image(self, stretch=None, vmin=None, vmax=None):
            """
            Builds a normalized and stretched version of the image for display.

            Parameters
            ----------
            data : numpy.ndarray
                The image data to be displayed. This data is used to compute
                reasonable normalization ranges if vmin or vmax are not
                supplied.

            stretch : str
                The stretching function to apply to the image

            vmin : int or float
                The bottom of the value display range.

            vmax : int or float
                Thetop of the value display range

            Returns
            -------
            norm : astropy.visualization.ImageNormalize
                The normalizing object to apply to the image when displayed
            """
            if stretch.lower() == 'linear':
                return ImageNormalize(
                    self.data,
                    interval=ZScaleInterval(),
                    vmin=vmin,
                    vmax=vmax,
                    stretch=LinearStretch()
                )

            if stretch.lower() == 'log':
                return ImageNormalize(
                    self.data,
                    interval=AsymmetricPercentileInterval(30, 99),
                    vmin=vmin,
                    vmax=vmax,
                    stretch=LogStretch()
                )

            if stretch.lower() == 'asinh':
                return ImageNormalize(
                    self.data,
                    interval=ZScaleInterval(),
                    vmin=vmin,
                    vmax=vmax,
                    stretch=AsinhStretch()
                )

    def show(self, axes=None, cmap='viridis', vmin=None, vmax=None,
            origin='lower', noShow=False, stretch='linear', **kwargs):
        """
        Displays an interactive image to the user.

        Parameters
        ----------
        axes : `matplotlib.axes` instance, optional, default: None
            If set, then the image will be displayed inside the provided axes.

        cmap : str or matplotlib Colormap instance, optional, default: `viridis`
            Set to a string naming a provided matplotlib colormap or a custom
            Colormap instance.

        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your settings
            for `vmin` and `vmax` will be ignored.

        origin : str, optional, default: `lower`
            Sets the location of the coordinates origin.

            `upper`
                places the origin at the upper left of the axes.
            `lower`
                places the origin at the lower left of the axes.

        noShow : bool, optional, default: False
            If True, then the figure, axes, and AxesImage instances are
            generated and returned to the user, but the image is not shown on
            the current display.

        stretch : str, optional, default: `linear`
            Sets the stretching function for displaying the array

            'linear'
                Displays data on a linear scale.
            'log'
                Displays data on a log scale.
            'asinh'
                Dispalys data on a hyperbolic arcsin scale.

        Other parameters
        ----------------
        kwargs : `~matplotlib.artist.Artist` properties.

        Returns
        -------
        image : `~matplotlib.image.AxesImage`
            The AxesImage instance of the displayed image
        """
        # Check the vmin value
        if not issubclass(type(vmin), (type(None),
            int, np.int, np.int8, np.int16, np.int32, np.int64,
            float, np.float, np.float16, np.float32, np.float64)):
            raise TypeError('`vmin` must be an int or float')

        # Check the vmax value
        if not issubclass(type(vmax), (type(None),
            int, np.int, np.int8, np.int16, np.int32, np.int64,
            float, np.float, np.float16, np.float32, np.float64)):
            raise TypeError('`vmax` must be an int or float')

        # Check if the provided `stretch` keyword argument is a good one.
        if type(stretch) is not str:
            raise TypeError('`stretch` must be a string')

        # TODO: Add asinh stretching
        if stretch not in ['linear', 'log']:
            raise ValueError('The provided `stretch` keyword is not recognized')

        # First, determine if the current state is "interactive"
        isInteractive = mpl.is_interactive()

        # Compute a good normalization for this image
        norm = self._normalize_and_stretch_image(
            stretch=stretch,
            vmin=vmin,
            vmax=vmax
        )

        # Construct the figure and axes to be displayed
        fig, axes = self._build_axes(axes)

        # Set the axes line properties to be thicker
        for axis in ['top','bottom','left','right']:
            axes.spines[axis].set_linewidth(4)

        # Finally display the image to the user!
        image = axes.imshow(self.data, origin=origin, norm = norm,
                           cmap=cmap, **kwargs)

        # Display the image to the user, if requested
        if not noShow:
            plt.ion()
            fig.show()

            # Reset the original "interactive" state
            if not isInteractive:
                plt.ioff()

        # Store the axIm instance and return to the user
        self.__image = image

        # Return the graphics objects to the user
        return image

    ######
    # USEFUL CODE FOR FIGURE MANAGEMENT
    ######

    # TODO
    # Test if figure exists

    # if self.fig is None or not plt.fignum_exists(self.fig.number):



    # If the figure number doesn't exist, then assume the figure was destroyed,
    # and must be recreated...
    #
    # The following code is a neat way to "resurrect" a dead figure.

    # def show_figure(fig):
    #
    #     # create a dummy figure and use its
    #     # manager to display "fig"
    #
    #     dummy = plt.figure()
    #     new_manager = dummy.canvas.manager
    #     new_manager.canvas.figure = fig
    #     fig.set_canvas(new_manager.canvas)
