# Scipy imports
import numpy as np

# Import testing utitities
from nose.tools import *
from nose import with_setup

# Astropy imports
from astropy.io import fits
import astropy.units as u

# AstroImage imports
import astroimage as ai

###
# I have not yet learned how to use setup and tearndown functions usefully
###
# # Define the setup and tear down functions
# def setup_module(module):
#     print ("") # this is to get a newline after the dots
#     print ("setup_module before anything in this file")
#
# def teardown_module(module):
#     print ("teardown_module after everything in this file")
#
# def my_setup_function():
#     print ("my_setup_function")
#
# def my_teardown_function():
#     print ("my_teardown_function")
#
# @with_setup(my_setup_function, my_teardown_function)

# Setup practice arrays
arr0 = np.arange(300*300).reshape((300, 300)) + 100.0
forcedMode = np.median(arr0)
arr1 = arr0
arr1[100:200, 100:200] = forcedMode
arr2 = 2*arr0
arr3 = arr0 - forcedMode
sig1 = 0.025*arr1
sig2 = 3*sig1
head1 = fits.Header({'PRESCAN': 10, 'POSTSCAN':10})

@raises(AttributeError)
def test_apply_overscan_correction_without_overscans():
    # Test whether an attempt to apply an overscan correction to images
    # with no overscans raises an error
    img1 = ai.raw.RawImage(arr1)

    # Attempt to apply an overscan correction
    img1._apply_overscan_correction()

def test_apply_overscan_correction():
    # Test the "apply_overscan_correction" of the RawImage meta-class
    img1 = ai.raw.RawImage(arr1, header=head1)

    # # Make sure that the prescan ond postscan regions are the correct width
    # assert_equal(img1.prescanWidth, head1['PRESCAN']+40)
    # assert_equal(img1.overscanWidth, head1['POSTSCAN']+40)

    # Attempt to apply an overscan correction and test boolean flag
    img1._apply_overscan_correction()
    assert img1.overscanCorrected

@raises(TypeError)
def test_process_image_parsing_bad_bias():
    # Test whether an passing the incorrect image types to the "process_image"
    # method raises an error as it should

    # Build the basic images
    rawSci  = ai.raw.RawScience(arr1, properties={'obsType':'OBJECT', 'gain': 3.2})
    rawBias = ai.raw.RawBias(arr1, properties={'obsType':'BIAS'})

    # Attempt to process the science image
    rawSci.process_image(
        bias=rawBias
    )

@raises(TypeError)
def test_process_image_parsing_bad_dark():
    # Test whether an passing the incorrect image types to the "process_image"
    # method raises an error as it should

    # Build the basic images
    rawSci  = ai.raw.RawScience(arr1, properties={'obsType':'OBJECT', 'gain': 3.2})
    rawDark = ai.raw.RawDark(arr1, properties={'obsType':'DARK'})
    masterBias = ai.reduced.MasterBias(arr1, uncertainty=sig1, properties={'obsType':'BIAS'})

    # Attempt to process the science image
    rawSci.process_image(
        bias=masterBias,
        dark=rawDark
    )

@raises(TypeError)
def test_process_image_parsing_bad_flat():
    # Test whether an passing the incorrect image types to the "process_image"
    # method raises an error as it should

    # Build the basic images
    rawSci  = ai.raw.RawScience(arr1, properties={'obsType':'OBJECT', 'gain': 3.2})
    rawFlat = ai.raw.RawFlat(arr1, properties={'obsType':'FLAT'})
    masterBias = ai.reduced.MasterBias(arr1, uncertainty=sig1, properties={'obsType':'BIAS'})
    masterDark = ai.reduced.MasterDark(arr1, uncertainty=sig1, properties={'obsType':'DARK'})

    # Attempt to process the science image
    rawSci.process_image(
        bias=masterBias,
        dark=masterDark,
        flat=rawFlat
    )

def test_process_image_with_good_inputs():
    # Test whether an passing the incorrect image types to the "process_image"
    # method raises an error as it should

    # Build the basic images
    rawSci     = ai.raw.RawScience(
        arr1 + 5,
        properties={'expTime': 3.0, 'obsType':'OBJECT', 'gain': 3.2}
    )
    masterBias = ai.reduced.MasterBias(
        arr1,
        uncertainty=sig1,
        properties={'obsType':'BIAS'}
    )
    masterDark = ai.reduced.MasterDark(
        arr1 - np.median(arr1),
        uncertainty=sig1,
        properties={'obsType':'DARK'}
    )
    masterFlat = ai.reduced.MasterFlat(
        arr1/np.median(arr1),
        uncertainty=sig1,
        properties={'obsType':'FLAT'}
    )

    # Attempt to process the science image
    rawSci.process_image(
        bias=masterBias,
        dark=masterDark,
        flat=masterFlat
    )

def test_bias_readNoise():
    # Test the readNoise property of the RawBias frames
    noiseArr = np.random.randn(300*300).reshape((300,300))
    img1 = ai.raw.RawBias(noiseArr, header=head1, properties={'obsType': 'BIAS'})

    # Get the read-noise
    assert img1.readNoise.unit is u.adu
    assert (np.abs(img1.readNoise.value - 1) < 0.01)

def test_flat_mode():
    # Test the mode property of the RawFlat frames
    img1 = ai.raw.RawFlat(arr1, properties={'obsType': 'FLAT'})

    # Compute a mode-normalized image
    tmpMode = img1.mode
    img2    = img1/tmpMode

    # Force a header update
    img2._properties_to_header()

    # Check that the result is unitless
    assert img2.unit is u.dimensionless_unscaled

    # Permit a 5% error
    assert (np.abs(img2.mode.value - 1.0) < 0.05)
