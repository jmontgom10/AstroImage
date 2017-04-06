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
    img1 = ai.RawImage(arr1)

    # Attempt to apply an overscan correction
    img1._apply_overscan_correction()

def test_apply_overscan_correction():
    # Test the "apply_overscan_correction" of the RawImage meta-class
    img1 = ai.RawImage(arr1, header=head1)

    # Make sure that the prescan ond postscan regions are the correct width
    assert_equal(img1.prescanWidth, head1['PRESCAN']+40)
    assert_equal(img1.overscanWidth, head1['POSTSCAN']+40)

    # Attempt to apply an overscan correction and test boolean flag
    img1._apply_overscan_correction()
    assert img1.overscanCorrected

@raises(TypeError)
def test_process_image_parsing_bad_bias():
    # Test whether an passing the incorrect image types to the "process_image"
    # method raises an error as it should

    # Build the basic images
    rawSci  = ai.RawScience(arr1, properties={'obsType':'OBJECT'})
    rawBias = ai.RawBias(arr1, properties={'obsType':'BIAS'})

    # Attempt to process the science image
    rawSci.process_image(
        bias=rawBias
    )

@raises(TypeError)
def test_process_image_parsing_bad_dark():
    # Test whether an passing the incorrect image types to the "process_image"
    # method raises an error as it should

    # Build the basic images
    rawSci  = ai.RawScience(arr1, properties={'obsType':'OBJECT'})
    rawDark = ai.RawDark(arr1, properties={'obsType':'DARK'})
    masterBias = ai.MasterBias(arr1, uncertainty=sig1, properties={'obsType':'BIAS'})

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
    rawSci  = ai.RawScience(arr1, properties={'obsType':'OBJECT'})
    rawFlat = ai.RawFlat(arr1, properties={'obsType':'FLAT'})
    masterBias = ai.MasterBias(arr1, uncertainty=sig1, properties={'obsType':'BIAS'})
    masterDark = ai.MasterDark(arr1, uncertainty=sig1, properties={'obsType':'DARK'})

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
    rawSci     = ai.RawScience(arr1, properties={'expTime': 3.0, 'obsType':'OBJECT'})
    masterBias = ai.MasterBias(arr1, uncertainty=sig1, properties={'obsType':'BIAS'})
    masterDark = ai.MasterDark(arr1, uncertainty=sig1, properties={'obsType':'DARK'})
    masterFlat = ai.MasterFlat(arr1, uncertainty=sig1, properties={'obsType':'FLAT'})

    # Attempt to process the science image
    rawSci.process_image(
        bias=masterBias,
        dark=masterDark,
        flat=masterFlat
    )

def test_bias_readNoise():
    # Test the readNoise property of the RawBias frames
    img1 = ai.RawBias(arr1, header=head1, properties={'obsType': 'BIAS'})

    # # Apply an overscan correction
    # img1._apply_overscan_correction()

    # Get the read-noise
    assert img1.readNoise.unit is u.adu
    print(img1.readNoise)
    assert (np.abs(img1.readNoise.value - 3535.8088701129286) < 1)

def test_flat_mode():
    # Test the mode property of the RawFlat frames
    img1 = ai.RawFlat(arr1, properties={'obsType': 'FLAT'})

    # Compute a mode-normalized image
    tmpMode = img1.mode
    img2    = img1/tmpMode

    # Force a header update
    img2._properties_to_header()

    # Check that the result is unitless
    assert img2.unit is u.dimensionless_unscaled

    # Permit a 1% error
    assert (np.abs(img2.mode - 1.0) < 0.01)
