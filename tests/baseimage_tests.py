# Scipy imports
import numpy as np

# Import testing utitities
from nose.tools import *
from nose import with_setup

# Astropy imports
import astropy.units as u
from astropy.io import fits

# Import the base class
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


# You can handle equality statements of multiple array elements using the
# "a.all()" method.
# e.g. "assert (arr1 == arr2).all()"

# Setup test  arrays
arr0 = np.arange(300*300).reshape((300, 300)) + 100.0
forcedMode = np.median(arr0)
arr1 = arr0
arr1[100:200, 100:200] = forcedMode
arr2 = 2*arr0
arr3 = arr0 - forcedMode
sig1 = 0.025*arr1
sig2 = 3*sig1
head1 = fits.Header()

# Write the basic unit tests

def test_has_angle_units():
    img1 = ai.baseimage.BaseImage(arr1, properties={'unit':'deg'})

    assert img1.has_angle_units

    img1 = ai.baseimage.BaseImage(arr1, properties={'unit':'rad'})

    assert img1.has_angle_units

    img1 = ai.baseimage.BaseImage(arr1, properties={'unit':'arcsec'})

    assert img1.has_angle_units

def test_has_dimensionless_units():
    img1 = ai.baseimage.BaseImage(arr1, properties={'unit':'pc'})
    img2 = img1.convert_units_to('m')

    img3 = img2/img1

    assert img3.has_dimensionless_units

def test_convert_units_to():
    img1 = ai.baseimage.BaseImage(arr1, properties={'unit':'deg'})
    img2 = img1.convert_units_to('arcsec')

    assert (np.abs(img2.data/(arr1*3600) - 1.0) < 1e-4).all()

    img1 = ai.baseimage.BaseImage(arr1, properties={'unit':'pc'})
    img2 = img1.convert_units_to('m')

    assert (np.abs(img2.data/((arr1*u.pc).to(u.m)).value - 1.0) < 1e-4).all()

def test_pos():
    # Build test images
    img1 = ai.baseimage.BaseImage(arr1)

    # Positive test image
    img2 = +img1

    # Positive numpy arrays
    res_a = +arr1

    # Assert that the numpy and BaseImage yield the same result
    assert (img2.data == res_a).all()

def test_neg():
    # Build test images
    img1 = ai.baseimage.BaseImage(arr1)

    # Negate the test image
    img2 = -img1

    # Negate the numpy arrays
    res_a = -arr1

    # Assert that the numpy and BaseImage yield the same result
    assert (img2.data == res_a).all()

def test_abs():
    # Build test images
    img1 = ai.baseimage.BaseImage(arr3)

    # Absolute value test image
    img2 = np.abs(img1)

    # Absolute value numpy arrays
    res_a = np.abs(arr3)

    # Assert that the numpy and BaseImage yield the same result
    assert (img2.data == res_a).all()

def test_add():
    # Build test images
    img1 = ai.baseimage.BaseImage(arr1)
    img2 = ai.baseimage.BaseImage(arr2)

    # Add the images together
    img3a = img1 + img2
    img3b = img2 + img1

    # Add the numpy arrays
    res_a = arr1 + arr2
    res_b = arr2 + arr1

    # Assert that the numpy and BaseImage yield the same result
    assert (img3a.data == res_a).all()
    assert (img3b.data == res_b).all()

    # Test scalar addition
    img3c = img1 + 3
    res_c = arr1 + 3
    assert (img3c.data == res_c).all()

def test_subtract():
    # Build test images
    img1 = ai.baseimage.BaseImage(arr1)
    img2 = ai.baseimage.BaseImage(arr2)

    # Subtract the images
    img3a = img1 - img2
    img3b = img2 - img1

    # Subtract the numpy arrays
    res_a = arr1 - arr2
    res_b = arr2 - arr1

    # Assert that the numpy and BaseImage yield the same result
    assert (img3a.data == res_a).all()
    assert (img3b.data == res_b).all()

    # Test scalar subtraction
    img3c = img1 - 3
    res_c = arr1 - 3
    assert (img3c.data == res_c).all()

def test_multiply():
    # Build test images
    img1 = ai.baseimage.BaseImage(arr1)
    img2 = ai.baseimage.BaseImage(arr2)

    # Multiply the images together
    img3a = img1 * img2
    img3b = img2 * img1

    # Subtract the numpy arrays
    res_a = arr1 * arr2
    res_b = arr2 * arr1

    # Assert that the numpy and BaseImage yield the same result
    assert (img3a.data == res_a).all()
    assert (img3b.data == res_b).all()

    # Test scalar multiplication
    img3c = img1 * 3
    res_c = arr1 * 3
    assert (img3c.data == res_c).all()

def test_divide():
    # Build test images
    img1 = ai.baseimage.BaseImage(arr1)
    img2 = ai.baseimage.BaseImage(arr2)

    # Multiply the images together
    img3a = img1 / img2
    img3b = img2 / img1

    # Subtract the numpy arrays
    res_a = arr1 / arr2
    res_b = arr2 / arr1

    # Assert that the numpy and BaseImage yield the same result
    assert (img3a.data == res_a).all()
    assert (img3b.data == res_b).all()

    # Test scalar subtraction
    img3c = img1 / 3
    res_c = arr1 / 3
    assert (img3c.data == res_c).all()

def test_pow():
    # Build a test image
    img1 = ai.baseimage.BaseImage(arr1)

    # Exponentiate the image
    img2 = img1**3

    # Exponentiate the numpy array
    res_a = arr1**3

    # Assert that the numpy and BaseImage yield the same result
    assert (img2.data == res_a).all()

def test_astype():
    # Build the test image
    img1 = ai.baseimage.BaseImage(arr1)

    # Check that the array and image data type agree
    assert_equal(img1.dtype,  arr1.dtype)

    # Recast the image into a different type
    img2 = img1.astype(np.int, copy=True)

    # Check that the new types are the expected types
    assert_equal(img1.dtype,  arr1.dtype)
    assert_equal(img2.dtype, np.dtype(np.int))

def test_copy():
    # Tests the copy method
    properties = {
        'airmass': 1.0,
        'binning': (2, 2),
        'date': '2017-03-21T12:00:00',
        'dec': 3.123456,
        'expTime': 2.5,
        'filter': 'V',
        'instrument': 'PRISM',
        'ra': 4.123456,
        'units': 'electron',
    }
    img1 = ai.baseimage.BaseImage(arr1, properties=properties)
    img2 = img1.copy()
    attList  = [
        'airmass',
        'binning',
        'data',
        'date',
        'dec',
        'dtype',
        'expTime',
        'filter',
        'header',
        'height',
        'instrument',
        'ra',
        'shape',
        'unit',
        'width'
    ]

    # Loop through the list of attributes
    for a in attList:
        # Grab each attribute
        a1, a2 = getattr(img1, a), getattr(img2, a)

        # Check if the attributes are equal
        try:
            assert a1 == a2

        # If that fails, convert to arrays and check if all ELEMENTS are equal
        except (AssertionError, ValueError):
            assert (np.array(a1) == np.array(a2)).all()

def test_write():
    # Test whether the writing function is working
    img1 = ai.baseimage.BaseImage(arr1, uncertainty=arr2)

    img1.write('img1.fits', clobber=True)

def test_read():
    # Test whether reading in that previously written image works
    img1 = ai.baseimage.BaseImage.read('img1.fits')

    # Test if the read in data matches the expected saved data
    assert (img1.data == arr1).all()

    # Test if the uncertainty was actually saved to disk and read in
    assert (img1._BaseImage__fullData.uncertainty is None)

    # Delete the test file
    import os
    rmProc = os.remove('img1.fits')
