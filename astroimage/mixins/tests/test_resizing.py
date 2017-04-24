# Scipy imports
import numpy as np

# Import testing utitities
from nose.tools import *
from nose import with_setup

# Astropy imports
from astropy.io import fits
from astropy.coordinates import match_coordinates_sky, SkyCoord
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

# Setup test  arrays
arr0 = np.arange(300*300).reshape((300, 300)) + 100.0
forcedMode = np.median(arr0)
arr1 = arr0
arr1[100:200, 100:200] = forcedMode
arr2 = 2*arr0
arr3 = arr0 - forcedMode
sig1 = 0.025*arr1
sig2 = 3*sig1
head1 = fits.Header({'BUNIT': 'adu'})

# Read in an ACTUAL image of M104
realImg1 = ai.reduced.ReducedScience.read('.\\tests\\M104_V_I.fits')


def test_pad():
    # Use the real image of M104 (contains WCS)
    img1 = realImg1.copy()

    # Pad the image
    img2 = img1.pad(((10, 0), (10, 0)), 'constant')

    # Check if the shape is the expected shape
    assert_equal(img2.shape, (img1.shape[0] + 10, img1.shape[1] + 10))

    # Check if the first 10 pixels are all zeros
    assert_equal(np.sum(img2.data[0:10, 0:10]), 0)

def test_crop():
    # Use the real image of M104 (contains WCS)
    img1 = realImg1.copy()

    # Crop the image
    img2 = img1.crop((100, 100), (600, 600))

    # Assert that the final shape should be the expected shape
    assert_equal(img2.shape, (500, 500))

    # Assert that the arrays are what is expected
    assert (img2.data == img1.data[100:600, 100:600]).all()


def test_rebin():
    # Initalize a basic image
    img1 = ai.reduced.ReducedScience(
        arr1,
        uncertainty=sig1,
        properties={'binning': (2,2)}
    )

    # Make a copy of that image
    img2 = img1.copy()

    # Set the binning along each axis
    dy,  dx  = 3, 3
    ny0, nx0 = img1.shape
    ny,  nx  = ny0//dy, nx0//dx

    # Perform the rebinning
    img1a = img1.rebin((ny, nx), total=True)
    img2a = img2.rebin((ny, nx), total=False)

    # Compute the expected output
    sh = (ny, dx,
          nx, dy)

    res1 = arr1.reshape(sh).sum(-1).sum(1)
    res2 = res1/(dx*dy)

    # Check if the output matches the expected result
    assert (np.abs((img1a.data/res1) - 1.0) < 1e-4).all()
    assert (np.abs((img2a.data/res2) - 1.0) < 1e-4).all()
    assert_equal(img1a.binning, (2*dy, 2*dx))

    # Initalize a basic image
    img1 = ai.reduced.ReducedScience(arr1, uncertainty=sig1, properties={'binning':(2,2)})

    # Make a copy of that image
    img2 = img1.copy()

    # Set the binning along each axis
    dy,  dx  = 0.5, 0.5
    ny0, nx0 = img1.shape
    ny,  nx  = np.int(ny0//dy), np.int(nx0//dx)

    # Perform the rebinning
    img1a = img1.rebin((ny, nx), total=True)
    img2a = img2.rebin((ny, nx), total=False)

    # Compute the expected output
    sh = (ny, dx,
          nx, dy)

    res2 = np.kron(
        arr1,
        np.ones((ny//ny0, nx//nx0))
    )
    res1 = res2*(dx*dy)

    # Check if the output matches the expected result
    assert (np.abs((img1a.data/res1) - 1.0) < 1e-4).all()
    assert (np.abs((img2a.data/res2) - 1.0) < 1e-4).all()
    assert_equal(img1a.binning, (2*dy, 2*dx))
    assert_equal(np.sum(res1), np.sum(img1.data))

    # Check if the output uncertainty has the expected value
    assert (img1a.uncertainty[::2,::2] == sig1*np.sqrt(dx*dy)).all()

    # Test if REBINNING back to original size produces the expected result
    dy, dx = 2, 2
    ny0, nx0 = img1a.shape
    ny,  nx  = np.int(ny0//dy), np.int(nx0//dx)
    img1b = img1a.rebin((ny, nx), total=True)

    # Check if the output uncertainty has the expected value
    print(img1b.uncertainty)
    print(sig1)
    assert (np.abs((img1b.uncertainty/sig1) - 1.0) < 1e-4).all()
