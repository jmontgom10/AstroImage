# Scipy imports
import numpy as np

# Import testing utitities
from nose.tools import *
from nose import with_setup

# Astropy imports
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import match_coordinates_sky, SkyCoord
import astropy.units as u

# AstroImage imports
import astroimage as ai

# Read in a real image for testing purposes
realImg1 = ai.reduced.ReducedScience.read('.\\tests\\M104_V_I.fits')

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

# # Read in an ACTUAL image of M104
# realImg1 = ai.reduced.ReducedScience.read('.\\tests\\M104_V_I.fits')
#
#
# # Build a synthenic array of stars
# ny, nx    = 1000, 1000
# np.random.seed(123456789)
# arr1      = 8*np.random.standard_normal((ny, nx)) + 64.0
# ns        = 50
# f0, sx0, sy0 = (
#     1000*np.random.rand(ns) + 500,
#     0.9*nx*np.random.rand(ns) + 0.05*nx,
#     0.9*ny*np.random.rand(ns) + 0.05*ny
# )
#
# # Loop through the star positions and populate them in the array
# yy, xx = np.mgrid[:ny, :nx]
# std0   = 2.2
# for f1, sx, sy in zip(f0, sx0, sy0):
#     arr1 += f1*np.exp(-0.5*
#         (
#         (xx-sx)**2 +
#         (yy-sy)**2
#         ) / (std0)**2
#     )

def test_error_propagation():
    ### Test if a simple set of arrays has its uncertainty correctly propagated.

    img1 = realImg1[400:410,400:410]
    gain = 3.3
    img1.data        = 100*np.ones((10, 10))
    img1.uncertainty = np.sqrt((img1.data)/gain)

    # Construct the list of images
    imgList = []
    for i in range(4):
        img2 = img1.copy()
        img2.data = img1.data + i*0.5
        img2.uncertainty = np.sqrt((img1.data + i)/gain)
        imgList.append(img2)

    # correct for airmass
    imgList = [img.correct_airmass(0.4) for img in imgList]

    # Construct the ImageStack object
    imgStack = ai.utilitywrappers.ImageStack(imgList)

    # Execute the combine_images method
    imgStack.align_images_with_wcs()

    # Compute the combined image
    comboImg = imgStack.combine_images()

    # Now compute the numpy version
    arrStack = np.array([img.data for img in imgStack.imageList])

    # Compute the average array
    avgArr = np.mean(arrStack, axis=0)

    # Assert equality of numpy and ImageStack method
    assert np.all(avgArr == comboImg.data)

    # Compute the expected uncertainty
    uncStack = np.array([img.uncertainty for img in imgStack.imageList])

    # Prop uncertainty
    propagatedUnc = np.sqrt(np.mean(uncStack**2, axis=0))

    # Assert equality of numpy and ImageStack method
    assert np.all(propagatedUnc == comboImg.uncertainty)

# TODO: build basic alignment tests for WCS

# TODO: build basic alignment tests for cross_correlation

# TODO; build basic image combination tests (make images with hot pixels to kill)
