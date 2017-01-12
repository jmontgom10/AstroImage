# Import numpy for making test number stuff
import numpy as np

# Import testing utitities
from nose.tools import *
from nose import with_setup

# Import the base class
from astroimage.astroimage import AstroImage

###
# I have not yet learned how to use setup and tearndown functions
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

# Write the basic unit tests
def test_add():
    # Built test images
    img1 = AstroImage()
    img2 = AstroImage()

    # Builg test arrays
    arr1 = np.array([5])
    arr2 = np.array([6])

    # Built the test uncertainties
    sig1 = np.array([2])
    sig2 = np.array([3])

    # Populate the array attributes
    img1.arr = arr1
    img2.arr = arr2
    img1.sigma = sig1
    img2.sigma = sig2

    # Add the images together
    img3a = img1 + img2
    img3b = img2 + img1

    # Assert that the numpy additino and image addition should yield the same
    # result
    assert_equal(img3a.arr, np.array([11]))
    assert_equal(img3b.arr, np.array([11]))
    assert_equal(img3a.sigma, np.array([np.sqrt(2**2 + 3**2)]))
    assert_equal(img3b.sigma, np.array([np.sqrt(2**2 + 3**2)]))

def test_subtract():
    # Built test images
    img1 = AstroImage()
    img2 = AstroImage()

    # Builg test arrays
    arr1 = np.array([5])
    arr2 = np.array([6])

    # Built the test uncertainties
    sig1 = np.array([2])
    sig2 = np.array([3])

    # Populate the array attributes
    img1.arr = arr1
    img2.arr = arr2
    img1.sigma = sig1
    img2.sigma = sig2

    # Add the images together
    img3a = img1 - img2
    img3b = img2 - img1

    # Assert that the numpy additino and image addition should yield the same
    # result
    assert_equal(img3a.arr, np.array([-1]))
    assert_equal(img3b.arr, np.array([+1]))
    assert_equal(img3a.sigma, np.array([np.sqrt(2**2 + 3**2)]))
    assert_equal(img3b.sigma, np.array([np.sqrt(2**2 + 3**2)]))
