# Import numpy for making test number stuff
import numpy as np

# Import testing utitities
from nose.tools import *
from nose import with_setup

# import astropy Header class
from astropy.io.fits import Header

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
def test_pos():
    # Built test images
    img1 = AstroImage()

    # Builg test arrays
    arr1 = np.array([5])

    # Built the test uncertainties
    sig1 = np.array([2])

    # Populate the array attributes
    img1.arr = arr1
    img1.sigma = sig1

    # Negate the test image
    img2 = +img1

    # Negate the numpy arrays
    res_a = +arr1

    # Compute expected uncertainties
    sig_a = sig1

    # Assert that the numpy and AstroImage yield the same result
    assert_equal(img2.arr, res_a)
    assert_equal(img2.sigma, sig_a)

def test_neg():
    # Built test images
    img1 = AstroImage()

    # Builg test arrays
    arr1 = np.array([5])

    # Built the test uncertainties
    sig1 = np.array([2])

    # Populate the array attributes
    img1.arr = arr1
    img1.sigma = sig1

    # Negate the test image
    img2 = -img1

    # Negate the numpy arrays
    res_a = -arr1

    # Compute expected uncertainties
    sig_a = sig1

    # Assert that the numpy and AstroImage yield the same result
    assert_equal(img2.arr, res_a)
    assert_equal(img2.sigma, sig_a)

def test_abs():
    # Built test images
    img1 = AstroImage()
    img2 = AstroImage()

    # Builg test arrays
    arr1 = np.array([-5])

    # Built the test uncertainties
    sig1 = np.array([2])

    # Populate the array attributes
    img1.arr = arr1
    img1.sigma = sig1

    # Negate the test image
    img2 = np.abs(img1)

    # Negate the numpy arrays
    res_a = np.abs(arr1)

    # Compute expected uncertainties
    sig_a = sig1

    # Assert that the numpy and AstroImage yield the same result
    assert_equal(img2.arr, res_a)
    assert_equal(img2.sigma, sig_a)

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

    # Subtract the numpy arrays
    res_a = arr1 + arr2
    res_b = arr2 + arr1

    # Compute expected uncertainties
    sig_a = np.sqrt(sig1**2 + sig2**2)
    sig_b = np.sqrt(sig2**2 + sig1**2)

    # Assert that the numpy and AstroImage yield the same result
    assert_equal(img3a.arr, res_a)
    assert_equal(img3b.arr, res_b)
    assert_equal(img3a.sigma, sig_a)
    assert_equal(img3b.sigma, sig_b)

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

    # Subtract the images
    img3a = img1 - img2
    img3b = img2 - img1

    # Subtract the numpy arrays
    res_a = arr1 - arr2
    res_b = arr2 - arr1

    # Compute expected uncertainties
    sig_a = np.sqrt(sig1**2 + sig2**2)
    sig_b = np.sqrt(sig2**2 + sig1**2)

    # Assert that the numpy and AstroImage yield the same result
    assert_equal(img3a.arr, res_a)
    assert_equal(img3b.arr, res_b)
    assert_equal(img3a.sigma, sig_a)
    assert_equal(img3b.sigma, sig_b)

def test_multiply():
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

    # Multiply the images together
    img3a = img1 * img2
    img3b = img2 * img1

    # Subtract the numpy arrays
    res_a = arr1 * arr2
    res_b = arr2 * arr1

    # Compute expected uncertainties
    sig_a = np.abs(res_a)*np.sqrt((sig1/arr1)**2 + (sig2/arr2)**2)
    sig_b = np.abs(res_b)*np.sqrt((sig2/arr2)**2 + (sig1/arr1)**2 )

    # Assert that the numpy and AstroImage yield the same result
    assert_equal(img3a.arr, res_a)
    assert_equal(img3b.arr, res_b)
    assert_equal(img3a.sigma, sig_a)
    assert_equal(img3b.sigma, sig_b)

def test_divide():
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

    # Multiply the images together
    img3a = img1 / img2
    img3b = img2 / img1

    # Subtract the numpy arrays
    res_a = arr1 / arr2
    res_b = arr2 / arr1

    # Compute expected uncertainties
    sig_a = np.abs(res_a)*np.sqrt((sig1/arr1)**2 + (sig2/arr2)**2)
    sig_b = np.abs(res_b)*np.sqrt((sig2/arr2)**2 + (sig1/arr1)**2 )

    # Assert that the numpy and AstroImage yield the same result
    assert_equal(img3a.arr, res_a)
    assert_equal(img3b.arr, res_b)
    assert_equal(img3a.sigma, sig_a)
    assert_equal(img3b.sigma, sig_b)

def test_const_pow1():
    # Built test images
    img1 = AstroImage()

    # Builg test arrays
    arr1 = np.array([5])

    # Built the test uncertainties
    sig1 = np.array([2])

    # Populate the array attributes
    img1.arr = arr1
    img1.sigma = sig1

    # Exponentiate the image (two different ways!)
    img2 = img1**3

    # Subtract the numpy arrays
    res_a = arr1**3

    # Compute expected uncertainties
    sig_a = np.abs(3*(arr1**2)*sig1)

    # Assert that the numpy and AstroImage yield the same result
    assert_equal(img2.arr, res_a)
    assert_equal(img2.sigma, sig_a)

# def test_const_pow2():
#     # Built test images
#     img1 = AstroImage()
#
#     # Builg test arrays
#     arr1 = np.array([5])
#
#     # Built the test uncertainties
#     sig1 = np.array([2])
#
#     # Populate the array attributes
#     img1.arr = arr1
#     img1.sigma = sig1
#
#     # Exponentiate the image (two different ways!)
#     img2 = 3**img1
#
#     # Subtract the numpy arrays
#     res_a = 3**arr1
#
#     # Compute expected uncertainties
#     sig_a = ???
#
#     # Assert that the numpy and AstroImage yield the same result
#     assert_equal(img2.arr, res_a)
#     assert_equal(img2.sigma, sig_a)

def test_img_pow():
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

    # Multiply the images together
    img3a = img1**img2
    img3b = img2**img1

    # Subtract the numpy arrays
    res_a = arr1**arr2
    res_b = arr2**arr1

    # Compute expected uncertainties
    sig_a = np.abs(res_a)*np.sqrt((arr2*(sig1/arr1))**2 + (np.log(arr1)*sig2)**2)
    sig_b = np.abs(res_b)*np.sqrt((arr1*(sig2/arr2))**2 + (np.log(arr2)*sig1)**2)

    # Assert that the numpy and AstroImage yield the same result
    assert_equal(img3a.arr, res_a)
    assert_equal(img3b.arr, res_b)
    assert_equal(img3a.sigma, sig_a)
    assert_equal(img3b.sigma, sig_b)

def test_sqrt():
    # Built test images
    img1 = AstroImage()

    # Builg test arrays
    arr1 = np.array([5])

    # Built the test uncertainties
    sig1 = np.array([2])

    # Populate the array attributes
    img1.arr = arr1
    img1.sigma = sig1

    # Exponentiate the image (two different ways!)
    img2 = np.sqrt(img1)
    img3 = img1**0.5

    # Subtract the numpy arrays
    res_a = np.sqrt(arr1)

    # Compute expected uncertainties
    sig_a = 0.5*sig1/res_a

    # Assert that the numpy and AstroImage yield the same result
    assert_equal(img2.arr, res_a)
    assert_equal(img2.sigma, sig_a)

    # Assert that sqrt and __pow(self, 0.5)__ yeild the same result
    assert_equal(img2.arr, img3.arr)
    assert_equal(img2.sigma, img3.sigma)

def test_null():
    # This should test if a "null" image can be generated and manipulated
    # Build a null image
    nullImg     = AstroImage()
    nullImg.arr = np.array([0])

    # Build a simple image
    img1       = AstroImage()
    img1.arr   = np.array([5])
    img1.sigma = np.array([2])

    # Add the null and simple image and see if the result is what is expected
    img2a = img1 + nullImg
    img2b = nullImg + img1

    assert_equal(img2a.arr, img1.arr)
    assert_equal(img2a.sigma, img1.sigma)
    assert_equal(img2b.arr, img1.arr)
    assert_equal(img2b.sigma, img1.sigma)

    # Now try using the += and -= operators...
    nullImg1  = nullImg.copy()
    nullImg1 += img1
    assert_equal(nullImg1.arr, img1.arr)
    assert_equal(nullImg1.sigma, img1.sigma)

    nullImg1  = nullImg.copy()
    nullImg1 -= img1
    assert_equal(nullImg1.arr, -img1.arr)
    assert_equal(nullImg1.sigma, img1.sigma)

# THESE SHOULD REALLY BE MORE GENERAL...
# Right now they're only testing a small set of special cases.
def test_shift1():
    # Test if an integer pixel shift yields the results expected
    dx, dy = 1, 1

    img = AstroImage()
    img.arr = np.zeros((5,5), dtype=float)
    img.arr[2,2] = 1.0
    img.sigma = np.zeros((5,5), dtype=float)
    img.sigma[2,2] = 1.0

    res = np.zeros((5,5), dtype=float)
    res[3,3] = 1.0
    sig = np.zeros((5,5), dtype=float)
    sig[3,3] = 1.0

    img.shift(dx, dy, padding=0.0)
    assert (img.arr == res).all()
    assert (img.sigma == sig).all()

def test_shift2():
    # Test if an integer pixel shift yields the results expected
    dx, dy = 1.5, 1.5

    # Initalize a basic image
    img = AstroImage()
    img.arr = np.zeros((5,5), dtype=float)
    img.arr[2,2] = 1.0
    img.sigma = np.zeros((5,5), dtype=float)
    img.sigma[2,2] = 1.0

    # Compute the expected output
    res = np.zeros((5,5), dtype=float)
    res[3:5,3:5] = 0.25
    sig = np.zeros((5,5), dtype=float)
    sig[3:5,3:5] = 0.25

    # Perform the image shift and check if it matches the expected output
    img.shift(dx, dy, padding=0.0)
    assert (img.arr == res).all()
    assert (img.sigma == sig).all()

def test_rebin():
    # Test if an rebinning an image yields the expected results
    binX, binY = 2, 2

    # Initalize a basic image
    img1 = AstroImage()
    img1.binning = (1, 1)
    img1.arr = np.ones((6,6), dtype=float)
    img1.arr[2:4,2:4] = 2.0
    img1.sigma = np.ones((6,6), dtype=float)
    img1.sigma[2:4,2:4] = 2.0

    # Make a copy of that image
    img2 = img1.copy()

    # Perform the rebinning
    img1.rebin(3, 3, total=False)
    img2.rebin(3, 3, total=True)

    # Compute the expected output
    res1 = np.ones((3,3), dtype=float)
    res1[1,1] = 2.0
    sig1 = 0.5*np.ones((3,3), dtype=float)
    sig1[1,1] = 1.0
    res2 = 4*np.ones((3,3), dtype=float)
    res2[1,1] = 8.0
    sig2 = 2.0*np.ones((3,3), dtype=float)
    sig2[1,1] = 4.0

    # Check if the output matches the expected result
    assert (img1.arr == res1).all()
    assert (img1.sigma == sig1).all()
    assert (img2.arr == res2).all()
    assert (img2.sigma == sig2).all()

def test_scale():
        # Initalize a basic image
        img0 = AstroImage()
        img0.arr = 2.0*np.ones((1,1), dtype=float)
        img0.sigma = 3.0*np.ones((1,1), dtype=float)
        img0.header.update(dict(zip(['BSCALE', 'BZERO'], [3.0, 4.0])))

        # Scale the image using slightly different methods
        img1 = img0.scale(copy=True)
        img2 = img0.copy()
        img2.scale()

        # Compare the results with what is expected
        assert_equal(img1.arr[0,0], 2.0*3.0 + 4.0)
        assert_equal(img1.sigma[0,0], 3.0*3.0)
        assert_equal(img1.arr[0,0], img2.arr[0,0])
        assert_equal(img1.sigma[0,0], img2.sigma[0,0])

        # Now attempt to scale BACK img1 and img2
        img1 = img1.scale(copy=True)
        img2.scale()

        # Test that they are both returned to their original value
        assert_equal(img1.arr[0,0], img0.arr[0,0])
        assert_equal(img1.arr[0,0], img2.arr[0,0])

        # Add an uncertainty to the BSCALE keyword and make sure that is being
        # handled properly
        img0.header['SBSCALE'] = 0.5
        img3 = img0.scale(copy=True)

        # Compare the results with what is expected
        assert_equal(img3.arr[0,0], 2.0*3.0 + 4.0)
        uncert = np.abs(img3.arr[0,0])*np.sqrt(
            (img0.sigma[0,0]/img0.arr[0,0])**2 +
            (img0.header['SBSCALE']/img0.header['BSCALE'])**2)
        assert_equal(img3.sigma[0,0], uncert)
