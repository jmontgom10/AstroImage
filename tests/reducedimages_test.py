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

def test_add():
    # Build test images
    img1 = ai.ReducedImage(arr1, uncertainty=sig1, header=head1)
    img2 = ai.ReducedImage(arr2, uncertainty=sig2, header=head1)

    assert (img1.data == arr1).all()
    assert (img1.uncertainty == sig1).all()
    assert (img2.data == arr2).all()
    assert (img2.uncertainty == sig2).all()

    # Add the images together
    img3a = img1 + img2
    img3b = img2 + img1

    # Add the numpy arrays
    res_a = arr1 + arr2
    res_b = arr2 + arr1

    # Compute the expected uncertainty
    sig_a = np.sqrt(sig1**2 + sig2**2)
    sig_b = np.sqrt(sig1**2 + sig2**2)

    # Assert that the numpy and ai.ReducedImage yield the same result
    assert (img3a.data == res_a).all()
    assert (img3b.data == res_b).all()
    assert (img3a.uncertainty == sig_a).all()
    assert (img3b.uncertainty == sig_b).all()

    # Test scalar addition
    img3c = img1 + (3*img1.unit)
    res_c = arr1 + 3
    assert (img3c.data == res_c).all()
    assert (img3c.uncertainty == img1.uncertainty).all()

    # Test error propagation when only one image has uncertainty
    img1a = ai.ReducedImage(arr1, header=head1)
    img3d = img1a + img2
    img3e = img2  + img1a
    assert (img3d.data == res_a).all()
    assert (img3d.uncertainty == img2.uncertainty).all()
    assert (img3e.data == res_b).all()
    assert (img3e.uncertainty == img2.uncertainty).all()

def test_subtract():
    # Build test images
    img1 = ai.ReducedImage(arr1, uncertainty=sig1, header=head1)
    img2 = ai.ReducedImage(arr2, uncertainty=sig2, header=head1)

    assert (img1.data == arr1).all()
    assert (img1.uncertainty == sig1).all()
    assert (img2.data == arr2).all()
    assert (img2.uncertainty == sig2).all()

    # Subtract the images together
    img3a = img1 - img2
    img3b = img2 - img1

    # Subtract the numpy arrays
    res_a = arr1 - arr2
    res_b = arr2 - arr1

    # Compute the expected uncertainty
    sig_a = np.sqrt(sig1**2 + sig2**2)
    sig_b = np.sqrt(sig1**2 + sig2**2)

    # Assert that the numpy and ai.ReducedImage yield the same result
    assert (img3a.data == res_a).all()
    assert (img3b.data == res_b).all()
    assert (img3a.uncertainty == sig_a).all()
    assert (img3b.uncertainty == sig_b).all()

    # Test scalar addition
    img3c = img1 - (3*img1.unit)
    res_c = arr1 - 3
    assert (img3c.data == res_c).all()
    assert (img3c.uncertainty == img1.uncertainty).all()

    # Test error propagation when only one image has uncertainty
    img1a = ai.ReducedImage(arr1, header=head1)
    img3d = img1a - img2
    img3e = img2  - img1a
    assert (img3d.data == res_a).all()
    assert (img3d.uncertainty == img2.uncertainty).all()
    assert (img3e.data == res_b).all()
    assert (img3e.uncertainty == img2.uncertainty).all()

def test_multiply():
    # Build test images
    img1 = ai.ReducedImage(arr1, uncertainty=sig1, header=head1)
    img2 = ai.ReducedImage(arr2, uncertainty=sig2, header=head1)

    assert (img1.data == arr1).all()
    assert (img1.uncertainty == sig1).all()
    assert (img2.data == arr2).all()
    assert (img2.uncertainty == sig2).all()

    # Multiply the images together
    img3a = img1 * img2
    img3b = img2 * img1

    # Multiply the numpy arrays
    res_a = arr1 * arr2
    res_b = arr2 * arr1

    # Compute the expected uncertainty
    sig_a = np.abs(res_a)*np.sqrt((sig1/arr1)**2 + (sig2/arr2)**2)
    sig_b = np.abs(res_b)*np.sqrt((sig2/arr2)**2 + (sig1/arr1)**2)

    # Assert that the numpy and ai.ReducedImage yield the same result
    assert (img3a.data == res_a).all()
    assert (img3b.data == res_b).all()

    # Assert that uncertainty is within 0.25% of the expected result
    assert (np.abs(img3a.uncertainty/sig_a - 1.0) < 0.0025).all()
    assert (np.abs(img3b.uncertainty/sig_b - 1.0) < 0.0025).all()

    # Test scalar addition
    img3c = img1 * 3
    res_c = arr1 * 3
    assert (img3c.data == res_c).all()
    assert (img3c.uncertainty == 3*img1.uncertainty).all()

    # Test error propagation when only one image has uncertainty
    img1a = ai.ReducedImage(arr1, header=head1)
    img3d = img1a * img2
    img3e = img2  * img1a
    assert (img3d.data == res_a).all()
    assert (img3d.uncertainty == img1a.data*img2.uncertainty).all()
    assert (img3e.data == res_b).all()
    assert (img3e.uncertainty == img1a.data*img2.uncertainty).all()

def test_divide():
    # Build test images
    img1 = ai.ReducedImage(arr1, uncertainty=sig1, header=head1)
    img2 = ai.ReducedImage(arr2, uncertainty=sig2, header=head1)

    assert (img1.data == arr1).all()
    assert (img1.uncertainty == sig1).all()
    assert (img2.data == arr2).all()
    assert (img2.uncertainty == sig2).all()

    # Divide the images together
    img3a = img1 / img2
    img3b = img2 / img1

    # Divide the numpy arrays
    res_a = arr1 / arr2
    res_b = arr2 / arr1

    # Compute the expected uncertainty
    sig_a = np.abs(res_a)*np.sqrt((sig1/arr1)**2 + (sig2/arr2)**2)
    sig_b = np.abs(res_b)*np.sqrt((sig2/arr2)**2 + (sig1/arr1)**2)

    # Assert that the numpy and ai.ReducedImage yield the same result
    assert (img3a.data == res_a).all()
    assert (img3b.data == res_b).all()

    # Assert that uncertainty is within 0.25% of the expected result
    assert (np.abs(img3a.uncertainty/sig_a - 1.0) < 0.0025).all()
    assert (np.abs(img3b.uncertainty/sig_b - 1.0) < 0.0025).all()

    # Test scalar addition
    img3c = img1 / 3
    res_c = arr1 / 3
    assert (img3c.data == res_c).all()
    assert (img3c.uncertainty == img1.uncertainty/3).all()

    # Test error propagation when only one image has uncertainty
    img1a = ai.ReducedImage(arr1, header=head1)
    img3d = img1a / img2
    img3e = img2  / img1a

    assert (img3d.data == res_a).all()
    img3dUncertA = img3d.uncertainty
    img3dUncertB = img1a.data*(img2.uncertainty/(img2.data**2))
    assert (np.abs(img3dUncertA/img3dUncertB - 1.0) < 0.0025).all()
    assert (img3e.data == res_b).all()
    img3eUncertA = img3e.uncertainty
    img3eUncertB = img2.uncertainty/img1a.data
    assert (np.abs(img3eUncertA/img3eUncertB - 1.0) < 0.0025).all()


def test_rebin():
    # Initalize a basic image
    img1 = ai.ReducedImage(arr1, uncertainty=sig1, properties={'binning':(2,2)})

    # Make a copy of that image
    img2 = img1.copy()

    # Set the binning along each axis
    dy,  dx  = 3, 3
    ny0, nx0 = img1.shape
    ny,  nx  = ny0//dy, nx0//dx

    # Perform the rebinning
    img1a = img1.rebin(nx, ny, total=True)
    img2a = img2.rebin(nx, ny, total=False)

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
    img1 = ai.ReducedImage(arr1, uncertainty=sig1, properties={'binning':(2,2)})

    # Make a copy of that image
    img2 = img1.copy()

    # Set the binning along each axis
    dy,  dx  = 0.5, 0.5
    ny0, nx0 = img1.shape
    ny,  nx  = np.int(ny0//dy), np.int(nx0//dx)

    # Perform the rebinning
    img1a = img1.rebin(nx, ny, total=True)
    img2a = img2.rebin(nx, ny, total=False)

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
    img1b = img1a.rebin(nx, ny, total=True)

    # Check if the output uncertainty has the expected value
    print(img1b.uncertainty)
    print(sig1)
    assert (np.abs((img1b.uncertainty/sig1) - 1.0) < 1e-4).all()
