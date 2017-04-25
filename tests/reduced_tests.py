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

def test_add():
    # Build test images
    img1 = ai.reduced.ReducedImage(arr1, uncertainty=sig1, header=head1)
    img2 = ai.reduced.ReducedImage(arr2, uncertainty=sig2, header=head1)

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

    # Assert that the numpy and ai.reduced.ReducedImage yield the same result
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
    img1a = ai.reduced.ReducedImage(arr1, header=head1)
    img3d = img1a + img2
    img3e = img2  + img1a
    assert (img3d.data == res_a).all()
    assert (img3d.uncertainty == img2.uncertainty).all()
    assert (img3e.data == res_b).all()
    assert (img3e.uncertainty == img2.uncertainty).all()

def test_subtract():
    # Build test images
    img1 = ai.reduced.ReducedImage(arr1, uncertainty=sig1, header=head1)
    img2 = ai.reduced.ReducedImage(arr2, uncertainty=sig2, header=head1)

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

    # Assert that the numpy and ReducedImage yield the same result
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
    img1a = ai.reduced.ReducedImage(arr1, header=head1)
    img3d = img1a - img2
    img3e = img2  - img1a
    assert (img3d.data == res_a).all()
    assert (img3d.uncertainty == img2.uncertainty).all()
    assert (img3e.data == res_b).all()
    assert (img3e.uncertainty == img2.uncertainty).all()

def test_multiply():
    # Build test images
    img1 = ai.reduced.ReducedImage(arr1, uncertainty=sig1, header=head1)
    img2 = ai.reduced.ReducedImage(arr2, uncertainty=sig2, header=head1)

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

    # Assert that the numpy and ai.reduced.ReducedImage yield the same result
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
    img1a = ai.reduced.ReducedImage(arr1, header=head1)
    img3d = img1a * img2
    img3e = img2  * img1a
    assert (img3d.data == res_a).all()
    assert (img3d.uncertainty == img1a.data*img2.uncertainty).all()
    assert (img3e.data == res_b).all()
    assert (img3e.uncertainty == img1a.data*img2.uncertainty).all()

def test_divide():
    # Build test images
    img1 = ai.reduced.ReducedImage(arr1, uncertainty=sig1, header=head1)
    img2 = ai.reduced.ReducedImage(arr2, uncertainty=sig2, header=head1)

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

    # Assert that the numpy and ai.reduced.ReducedImage yield the same result
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
    img1a = ai.reduced.ReducedImage(arr1, header=head1)
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


#####
# SKIP THESE TESTS FOR NOW BECAUSE THEY JUST TAKE TOO LONG...
#####

# def test_get_sources():
#     # Build an ReducedScience instance from the initalized array
#     img1 = ai.reduced.ReducedScience(arr1)
#
#     # Use the get_sources method to retrieve the source positions
#     sx1, sy1 = img1.get_sources()
#
#     # Assert that ALL of the stars should be found!
#     assert (len(sx1) == len(sx0))
#     assert (len(sy1) == len(sy0))
#
#     # Set a plate scale of 0.5 arcsec/pixel
#     pl_sc = 0.5/3600.0 # deg/pixel
#
#     # Compute (RA, Dec) values for the input and retrieved star lists
#     ra0, dec0 = (sx0 - 0.5*nx)*pl_sc, (sy0 - 0.5*ny)*pl_sc
#     ra1, dec1 = (sx1 - 0.5*nx)*pl_sc, (sy1 - 0.5*ny)*pl_sc
#
#     # Convert these to Astropy SkyCoord objects
#     catalog0 = SkyCoord(ra0, dec0, unit='deg')
#     catalog1 = SkyCoord(ra1, dec1, unit='deg')
#
#     # Match the list of coordinates
#     idx, sep2d, _ = match_coordinates_sky(catalog1, catalog0)
#
#     # Assert that the best star matches should all be less than 0.5 arsec apart
#     assert (sep2d < 0.5*u.arcsec).all()
#
# def test_get_psf():
#     # Build an ReducedScience instance from the initalized array
#     img1 = ai.reduced.ReducedScience(arr1)
#
#     # Attempt to recover the PSF
#     PSFstamp, PSFparams = img1.get_psf()
#
#     # Assert that the recovered mean sx and sy should be within 1 percent
#     # of the actual value
#     std1 = np.sqrt(PSFparams['smajor']*PSFparams['sminor'])
#     assert (np.abs(std1/std0 - 1.0) < 0.01)

def test_shift():
    # Build a very simple, small, test array
    testArr = np.zeros((9,9))
    testArr[3:6,3:6] = 1.0

    # Build the test ReducedScience
    testImg1 = ai.reduced.ReducedScience(testArr)

    # Compute the total flux of the test image
    totalFlux = testImg1.data.sum()

    # Loop through a series of offsets
    for dx in np.arange(-2.0,2.5,0.5):
        for dy in np.arange(-2.0,2.5,0.5):
            testImg2 = testImg1.shift(dx, dy)

            # Assert that the flux was conserved
            assert_equal(totalFlux, testImg2.data.sum())

            # Assert that the image flux is distributed as expected
            # start by grabbing the corner and central values of the array
            bt = 3+np.int(np.floor(dy))
            tp = 3+np.int(np.ceil(dy))
            lf = 3+np.int(np.floor(dx))
            rt = 3+np.int(np.ceil(dx))
            cornerVal = testImg2.data[bt, lf]
            centerVal = testImg2.data[tp, rt]

            # If this would have been counted as an integer, then the corner
            # should be 1.0
            if np.round(dx, 12).is_integer():
                fracLf = 1.0
            else:
                # Compute the fractional contributions of each array
                dxRt = np.int(np.ceil(dx))
                dxLf = dxRt - 1
                fracRt = np.abs(dx - dxLf)
                fracLf = np.abs(dx - dxRt)

            if np.round(dy, 12).is_integer():
                fracBt = 1.0
            else:
                dyTp   = np.int(np.ceil(dy))
                dyBt   = dyTp - 1
                fracTp = np.abs(dy - dyBt)
                fracBt = np.abs(dy - dyTp)

            # Now make the assertion!
            assert_equal(fracLf*fracBt, cornerVal)
            assert_equal(centerVal, 1.0)

def test_gradient():
    # Build a very simple, small, test array
    testArr = np.zeros((9,9))
    testArr[3:6,3:6] = 1.0

    # Build the test ReducedScience
    testImg1 = ai.reduced.ReducedScience(testArr)

    # Compute the image gradient with the sobel operator
    Gx, Gy = testImg1.gradient(kernel='sobel')

    # Assert some expectations for these gradients
    assert_equal(Gx[2,:].min(), -1)
    assert_equal(Gx[2,:].max(), 1)
    assert_equal(Gx[3,:].min(), -3)
    assert_equal(Gx[3,:].max(), 3)
    assert_equal(Gx[4,:].min(), -4)
    assert_equal(Gx[4,:].max(), 4)
    assert_equal(Gx[:,4].sum(), 0)

    assert_equal(Gy[:,2].min(), -1)
    assert_equal(Gy[:,2].max(), 1)
    assert_equal(Gy[:,3].min(), -3)
    assert_equal(Gy[:,3].max(), 3)
    assert_equal(Gy[:,4].min(), -4)
    assert_equal(Gy[:,4].max(), 4)
    assert_equal(Gy[4,:].sum(), 0)

    # Compute the image gradient with the prewit operator
    Gx, Gy = testImg1.gradient(kernel='prewitt')

    # Assert some expectations for these gradients
    assert_equal(Gx[2,:].min(), -1)
    assert_equal(Gx[2,:].max(), 1)
    assert_equal(Gx[3,:].min(), -2)
    assert_equal(Gx[3,:].max(), 2)
    assert_equal(Gx[4,:].min(), -3)
    assert_equal(Gx[4,:].max(), 3)
    assert_equal(Gx[:,4].sum(), 0)

    assert_equal(Gy[:,2].min(), -1)
    assert_equal(Gy[:,2].max(), 1)
    assert_equal(Gy[:,3].min(), -2)
    assert_equal(Gy[:,3].max(), 2)
    assert_equal(Gy[:,4].min(), -3)
    assert_equal(Gy[:,4].max(), 3)
    assert_equal(Gy[4,:].sum(), 0)


def test_in_image():
    # Use the real image of M104 (contains WCS)
    img1 = realImg1.copy()

    # Generate a preselected list of RAs and Decs
    RAs  = [ 190.10869,  190.09656,  189.87761,  189.88588, # Out of image
             190.06298,  190.06031,  189.91760,  189.9157 ] # In image
    Decs = [-11.515961, -11.733118, -11.721072, -11.503332, # Out of image
             -11.55304, -11.680306, -11.686248, -11.561784] # In image

    # Convert these to SkyCoord objects
    coords = SkyCoord(RAs, Decs, unit='deg', frame='fk5')

    # Test if the coordinates are in the image
    testBools = img1.in_image(coords)

    # Assert that
    assert (testBools == np.array(4*[False] + 4*[True])).all()

def test_correct_airmass():
    # Build a test image
    img1 = ai.reduced.ReducedScience(arr1, uncertainty=np.sqrt(arr1), properties={'airmass': 1.5})

    # Apply an airmass correction
    kappa = 0.4
    img2 = img1.correct_airmass(atmExtCoeff=kappa)

    # Estimate correction factor
    correctionFactor = 10.0**(0.4*kappa*img1.airmass)

    # Estimate output and uncertainty
    res_a = img1.data*correctionFactor
    sig_a = img1.uncertainty*correctionFactor

    # Assert that the expected result is correct
    assert (np.abs(img2.data/res_a - 1.0) < 0.0025).all()
    assert (np.abs(img2.uncertainty/sig_a - 1.0) < 0.0025).all()

    # Assert that the airmass in the properties and header of the output are 0.0
    assert_equal(img2.airmass, 0)

    if 'AIRMASS' in ai.reduced.ReducedScience.headerKeywordDict:
        airmassKey = ai.reduced.ReducedScience.headerKeywordDict['AIRMASS']
        assert_equal(img2.header[airmassKey], 0)
