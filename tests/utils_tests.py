# Import numpy for making test number stuff
import numpy as np
from astropy.io import fits

# Import testing utitities
from nose.tools import *
from nose import with_setup

# Import the base class
from astroimage.astroimage import AstroImage
from astroimage import utils

# Start by testing the most algebraically intensive function...
def test_combine_images_sum():
    # Set test image shape
    ny, nx = (100, 100)

    imgList = []
    for i in range(4):
        tmpImg = AstroImage()
        tmpImg.arr     = np.zeros((ny, nx)) + 5 + i
        tmpImg.sigma   = np.zeros((ny, nx)) + 2 + i
        tmpImg.dtype   = tmpImg.arr.dtype
        tmpImg.binning = (1, 1)
        tmpImg.header  = fits.Header(cards={
            'NAXIS1':nx,
            'NAXIS2':ny,
            'AIRMASS':1 + 0.1*i})
        imgList.append(tmpImg)

    # Generate a null image
    nullImg = imgList[0].copy()
    nullImg.arr = 0*imgList[0].arr

    # Make a copy of the null image to add up the image list
    sumImg1 = nullImg.copy()

    # Manually compute the summed image
    for img in imgList:
        sumImg1 += img

    # Compute the sum of the imgList using the utilities
    sumImg2 = utils.combine_images(imgList, output='sum')

    # Assert that the numpy and AstroImage yield the same result
    assert (sumImg1.arr == sumImg2.arr).all()

# Start by testing the most algebraically intensive function...
def test_combine_images_mean():
    # Set test image shape
    ny, nx = (100, 100)

    imgList = []
    for i in range(4):
        tmpImg = AstroImage()
        tmpImg.arr     = np.zeros((ny, nx)) + 5 + i
        tmpImg.sigma   = np.zeros((ny, nx)) + 2 + i
        tmpImg.dtype   = tmpImg.arr.dtype
        tmpImg.binning = (1, 1)
        tmpImg.header  = fits.Header(cards={
            'NAXIS1':nx,
            'NAXIS2':ny,
            'AIRMASS':1 + 0.1*i})
        imgList.append(tmpImg)

    # Generate a null image
    nullImg = imgList[0].copy()
    nullImg.arr = 0*imgList[0].arr

    # Make a copy of the null image to add up the image list
    sumImg1 = nullImg.copy()

    # Manually compute mean
    for img in imgList:
        sumImg1 += img

    sumImg1 /= len(imgList)

    # Compute the mean of the imgList using the utilities
    sumImg2 = utils.combine_images(imgList, output='mean')

    # Assert that the numpy and AstroImage yield the same result
    assert (sumImg1.arr == sumImg2.arr).all()
    # assert_equal(sumImg)
