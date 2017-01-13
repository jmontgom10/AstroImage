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

    # Make a copy of the first image in the imgList
    resImg1 = imgList[0].copy()

    # Manually compute the summed image from the remaining images
    for img in imgList[1:]:
        resImg1 += img

    # Compute the sum of the imgList using the utilities
    resImg2 = utils.combine_images(imgList, output='sum')

    # Assert that the numpy and AstroImage yield the same result
    assert (resImg1.arr == resImg2.arr).all()
    assert (resImg1.arr == 26).all()
    assert (resImg1.sigma == resImg2.sigma).all()

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

    # Make a copy of the first image in the imgList
    resImg1 = imgList[0].copy()

    # Manually compute the summed image from the remaining images
    for img in imgList[1:]:
        resImg1 += img

    # Convert the summed image to a mean by dividing by the number of
    # images in the imgList
    resImg1 /= len(imgList)

    # Compute the mean of the imgList using the utilities
    resImg2 = utils.combine_images(imgList, output='mean')

    # Assert that the numpy and AstroImage yield the same result
    assert (resImg1.arr == resImg2.arr).all()
    assert (resImg1.arr == 6.5).all()
    assert (resImg1.sigma == resImg2.sigma).all()

def test_combine_images_weighted_mean():
    # Set test image shape
    ny, nx = (100, 100)

    imgList = []
    arrList  = []
    sigList  = []
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

        # Build array list
        arr = tmpImg.arr.copy()
        arrList.append(arr)

        # Build uncertainty list
        sigma = tmpImg.sigma.copy()
        sigList.append(sigma)

    # Manually compute the summed image from the remaining images
    resArr = 0*imgList[0].arr
    resSig = 0*imgList[0].sigma
    for arr, sig in zip(arrList, sigList):
        resArr += (arr/(sig**2))

    for sig in sigList:
        resSig += 1.0/(sig**2)

    # Normalize the result by the sum of the weights
    resArr /= resSig

    # Compute the uncertainty in the weighted mean
    sig_resArr = 1.0/np.sqrt(resSig)

    # Compute the mean of the imgList using the utilities
    resImg2 = utils.combine_images(imgList, output='weighted_mean')

    # Assert that the numpy and AstroImage yield the same result
    assert (resArr == resImg2.arr).all()
    assert (np.abs(resImg2.arr - 5.768124626) < 1e-5).all()
    assert (sig_resArr == resImg2.sigma).all()
    assert (np.abs(resImg2.sigma - 1.468666139) < 1e-5).all()
