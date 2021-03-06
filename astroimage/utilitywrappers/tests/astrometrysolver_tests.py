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


# Read in an ACTUAL image of M104
realImg1 = ai.reduced.ReducedScience.read('.\\tests\\M104_V_I.fits')

def test_solve_astrometry():
    # Use the real image of M104 so that astrometry can ACTUALLY be solved
    img1 = realImg1.copy()

    # Test if the exact same image is returned when an astrometric solution
    # already exists.
    astroSolver = ai.utilitywrappers.AstrometrySolver(img1)
    img2, success = astroSolver.run(clobber=False)

    assert (img2.wcs.wcs.crval == img1.wcs.wcs.crval).all()
    assert (img2.wcs.wcs.crpix == img1.wcs.wcs.crpix).all()
    assert (img2.wcs.wcs.cd    == img1.wcs.wcs.cd).all()
    assert img2.wcs.wcs.naxis == img1.wcs.wcs.naxis
    del img2

    # Next test the actual solution
    img1.clear_astrometry()
    astroSolver = ai.utilitywrappers.AstrometrySolver(img1)
    img2, success = astroSolver.run(clobber=True)

    # Test if this was successful
    assert success

    # TODO
    # Test if the output image has similar astrometry to the input image
    ny1, nx1  = img1.shape
    xpts1, ypts1 = np.array([0, nx1, nx1, 0]), np.array([0, 0, ny1, ny1])
    ra1, dec1 = realImg1.wcs.wcs_pix2world(xpts1, ypts1, 0, ra_dec_order=True)

    xpts2, ypts2 = img2.wcs.wcs_world2pix(ra1, dec1, 0)

    assert np.max(np.sqrt((xpts1 - xpts2)**2 + (ypts1 - ypts2)**2)) < 0.1
