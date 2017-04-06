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

# Define a header handler
def header_handler(header):
    # Move the binning information to the 'ADELX_01' and 'ADELY_01' keys
    header['ADELX_01'] = header['CRDELT1']
    header['ADELY_01'] = header['CRDELT2']

    return header

# Read in an ACTUAL image of M104
realImg1 = ai.AstroImage.read('.\\tests\\M104_V_I.fits', properties={'gain': 3.3})

# Set the header handler
ai.BaseImage.set_header_handler(header_handler)

# Read in an ACTUAL image of M104
realImg2 = ai.AstroImage.read('.\\tests\\M104_V_I.fits', properties={'gain': 3.3})

import pdb; pdb.set_trace()

# Use the real image of M104 so that astrometry can ACTUALLY be solved
img1 = realImg1.copy()

# Test if the exact same image is returned when an astrometric solution
# already exists.
astroSolver = ai.AstrometrySolver(img1)
img2, success = astroSolver.run(clobber=False)

assert (img2.wcs.wcs.crval == img1.wcs.wcs.crval).all()
assert (img2.wcs.wcs.crpix == img1.wcs.wcs.crpix).all()
assert (img2.wcs.wcs.cd    == img1.wcs.wcs.cd).all()
assert img2.wcs.wcs.naxis == img1.wcs.wcs.naxis
del img2

# Next test the actual solution
img1.clear_astrometry()
astroSolver = ai.AstrometrySolver(img1)
img2, success = astroSolver.run()

# Test if this was successful
assert success

# TODO
# Test if the output image has similar astrometry to the input image
ny1, nx1  = img1.shape
xpts1, ypts1 = np.array([0, nx1, nx1, 0]), np.array([0, 0, ny1, ny1])
ra1, dec1 = realImg1.wcs.wcs_pix2world(xpts1, ypts1, 0)

xpts2, ypts2 = img2.wcs.wcs_world2pix(ra1, dec1, 0)

assert np.max(np.sqrt((xpts1 - xpts2)**2 + (ypts1 - ypts2)**2)) < 0.1

import pdb; pdb.set_trace()
