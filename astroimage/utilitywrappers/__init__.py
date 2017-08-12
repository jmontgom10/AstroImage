"""
Provides several classes for extending the functionality of the image classes
in the AstroImage package.

Classes
-------
AdaptiveMesher        Performs a kind of adaptive mesh refinement (AMR) using
                      a user-defined statistic to determine which cells should
                      be further rebinned.

AstrometrySolver      Provides an interface with the Astrometry.net engine to
                      solve the astrometry of the image.
                      (Astrometry.net must must be separately installed by the
                      user, but see
                      https://sites.google.com/site/jmastronomy/Software/astrometry-net-setup
                      for some suggestions.)

ImageStack            Aligns a stack of images using their WCS or a
                      cross-correlation techique and combines the images using
                      a median-filtered-mean.

Inpainter             Indentifies any NaN or masked pixels and fills them using
                      a convolution based inpainted method.

PhotometryAnalyzer    Handles photometry of the sources in the image, including
                      median PSF estimation, curve-of-growth analysis, and
                      aperture photometry.

StokesParameters      Provides tools for computing Stokes parameter images from
                      four instrument independent polarimetric position angle
                      (IPPA) images and estimating polarization percentage (P)
                      and position angle (PA) from the Stokes parameters.
"""

# Make the primary class from each module accessible directly from the
# utilitywrappers subpackage.
from .adaptivemesher import AdaptiveMesher
from .astrometrysolver import AstrometrySolver
from .imagestack import ImageStack
from .inpainter import Inpainter
from .photometryanalyzer import PhotometryAnalyzer
from .stokesparameters import StokesParameters
