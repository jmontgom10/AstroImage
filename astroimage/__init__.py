"""
This package provides a set of image classes for reading and manipulating
astronomy FITS images. The RawImage subclasses RawBias, RawDark, and RawFlat are
useful for reducing raw calibration data from a detector. The reduced
calibration data will be stored in a the ReducedImage subclasses MasterBias,
MasterDark, and MasterFlat. These can be used to finally reduce RawScience
images to produce AstroImage instances. The AstroImage class has the ability to
hook into the users system processes and run the Astrometry.net engine to solve
the astrometry of each image.

The image combination can be performed using the ImageStack class, which
provides methods for aligning and combining a set of images cleanly.
"""

# Import each class from each file
from .baseimage import BaseImage
from .rawimages import *
from .reducedimages import *
from .reducedscience import ReducedScience
from .imagestack import ImageStack
from .astrometrysolver import AstrometrySolver
