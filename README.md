# AstroImage (for python 3)

This file contains the base class (and subclasses)  for use with the
PRISM_pyBDP, PRISM_pyPol, and Mimir_pyPol reduction scripts. The class is
essentially a container for a FITS image, but it also includes a decent set of
methods for common operations on images (e.g. computing gradients, croping,
shifting, etc...).

The curent version of the class includes several methods for operating on
"stacks" of images. These methods operate not on individual AstroImage instances
but on images but on LISTS of AstroImage instances. All of the stack operations
include the word "stack" or "stacked" in the method name. For example,
AstroImage.stacked_average(imgList) takes a list of AstroImage objects (imgList)
of identical shape and computes the average value for each pixel in that set of
images. Future versions may include a "ImageStack" class for handling these
obperations.

There are also "magic methods" defined for the AstroImage class. These methods
allow basic arithmetic operations between images. For example if img1 and img2
are AstroImage instances, then the code

```
img3 = img1 * img2
```

computes the product of the image arrays from img1 and img2 and stores that
array in a new AstroImage object assigned to the variable name "img3".

The name of the base class is "AstroImage", and the subclasses are "Bias",
"Dark", and "Flat". The base class can be imported using

```
from AstroImage import AstroImage
```

and FITS images can be read in with the following

```
img1 = AstroImage('/path/to/FITS/file/img1.fits')
```

# Dependencies and Setup

The AstroImage class depends on a number of other python packages. I recommend
using the Anaconda environment, as that comes with numpy, scipy, astropy, and matplotlib
preinstalled. If you elect not to use Anaconda, then make sure to get those
packages properly installed before proceeding to install the AstroImage
dependencies.

## wcsaxes

This package allows the WCS class from astropy to be used in the plotting
routines such that the axes of the image properly reflect the world coordinate
system contained in the fits image header. Future versions of astropy may
include wcsaxes, so check to see if you really need to install this package
separately.

### Installation

Using pip, you can install this package via

```
pip install wcsaxes
```

If you have Anaconda installed, then the process is a bit different. Use the
following code

```
conda install --channel https://conda.anaconda.org/astropy wcsaxes
```

## photutils

 This package allows the user to perform some basic photometric functions,
 namely locating stars in the image although future work may include a
 "do_photometry" method to streamline those kinds of functions.

### Installation

Using pip, you can install this package via

```
pip install --no-deps photutils
```

If you have Anaconda installed, then the process is a bit different. Use the
following code

```
conda install --channel https://conda.anaconda.org/astropy photutils
```
