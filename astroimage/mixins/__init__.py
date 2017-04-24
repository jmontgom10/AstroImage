"""
Contains several mixin classes to provided additional functionality for the
images in the `raw` and `reduced` modules.


Classes
-------
NumericsMixin    Provides the numerical methods (e.g., sin, cos, log, log10)
                 used by the ReducedScience image class.

ResizingMixin    Provides the cropping, slicing, and rebinning functionality
                 used by the ReducedScience image class.
"""

# Import each mixin class to make them accessible directly from the mixins
# subpackage.
from .numericsmixin import NumericsMixin
from .resizingmixin import ResizingMixin
