"""
Provides the `raw` and `reduced` image modules for processing raw and reduced
astronomical images. Extra functionalitiy is provided via the 'utilitywrappers'
subpackage.

Modules
-------
raw

reduced

Subpackages
--------
mixins             Contains several mixin classes for easily extending the
                   functionality of som of the image classes.

utilitywrappers    Contains several wrapper classes which take image objects as
                   as initializing argument and produce objects with extra
                   functionality. See the utilitywrappers documentation for
                   more information.
"""
# # Provide public access to each subpackage and module.
from . import raw
from . import reduced
from . import utilitywrappers
