"""
Provides the `raw` and `reduced` image modules for processing raw and reduced
astronomical images. Extra functionalitiy is provided via the 'utilitywrappers'
subpackage.

Modules
-------
instrument         Contains a simple class for storing instrument properties

raw                Contains all the classes used for handling raw telescope data

reduced            Contains all the classes used for handling reduced data

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

def set_instrument(instrument):
    """
    Provides a convenient way to set the instrument for the image classes.

    Parameters
    ----------
    instrument : dicts or str
        Defines the header keyword relationship to be used. Permissible string
        values for now are `PRISM` or `Mimir` (case insensitive).
    """
    presetInstruments = ['prism', 'mimir']

    if type(instrument) is str:
        if instrument.lower() == 'prism':
            from .instruments import PRISM_headerDict
            thisInstrument = PRISM_headerDict
        elif instrument.lower() == 'mimir':
            from .instruments import Mimir_headerDict
            thisInstrument = Mimir_headerDict
        elif instrument.lower() == '2mass':
            from .instruments import TMASS_headerDict
            thisInstrument = TMASS_headerDict
        elif instrument.lower() == 'none':
            thisInstrument = {}
        else:
            raise ValueError('`instrument` string of {} not recognized'.format(instrument))
    elif type(instrument) is dict:
        thisInstrument = instrument
    else:
        raise ValueError('`instrument` must be an dictionary or a string')

    # Now set the instrument for the BaseImage class (which will be inherited
    # but all other subclasses)
    from .baseimage import BaseImage
    BaseImage.set_headerKeywordDict(thisInstrument)
