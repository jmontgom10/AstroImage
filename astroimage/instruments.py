"""
Provides the two Perkins telescope instruments commonly used by BU astronomers.
This can easily be generalized to include any other instruments from any other
telescope. Indeed, this is where the ability to read images from any telescope
is best handled. Once the class has been told how to read the header, the rest
of the work will be handled by the image classes themselves.
"""
#############
### PRISM ###
#############

PRISM_headerDict = {
    'AIRMASS': 'AIRMASS',
    'BINNING': ('ADELX_01', 'ADELY_01'),
    'INSTRUMENT': 'INSTRUME',
    'FILTER': 'FILTNME3',
    'PRESCANWIDTH': 'PRESCAN',
    'OVERSCANWIDTH': 'POSTSCAN',
    'RA': 'TELRA',
    'DEC': 'TELDEC',
    # TODO: Eventually find a way to include this?
    # 'FRAME': 'RADESYS',
    'EXPTIME': 'EXPTIME',
    'DATETIME': 'DATE-OBS',
    'OBSTYPE': 'OBSTYPE',
    'UNIT': 'BUNIT',
    'SCALEFACTOR': 'BSCALE',
    'GAIN': 'AGAIN_01'
}

Mimir_headerDict = {
    'AIRMASS': 'AIRMASS',
    'INSTRUMENT': 'INSTRUME',
    'FILTER': 'FILTNME2',
    'RA': 'TELRA',
    'DEC': 'TELDEC',
    # TODO: Eventually find a way to include this?
    # 'FRAME': 'RADESYS',
    'EXPTIME': 'EXPTIME',
    'DATETIME': 'DATE-OBS',
    'OBSTYPE': 'OBSTYPE',
    'UNIT': 'BUNIT',
    'SCALEFACTOR': 'BSCALE'
}
