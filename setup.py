try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'AstroImage is a class (with some utilities) for convenient acces of astronomical image data.',
    'author': 'Jordan Montgomery',
    'url': 'https://github.com/jmontgom10/AstroImage',
    'download_url': 'https://github.com/jmontgom10/AstroImage/zipball/master',
    'author_email': 'jmontgom.10@gmail.com',
    'version': '0.1',
    'install_requires': [
        'nose',
        'astropy',
        'astroquery',
        'numpy',
        'scipy',
        'WCSAxes'
    ],
    'packages': ['astroimage'],
    'scripts': [],
    'name': 'astroimage'
}

setup(**config)
