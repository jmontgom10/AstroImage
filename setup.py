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
    'version': '0.8',
    'install_requires': [
        # 'python >= 3.5',
        'nose >= 1.3',
        'astropy >= 1.3',
        'wcsaxes >= 0.6',
        'astroquery >= 0.3',
        'numpy >= 1.11',
        'scipy >= 0.15',
        'matplotlib >= 1.1',
        'psutil >= 3.0',
        'photutils >= 0.3',
        'scikit-image >= 0.11',
        'scikit-learn >=0.18'
    ],
    'packages': ['astroimage'],
    'scripts': [],
    'name': 'astroimage'
}

setup(**config)
