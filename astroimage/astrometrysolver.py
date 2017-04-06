# Core imports
import os
import sys
import warnings
import subprocess

# Astropy imports
from astropy.nddata import NDDataArray, StdDevUncertainty
from astropy.io import fits
from astropy.wcs import WCS

# AstroImage imports
from .astroimage import AstroImage

# Define which functions, classes, objects, etc... will be imported via the command
# >>> from imagestack import *
__all__ = ['AstrometrySolver']

class AstrometrySolver(object):
    """
    A class to interface with the Astrometry.net solve-field executable

    Properties
    ----------
    image     The AstroImage instance for which to solve astrometry

    Methods
    -------
    run       Executes the Astrometry.net solver on the `image` instance
    """

    def __init__(self, image):
        """
        Constructs an AstrometrySolver instance using the supplied AstroImage
        image instance
        """
        # Check if an AstroImage was provided
        if type(image) is not AstroImage:
            raise TypeError('`image` must be an AstroImage instance')

        # Otherwise simply store the image in the image attribute
        self.image = image

    ###
    # Helper methods for solve_astrometry
    ###
    def _test_for_astrometry_dot_net(self):
        """Tests whether or not astrometry.net in installed"""
        # # TODO FIGURE OUT SOME PLATFORM INDEPENDENT WAY TO TEST IF
        # # ASTROMETRY.NET IS INSTALLED
        # #
        # # First test if the astrometry binary is installed
        # proc = subprocess.Popen(['where', 'solve-field'],
        #                         stdout=subprocess.PIPE,
        #                         universal_newlines=True)
        # astrometryEXE = ((proc.communicate())[0]).rstrip()
        # proc.terminate()
        #
        return True

    def _parse_on_disk_filename(self):
        """
        Gets the name of the on-disk file to pass to astrometry.net

        Returns
        -------
        fileToSolve: str
            The path to the file on which astrometry.net will operate

        temporaryFile: bool
            If True, then `fileToSolve` is a temporary file and should be
            deleted when finished
        """
        # Test if the file is already written to disk
        if os.path.isfile(str(self.image.filename)):
            fileToSolve   = self.image.filename
            temporaryFile = False
            return fileToSolve, temporaryFile

        # Assign a temporary file name and write it to disk
        fileToSolve   = os.path.join(os.getcwd(), 'tmp.fits')
        temporaryFile = True
        self.image.write(fileToSolve)

        return fileToSolve, temporaryFile

    @staticmethod
    def _get_executable_parameters(fileToSolve):
        """
        Returns a list of parameters used to build  `solve-field` command

        Parameters
        ----------
        fileToSolve : str
            The path to the file on which the astrometry.net `solve-field`
            executable will be run.

        Returns
        -------
        prefix : str
            The content to go *before* the `solve-field` command

        modifiedFilePath: str
            An updated path to the file, compatible with the system running the
            `solve-field` executable. In the case of Windows, this is either
            `Cygwin` or `BashOnWindows`. In the case of a POSIX machine, this
            will just be the regular path to the file.

        suffix: str
            The contet to ge *after the `solve-field` + modifidFilename command

        shellCommand: bool
            The `subprocess` option to run this as a shell command. This will be
            True for Windows machines and can be False for POSIX machines.
        """
        # Test what kind of system is running
        if 'win' in sys.platform:
            # If running in Windows, then...
            try:
                ###
                # Cygwin (on Windows)
                ###
                # Try to parse using cygpath executable...
                # then define the "bash --login -c (cd ...)" command
                # using Cygwin's "cygpath" to convert to POSIX format
                # Try to get the proper path using the cygpath executable
                proc = subprocess.Popen(['cygpath', os.getcwd()],
                                        stdout=subprocess.PIPE,
                                        universal_newlines=True)
                currentDirectory = ((proc.communicate())[0]).rstrip()
                proc.terminate()

                # Convert filename to Cygwin compatible format
                proc = subprocess.Popen(['cygpath', fileToSolve],
                                        stdout=subprocess.PIPE,
                                        universal_newlines=True)
                modifiedFilePath = ((proc.communicate())[0]).rstrip()
                proc.terminate()

                # Use this syntax to setup commands for the Cygwin bash.exe
                prefix = 'bash --login -c ("cd ' + currentDirectory + '; '
                suffix = '")'
                shellCommand = True
            except:
                ###
                # BashOnWindows
                ###
                # Attempt to convert the current directory into the proper format
                currentDirectory = os.getcwd()
                driveLetter      = currentDirectory[0]
                currentDirectory = currentDirectory.replace(
                    driveLetter + ':\\', '/mnt/' + driveLetter.lower() + '/'
                    )
                currentDirectory = currentDirectory.replace('\\','/')

                # Apply the same transformation to the input file
                driveLetter = fileToSolve[0]
                modifiedFilePath = fileToSolve.replace(
                    driveLetter + ':\\', '/mnt/' + driveLetter.lower() + '/')
                modifiedFilePath = modifiedFilePath.replace('\\','/')

                # Use this syntax to setup commands for BashOnWindows
                prefix = 'bash --login -c "cd ' + currentDirectory + '; '
                suffix = '"'
                shellCommand = True
        else:
            # If running a *nix system,
            # then define null prefix/suffix strings
            prefix = ''
            suffix = ''
            shellCommand = False

        return prefix, modifiedFilePath, suffix, shellCommand

    def _build_executable_string(self, prefix, modifiedFilePath, suffix):
        """
        Constructs the string to run the astrometry.net executable

        Parameters
        ----------
        prefix : str
            The content to be placed before the `solve-field` command

        modifiedFilePath : str
            The path to the file on which to operate

        suffix : str
            The content to be placed after the `solve-field` command

        Returns
        -------
        command : str
            The full solve-field command to be run as a subprocess
        """
        # TODO:
        # play with these parameters and get them to be the most up-to-date

        # Setup the basic input/output command options
        outputCmd    = ' --out tmp'
        noPlotsCmd   = ' --no-plots'
        overwriteCmd = ' --overwrite'
        dirCmd       = ''

        # Provide a guess at the plate scale
        scaleLowCmd  = ' --scale-low 0.10'
        scaleHighCmd = ' --scale-high 1.8'
        scaleUnitCmd = ' --scale-units arcsecperpix'

        # Provide some information about the approximate location
        if self.image.ra is not None:
            raCmd = ' --ra ' + str(self.image.ra.value)
        else: raCmd = ''

        if self.image.dec is not None:
            decCmd = ' --dec ' + str(self.image.dec.value)
        else: decCmd = ''

        radiusCmd    = ' --radius 0.3'

        # This is reduced data, so we won't need to clean up the image
        # imageOptions = '--no-fits2fits --no-background-subtraction'
        imageOptions = ''

        # Prevent writing any except the "tmp.wcs" file.
        # In the future it may be useful to set '--index-xyls'
        # to save star coordinates for photometry.

        noOutFiles = ' --new-fits none' + \
                     ' --index-xyls none' + \
                     ' --solved none' + \
                     ' --match none' + \
                     ' --rdls none' + \
                     ' --corr none'

        # Build the astrometry.net command
        solve_fieldCmd = 'solve-field' + \
                          outputCmd + \
                          noPlotsCmd + \
                          overwriteCmd + \
                          dirCmd + \
                          scaleLowCmd + \
                          scaleHighCmd + \
                          scaleUnitCmd + \
                          raCmd + \
                          decCmd + \
                          radiusCmd + \
                          imageOptions + \
                          noOutFiles + \
                          ' ' + modifiedFilePath

        # Build the full command
        command = prefix + solve_fieldCmd + suffix

        return command

    @staticmethod
    def _run_executable_solver(command, shellCommand):
        """Initiates a subprocess to run the `solve-field` executable"""
        try:
            # Attempt to run the solver
            astroProc = subprocess.Popen(command, shell=shellCommand)
            astroProc.wait()
            astroProc.terminate()
        except:
            raise OSError('Could not run the `solve-field` executable')

    # @staticmethod
    def _read_astrometric_solution_from_disk(self):
        """
        Reads the astrometry solution written WCS file and returns it

        Returns
        -------
        wcs : astropy.wcs.wcs.WCS
            A World Coordinate System (WCS) object containing the astrometric
            solution

        success : bool
            If True, then WCS was found and astrometry was successful. If False,
            then WCS was not found and astrometric solution failed.

        temporaryFilePaths : tuple
            A tuple of file paths to the temporary files created by the
            Astrometry.net engine.
        """
        # Construct the path to the newly created WCS file. The name of these
        # files is hardcoded into this class, so they are not variables.
        destinationDirectory = os.path.dirname(self.image.filename)
        wcsPath              = os.path.join(destinationDirectory, 'tmp.wcs')
        axyPath              = os.path.join(destinationDirectory, 'tmp.axy')

        # Construct a tuple of temporary file paths to delete
        temporaryFilePaths = (wcsPath, axyPath)

        # Read in the tmp.wcs file and create a WCS object
        if os.path.isfile(wcsPath):
            # Read in the WCS solution
            HDUlist = fits.open(wcsPath)

            # Correct the number of axes in the WCS header
            HDUlist[0].header['NAXIS'] = self.image.header['NAXIS']

            # Copy the WCS from the solved file
            wcs = WCS(HDUlist[0].header)
            HDUlist.close()

            # Return the WCS and a True success value
            success = True

            return wcs, success, temporaryFilePaths

        else:
            # Return empty WCS with a False success value
            wcs     = None
            success = False

            return wcs, success, temporaryFilePaths


    @staticmethod
    def _cleanup_temporary_files(temporaryFilePaths, fileToSolve, temporaryFile):
        """Deletes any residual temporary files from the disk"""
        # Delete all the Astrometry.net temporary files
        for filePath in temporaryFilePaths:
            if os.path.isfile(filePath): os.remove(filePath)

        # If the .fits file was marked as temporary, then delete it
        if temporaryFile:
            if os.path.isfile(filePath): os.remove(fileToSolve)

    def run(self, clobber=False):
        """
        Invokes the astrometry.net engine and solves the image astrometry.

        Parameters
        ----------
        clobber : bool, optional, default: False
            If true, then whatever WCS may be stored in the header will be
            deleted and overwritten with a new solution.

        Returns
        -------
        outImg : AstroImage
            A copy of the original image with the best fitting astrometric
            solution stored in the `header` and `wcs` properties. If the
            astrometric solution was not successful, then this will simply be
            the original image.

        success : bool
            A flag to indicate whether the astrometric solution was successful.
            A True value indicates success and a False value indicates failure.
        """
        # Check if a solution exists and we're not supposed to overwrite it.
        if self.image.has_wcs and not clobber:
            warnings.warn('Astrometry for file {0:s} already solved... skipping.'.format(
                os.path.basename(str(self.image.filename))
            ))
            # Return the original image and declare success
            outImg  = self.image
            success = True

            return outImg, success

        # Check if astrometry.net is installed on this system
        if not self._test_for_astrometry_dot_net():
            # Otherwise report an error!
            raise OSError('Astrometry.net is not installed on this system.')

        # Determine the name of the file on which to perform operations
        fileToSolve, temporaryFile = self._parse_on_disk_filename()

        # Convert that filename to a astrometry.net friendly name and get other
        # parameters for running the solver on this machine.
        tmp = self._get_executable_parameters(fileToSolve)
        prefix, astrometryCompatibleFilename, suffix, shellCommand = tmp

        # Combine the parameters into a singe command to be run in a shell
        command = self._build_executable_string(
            prefix, astrometryCompatibleFilename, suffix
        )

        # Use the subprocess module to run the `solve-field` executable
        self._run_executable_solver(command, shellCommand)

        # Check if the expected output files are there and extract astrometry
        wcs, success, tmpFilePaths = self._read_astrometric_solution_from_disk()

        if success:
            # If there was a WCS to be read, then...
            # Make a copy of the image to be returned
            outImg = self.image.copy()

            # Clear out the old astrometry and insert new astrometry
            outImg.astrometry_to_header(wcs)

            # Place the same WCS in the output image __fullData attribute
            outImg._BaseImage__fullData = NDDataArray(
                outImg.data,
                uncertainty=StdDevUncertainty(outImg.uncertainty),
                unit=outImg.unit,
                wcs=wcs
            )

        else:
            # If there was no WCS, then return original image and  a False
            # success value
            outImg = self.image

        # Delete the temporary files now that the WCS has been extracted
        self._cleanup_temporary_files(tmpFilePaths, fileToSolve, temporaryFile)

        # Return the results to the user
        return outImg, success
