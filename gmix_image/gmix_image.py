"""
gmix_image

Defines a class to fit a gaussian mixture model to an image using Expectation
Maximization.  

See docs for gmix_image.GMix for more details.

The code is primarily in a C library. The GMix object is a convenience wrapper
for that code.
"""
import copy
import numpy
from numpy import median
import _gmix_image
from _gmix_image import GVec

GMIX_ERROR_NEGATIVE_DET         = 0x1
GMIX_ERROR_MAXIT                = 0x2
GMIX_ERROR_NEGATIVE_DET_SAMECEN = 0x4


class GMix(_gmix_image.GMix):
    """
    Fit a gaussian mixture model to an image using Expectation Maximization.

    construction parameters
    -----------------------
    image: numpy array
        A two dimensional, 64-bit float numpy array.  If the image
        is a different type it will be converted.
    gauss_guess: list of dictionaries
        A list of dictionaries, with each dict defining the starting
        parameters for a gaussian.  The length of this list determines
        how many gaussians are fit and the initial start parameters.
        Each dict must have the following entries

            p: A normalization
            row: center of the gaussian in the first dimension
            col: center of the gaussian in the second dimension
            irr: Covariance matrix element for row*row
            irc: Covariance matrix element for row*col
            icc: Covariance matrix element for col*col
    sky: number
        The sky level in the image. Must be non-zero for the EM algorithm to
        converge, since it is essentially treated like an infinitely wide
        gaussian.  If not sent, the median of the image is used, so it is
        recommeded to send your best estimate.

    counts: number
        The total counts in the image.  If not sent it is calculated
        from the image, which is fine.
        
    maxiter: number
        The maximum number of iterations.
    tol:
        The tolerance to determine convergence.  This is the fractional
        difference in the weighted summed moments between iterations.
    verbose:
        Print out some information for each iteration.

    properties
    ----------
    pars: list of dictionaries
        The gaussian parameters at the end of the last iteration.  These have
        the same entries as the gauss_guess parameter with the addition of
        "det", which has the determinant of the covariance matrix.  There
        is a corresponding method get_pars() if you prefer.

    flags: number

        A bitmask holding flags for the processing.  Should be zero for
        success.  These flags are defined as attributes to the gmix_image
        module.

            GMIX_ERROR_NEGATIVE_DET         1 # negative determinant in covar
            GMIX_ERROR_MAXIT                2 # max iteration reached
            GMIX_ERROR_NEGATIVE_DET_SAMECEN 4 # not used currently

        There is a corresponding method get_flags() if you prefer.

    numiter: number
        The number of iterations used during processing.  Will equal maxiter if
        the maximum iterations was reached.  There is a corresponding method
        get_numiter() if you prefer.

    fdiff: number
        The fractional difference between the weighted moments for the last two
        iterations.  For convergence this will be less than the input
        tolerance.  There is a corresponding method get_fdiff() if you prefer.

    examples
    --------
    import gmix_image

    # initial guesses
    guess = [{'p':0.4,'row':10,'col':10,'irr':2.5,'irc':0.1,'icc':3.1},
             {'p':0.6,'row':15,'col':17,'irr':1.7,'irc':0.3,'icc':1.5}]

    # create the gaussian mixture
    gm = gmix_image.GMix(image, guess, sky=100, maxiter=2000, tol=1.e-6)

    # Work with the results
    if gm.flags != 0:
        print 'failed with flags:',gm.flags

    print 'number of iterations:',gm.fdiff
    print 'fractional diff on last iteration:',gm.numiter

    pars = gm.pars
    print 'best fit center for first guass:',pars[0]['row'],pars[0]['col']
    """
    def __init__(self, im, gauss_guess, 
                 sky=None,
                 counts=None,
                 maxiter=1000,
                 tol=1.e-6,
                 verbose=False):

        self._image = numpy.array(im, ndmin=2, dtype='f8', copy=False)
        self._gauss_guess = copy.deepcopy(gauss_guess)
        self._sky=sky
        self._counts=counts
        self._maxiter=maxiter
        self._tol=tol
        self._verbose=verbose

        if self._sky is None:
            self._sky = median(im)
        if self._counts is None:
            self._counts = im.sum()

        if self._verbose:
            verbosity=1
        else:
            verbosity=0

        # this will get modified internally
        self._gvec = _gmix_image.GVec(self._gauss_guess)
        super(GMix,self).__init__(self._image,
                                  self._sky,
                                  self._counts,
                                  self._gvec,
                                  self._maxiter,
                                  self._tol,
                                  verbosity)

    def get_pars(self):
        """
        Get the gaussian parameters for the last iteration
        as a list of dictionaries.
        """
        return self._gvec.asdicts()

    pars=property(get_pars)
    flags=property(_gmix_image.GMix.get_flags)
    numiter=property(_gmix_image.GMix.get_numiter)
    fdiff=property(_gmix_image.GMix.get_fdiff)
