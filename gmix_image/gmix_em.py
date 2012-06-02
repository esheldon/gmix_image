"""
gmix_image
    Fit gaussian mixture models to images.

Classes
-------

GMix: 
    A class to fit a gaussian mixture model to an image using Expectation
    Maximization.  See docs for gmix_image.GMix for more details.

functions
---------
gmix2image:
    Create an image from the gaussian input mixture model.  Optionally
    send a PSF, which results in a call to gmix2image_psf.

gmix2image_psf:
    Create an image from the input gaussian mixture model and psf mixture
    model.

ogrid_image:
    Create an image using the ogrid function from numpy

total_moms:
    Get the total moments for the mixture model; only easily
    interpreted when the centers coincide.  Optionally send a
    psf, which results in a call to total_moms_psf
total_moms_psf:
    Get the total moments for the mixture model and psf; only easily
    interpreted when the centers coincide.
"""

import copy
import numpy
from numpy import array, median, zeros, ogrid, exp, sqrt
import _gmix_image

GMIXEM_ERROR_NEGATIVE_DET         = 0x1
GMIXEM_ERROR_MAXIT                = 0x2
GMIXEM_ERROR_NEGATIVE_DET_COCENTER = 0x4


class GMixEM(_gmix_image.GMix):
    """
    Fit a gaussian mixture model to an image using Expectation Maximization.

    A gaussian mixture model is represented by a list of dictionaries with the
    gaussian parameters.

    construction parameters
    -----------------------
    image: numpy array
        A two dimensional, 64-bit float numpy array.  If the image
        is a different type it will be converted.
    guess: list of dictionaries
        A gaussian mixture model represented as a list of dictionaries.  Each
        dict defines the parameters of a gaussian.  The length of this list
        determines how many gaussians will be fit and the initial start
        parameters.  Each dict must have the following entries

            p: A normalization
            row: center of the gaussian in the first dimension
            col: center of the gaussian in the second dimension
            irr: Covariance matrix element for row*row
            irc: Covariance matrix element for row*col
            icc: Covariance matrix element for col*col

        Note the normalizations are only meaningful in a relative sense.

    sky: number, optional
        The sky level in the image. Must be non-zero for the EM algorithm to
        converge, since it is essentially treated like an infinitely wide
        gaussian.  If not sent, the median of the image is used, so it is
        recommeded to send your best estimate.

    counts: number, optional
        The total counts in the image.  If not sent it is calculated
        from the image, which is fine.

    psf: list of dictionaries, optional
        A gaussian mixture model representing the PSF.  The best fit gaussian
        mixture will thus be the pre-psf values.  Centers are not necessary for
        the psf mixture model.
    bound: dict
        A boundary region over which to work.  The dictionary should have the
        following integer entries
            rowmin
            rowmax
            colmin
            colmax
        The values will be clipped to within the actual image boundaries, and
        such that *max >= *min

    maxiter: number, optional
        The maximum number of iterations.
    tol: number, optional
        The tolerance to determine convergence.  This is the fractional
        difference in the weighted summed moments between iterations.
    cocenter: bool
        If True, force the centers of all gaussians to agree.
    coellip: bool
        If True, force the centers of all gaussians to agree and
        the covariance matrices to be proportional.
    verbose: bool, optional
        Print out some information for each iteration.

    properties
    ----------
    pars: list of dictionaries
        The fitted gaussian mixture model at the end of the last iteration.
        These have the same entries as the guess parameter with the addition of
        "det", which has the determinant of the covariance matrix.  There is a
        corresponding method get_pars() if you prefer.

    flags: number
        A bitmask holding flags for the processing.  Should be zero for
        success.  These flags are defined as attributes to the gmix_image
        module.

            GMIXEM_ERROR_NEGATIVE_DET         1 # determinant <= 0 in covar
            GMIXEM_ERROR_MAXIT                2 # max iteration reached
            GMIXEM_ERROR_NEGATIVE_DET_COCENTER 4 # not used currently

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

    # initial guesses as a gaussian mixture model
    guess = [{'p':0.4,'row':10,'col':10,'irr':2.5,'irc':0.1,'icc':3.1},
             {'p':0.6,'row':15,'col':17,'irr':1.7,'irc':0.3,'icc':1.5}]

    # create the gaussian mixture
    gm = gmix_image.GMix(image, guess, sky=100, maxiter=2000, tol=1.e-6)

    # Work with the results
    if gm.flags != 0:
        print 'failed with flags:',gm.flags

    print 'number of iterations:',gm.numiter
    print 'fractional diff on last iteration:',gm.fdiff

    pars = gm.pars
    print 'center for first guassian:',pars[0]['row'],pars[0]['col']

    # Find the gaussian mixture accounting for a point spread function.  The
    # psf is just another gaussian mixture model.  The fit gaussian mixture
    # will thus be "pre-psf". Centers are not necessary for the psf.

    psf = [{'p':0.8,'irr':1.2,'irc':0.2,'icc':1.0},
           {'p':0.2,'irr':2.0,'irc':0.1,'icc':1.5}]
    gm = gmix_image.GMix(image, guess, psf=psf, sky=100)

    # run some unit tests
    gmix_image.test()
    gmix_image.test(add_noise=True)
    gmix_image.test_psf(add_noise=False)
    gmix_image.test_psf_colocate(add_noise=True)

    """
    def __init__(self, im, guess, 
                 sky=None,
                 counts=None,
                 maxiter=1000,
                 tol=1.e-6,
                 psf=None,
                 bound=None,
                 cocenter=False,
                 coellip=False,
                 verbose=False):

        self._image = array(im, ndmin=2, dtype='f8', copy=False)
        self._guess = copy.deepcopy(guess)
        self._sky=sky
        self._counts=counts
        self._maxiter=maxiter
        self._tol=tol
        self._psf=self._fixup_psf(copy.deepcopy(psf))
        self._bound=copy.deepcopy(bound)
        self._cocenter = cocenter
        self._coellip = coellip
        self._verbose=verbose

        if self._sky is None:
            self._sky = median(im)
        if self._counts is None:
            self._counts = im.sum()

        verbosity  = 1 if self._verbose else 0
        do_cocenter = 1 if self._cocenter else 0
        do_coellip = 1 if self._coellip else 0

        super(GMix,self).__init__(self._image,
                                  self._sky,
                                  self._counts,
                                  self._guess,
                                  self._maxiter,
                                  self._tol,
                                  psf=self._psf,
                                  bound=self._bound,
                                  cocenter=do_cocenter,
                                  coellip=do_coellip,
                                  verbose=verbosity)

    # just to make access nicer.
    pars=property(_gmix_image.GMix.get_pars)
    flags=property(_gmix_image.GMix.get_flags)
    numiter=property(_gmix_image.GMix.get_numiter)
    fdiff=property(_gmix_image.GMix.get_fdiff)

    def _fixup_psf(self, psf):
        """
        Add center info if not there, just to make it a full gvec definition
        """
        if psf is None:
            return None

        if not isinstance(psf,list):
            raise ValueError("expected psf to be a list of dicts")

        for p in psf:
            if not isinstance(p,dict):
                raise ValueError("expected psf to be a list of dicts")

            if 'row' not in p:
                p['row'] = -1
            if 'col' not in p:
                p['col'] = -1
        return psf

def flagname(flag):
    if flag == GMIXEM_ERROR_NEGATIVE_DET:
        return 'GMIXEM_ERROR_NEGATIVE_DET'
    elif flag == GMIXEM_ERROR_MAXIT:
        return 'GMIXEM_ERROR_MAXIT'
    elif flag == GMIXEM_ERROR_NEGATIVE_DET_COCENTER:
        return 'GMIXEM_ERROR_NEGATIVE_DET_cocenter'
    else:
        raise ValueError("unknown flag value: %s" % flag)
def flagval(flag):
    if flag == 'GMIXEM_ERROR_NEGATIVE_DET':
        return GMIXEM_ERROR_NEGATIVE_DET
    elif flag == 'GMIXEM_ERROR_MAXIT':
        return GMIXEM_ERROR_MAXIT
    elif flag == 'GMIXEM_ERROR_NEGATIVE_DET_COCENTER':
        return GMIXEM_ERROR_NEGATIVE_DET_COCENTER
    else:
        raise ValueError("unknown flag name: '%s'" % flag)

def gmix_print(gmix):
    hfmt = ['%10s']*6
    hfmt = ' '.join(hfmt)
    h = hfmt % ('p','row','col','irr','irc','icc')
    print h

    fmt = ['%10.6g']*6
    fmt = ' '.join(fmt)
    for g in gmix:
        print fmt % tuple([g[k] for k in ['p','row','col','irr','irc','icc']])


def gmix2image(gauss_list, dims, 
               psf=None, aslist=False, renorm=True, 
               order='c',
               nsub=1,
               counts=1.0):
    """
    Create an image from the gaussian input mixture model.

    By default

        im = sum( pi*counts*imi )/sum(pi)

    where imi is normalized to one.  Send renorm=False to not divide
    by sum(pi)

    parameters
    ----------
    gauss_list:
        The gaussian mixture model as a list of dictionaries.
    dims:
        The dimensions of the result.  This matters since
        the gaussian centers are in this coordinate syste.
    psf: optional
        An optional gaussian mixture PSf model.  The models will be convolved
        with this PSF.
    aslist:
        Get a list of images.  If a psf is sent, you get a list of lists.

    renorm:
        Make images as sum( pi*counts*imi ), not dividing by sum(pi).  For
        gmix we want to divide by sum(pi) but not in the optimizer for 
        example.
    order:
        'c' for C order, 'f' for fortran order.  The images are made in
        fortran, so using 'f' would be faster but take care.
    counts: optional
        The total counts in the image.  Default 1. If  renorm is False, 
        the image sum will be sum(pi)*counts.
    """
    try:
        import fimage
        model_image = fimage.model_image
    except:
        #print 'using ogrid_image'
        model_image = ogrid_image

    if psf is not None:
        return gmix2image_psf(gauss_list, psf, dims, 
                              aslist=aslist, renorm=renorm,
                              order=order, nsub=nsub,
                              counts=counts)
    if aslist:
        modlist=[]
    else:
        im = zeros(dims)

    psum = sum([g['p'] for g in gauss_list])
    for g in gauss_list:
        gp = g['p']
        if renorm:
            gp /= psum

        tmp =  model_image('gauss',
                           dims,
                           [g['row'],g['col']],
                           [g['irr'],g['irc'],g['icc']],
                           counts=gp*counts,
                           nsub=nsub,
                           order=order)
        if aslist:
            modlist.append(tmp)
        else:
            im += tmp

    if aslist:
        return modlist
    else:
        return im


def gmix2image_psf(gauss_list, psf_list, dims, 
                   aslist=False, renorm=True, 
                   order='c',
                   nsub=1,
                   counts=1.0):
    """
    Create an image from the input gaussian mixture model and psf mixture
    model.
    """
    try:
        import fimage
        model_image = fimage.model_image
    except:
        #print 'using ogrid_image'
        model_image = ogrid_image

    if aslist:
        modlist=[]
    else:
        im = zeros(dims)
        tmp_im = zeros(dims)

    g_psum = sum([g['p'] for g in gauss_list])

    for g in gauss_list:
        gp = g['p']
        if renorm:
            gp /= g_psum

        row=g['row']
        col=g['col']

        # we always normalize the psf
        p_psum = sum([psf['p'] for psf in psf_list])

        if aslist:
            modlist.append([])
        else:
            tmp_im[:,:] = 0.0

        for psf in psf_list:
            pp = psf['p']/p_psum
            irr = g['irr'] + psf['irr']
            irc = g['irc'] + psf['irc']
            icc = g['icc'] + psf['icc']

            cnt = counts*gp*pp
            pim = model_image('gauss',
                              dims,
                              [row,col],
                              [irr,irc,icc],
                              counts=cnt,
                              nsub=nsub,
                              order=order)
            if aslist:
                modlist[-1].append(pim)
            else:
                tmp_im[:,:] += pim
        if not aslist:
            im += tmp_im

    if aslist:
        return modlist
    else:
        return im

def ogrid_image(model, dims, cen, cov, counts=1.0, **keys):
    """
    Create an image using the ogrid function from numpy

    parameters
    ----------
    model: string
        gauss,exp,dev
    dims: list
        two element list giving the dimensions
    cen: list
        two element list giving the center position
    cov: list
        Three element list giving the covariance
        elements Irr,Irc,Icc
    counts: number, optional
        The total counts.
    """
    Irr,Irc,Icc = cov
    det = Irr*Icc - Irc**2
    if det == 0.0:
        raise RuntimeError("Determinant is zero")

    Wrr = Irr/det
    Wrc = Irc/det
    Wcc = Icc/det

    # ogrid is so useful
    row,col=ogrid[0:dims[0], 0:dims[1]]

    rm = array(row - cen[0], dtype='f8')
    cm = array(col - cen[1], dtype='f8')

    rr = rm**2*Wcc -2*rm*cm*Wrc + cm**2*Wrr

    model = model.lower()
    if model == 'gauss':
        rr = 0.5*rr
    elif model == 'exp':
        rr = sqrt(rr*3.)
    elif model == 'dev':
        rr = 7.67*( (rr)**(.125) -1 )
    else: 
        raise ValueError("model must be one of gauss, exp, or dev")

    image = exp(-rr)

    image *= counts/image.sum()

    return image



def total_moms(gauss_list, psf=None):
    """
    Only makes sense if the centers are the same

    parameters
    ----------
    gauss_list: 
        A gaussian mixture model as a list of dicts.
    psf: optional
        A PSF as a gaussian mixture model.  The result
        will be convolved with the PSF.
    """
    if psf is not None:
        return total_moms_psf(gauss_list, psf)

    d={'irr':0.0, 'irc':0.0, 'icc':0.0}
    psum=0.0
    for g in gauss_list:
        p=g['p']
        psum += p
        d['irr'] += p*g['irr']
        d['irc'] += p*g['irc']
        d['icc'] += p*g['icc']

    d['irr'] /= psum
    d['irc'] /= psum
    d['icc'] /= psum
    return d

def total_moms_psf(gauss_list, psf_list):
    """
    Only makes sense if the centers are the same
    """
    d={'irr':0.0, 'irc':0.0, 'icc':0.0}
    psf_totmom = total_moms(psf_list)

    psum=0.0
    for g in gauss_list:
        p=g['p']
        psum += p
        d['irr'] += p*(g['irr'] + psf_totmom['irr'])
        d['irc'] += p*(g['irc'] + psf_totmom['irc'])
        d['icc'] += p*(g['icc'] + psf_totmom['icc'])

    d['irr'] /= psum
    d['irc'] /= psum
    d['icc'] /= psum
    return d

