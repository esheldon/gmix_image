"""
gmix_em
    Fit gaussian mixture models to images using Expectation Maximization

Classes
-------

GMixEM: 
    A class to fit a gaussian mixture model to an image using Expectation
    Maximization.  See docs for gmix_image.GMixEM for more details.

functions
---------
gmix2image_em:
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
import pprint
import numpy
from numpy import array, median, zeros, ogrid, exp, sqrt
import _gmix_em

import gmix_image
from .gmix import GMix
from .util import srandu

GMIXEM_ERROR_NEGATIVE_DET          = 0x1
GMIXEM_ERROR_MAXIT                 = 0x2
GMIXEM_ERROR_NEGATIVE_DET_COCENTER = 0x4
GMIXEM_ERROR_ADMOM_FAILED          = 0x8


class GMixEM(_gmix_em.GMixEM):
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
    gm = gmix_image.GMixEM(image, guess, sky=100, maxiter=2000, tol=1.e-6)

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
    gm = gmix_image.GMixEM(image, guess, psf=psf, sky=100)

    # run some unit tests
    gmix_image.test.test_em()
    gmix_image.test.test_em(add_noise=True)
    gmix_image.test.test_psf_em(add_noise=False)
    gmix_image.test.test_psf_colocate_em(add_noise=True)

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
            raise ValueError("send sky")
        if self._counts is None:
            self._counts = im.sum()

        verbosity  = 1 if self._verbose else 0
        do_cocenter = 1 if self._cocenter else 0
        do_coellip = 1 if self._coellip else 0

        super(GMixEM,self).__init__(self._image,
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
    pars=property(_gmix_em.GMixEM.get_pars)
    flags=property(_gmix_em.GMixEM.get_flags)
    numiter=property(_gmix_em.GMixEM.get_numiter)
    fdiff=property(_gmix_em.GMixEM.get_fdiff)

    def get_gmix(self):
        return self.get_pars()
    def get_model(self):
        pars=self.get_pars()
        return gmix2image_em(pars, self._image.shape,
                             psf=self._psf,
                             counts=self._counts)

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


class GMixEMPSF:
    """
    Process the image as if it is a PSF

    Send either cen= or ares= (adaptive moments output)

    Note if the initial adaptive moments run fails, then
    the result is simply {'flags':GMIXEM_ERROR_ADMOM_FAILED}
    """
    def __init__(self, image, ivar, ngauss, 
                 cen=None,
                 ares=None,
                 maxiter=5000, 
                 tol=1.e-6,
                 cocenter=False):
        self.image=image
        self.counts=image.sum()
        self.ivar=ivar
        self.cen_guess=cen
        self.ngauss=ngauss

        self.maxtry_admom=10
        self.maxtry_em=10

        self.maxiter=maxiter
        self.tol=tol
        self.cocenter=cocenter

        self.ares=ares

        if cen is None and ares is None:
            raise ValueError("send either cen= or ares=")
        self._run_em()

    def get_result(self):
        return self.result
    def get_gmix(self):
        return self.result['gmix']

    def _run_admom(self):
        import admom

        ntry=self.maxtry_admom
        for i in xrange(ntry):
            ares = admom.admom(self.image,
                               self.cen_guess[0],
                               self.cen_guess[1],
                               sigsky=sqrt(1/self.ivar),
                               guess=2.,
                               nsub=1)
            if ares['whyflag']==0:
                break

        if i==(ntry-1):
            self.ares=None
        else:
            self.ares=ares

    def _run_em(self):
        # only run admom if not sent
        if self.ares is None:
            self._run_admom()
        if self.ares is None:
            self.result={'flags':GMIXEM_ERROR_ADMOM_FAILED}
            return

        im,sky,guess0=self._do_prep()

        ntry=self.maxtry_em
        for i in xrange(ntry):
            guess = self._perturb_gmix(guess0)
            gm = GMixEM(im, guess, sky=sky, 
                        maxiter=self.maxiter, tol=self.tol,
                        cocenter=self.cocenter)
            flags = gm.get_flags()
            if flags==0:
                break
            """
            else:
                print 'em flags:'
                gmix_image.printflags('em',flags)
                pprint.pprint(gm.get_gmix())
            """

        if i==(ntry-1):
            print 'em flags:'
            gmix_image.printflags('em',gm.get_flags())
            #pprint.pprint(gm.get_gmix())



        self._fitter=gm
        gmix=GMix(gm.get_gmix())
        self.result={'gmix':gmix,
                     'flags':gm.get_flags(),
                     'numiter':gm.get_numiter(),
                     'fdiff':gm.get_fdiff(),
                     'ntry':i+1}


    def _perturb_gmix(self, gmix_in):
        gmix=copy.deepcopy(gmix_in) 
        for i in xrange(len(gmix)):
            # weirdness with references....
            g=gmix[i]
            g['p']   = g['p']*(1+0.01*srandu())
            g['row'] = g['row']*(1+0.01*srandu())
            g['col'] = g['col']*(1+0.01*srandu())
            g['irr'] = g['irr']*(1+0.05*srandu())
            g['irc'] = g['irc']*(1+0.05*srandu())
            g['icc'] = g['icc']*(1+0.05*srandu())

            gmix[i]=g

        return gmix



    def _do_prep(self):
        ares=self.ares
        row,col=ares['wrow'],ares['wcol']
        Tadmom=ares['Irr']+ares['Icc']
        irr_admom=ares['Irr']
        irc_admom=ares['Irc']
        icc_admom=ares['Icc']

        if self.ngauss==1:
            guess=[{'p':1.0,'row':row,'col':col,
                    'irr':ares['Irr'],
                    'irc':ares['Irc'],
                    'icc':ares['Icc']}]
        else:
            if self.ngauss==3:
                #Texamp=array([0.46,5.95,2.52])
                #pexamp=array([0.1,0.7,0.22])
                Texamp=array([0.3,1.0,.6])
                pexamp=array([0.1,0.7,0.22])
            elif self.ngauss==2:
                Texamp=array([12.6,3.8])
                pexamp=array([0.30, 0.70])
            else:
                raise ValueError("ngauss==3 or 2 for now")

            Tfrac=Texamp/Texamp.sum()
            pfrac=pexamp/pexamp.sum()

            guess=[]

            for i in xrange(self.ngauss):
                pi=pfrac[i]
                """
                Ti=Tadmom*Tfrac[i]
                g={'p':pi,'row':row,
                   'col':col,'irr':Ti/2,'irc':0.0,'icc':Ti/2}
                """
                g={'p':pi,'row':row, 'col':col,
                   'irr':Tfrac[i]*irr_admom,
                   'irc':Tfrac[i]*irc_admom,
                   'icc':Tfrac[i]*icc_admom}
                guess.append(g)

        im=self.image.copy()

        # need no zero pixels and sky value
        im_min = im.min()
        if im_min==0:
            sky=0.001
            im += sky
        elif im_min < 0:
            sky=0.001
            im += (sky-im_min)
        else:
            sky = im_min


        return im,sky,guess

    def get_model(self):
        """
        Model is normalized to equal the counts in the image
        """
        from .render import gmix2image

        if not hasattr(self,'_model'):
            gmix=self.result['gmix']
            model=gmix2image(gmix, self.image.shape)
            model *= self.counts/model.sum()
            self._model = model

        return self._model.copy()

    def get_loglike(self):
        """
        Get the log likelihood assuming counts of model==counts of image
        """

        model=self.get_model()
        diff=(model-self.image)*sqrt(self.ivar)
        loglike = -0.5*(diff**2).sum()
        return loglike

    def get_stats(self):
        """
        Get some stats on goodness of fit and bayesian information
        """
        import copy
        from math import log
        import scipy.stats

        if not hasattr(self, '_stats'):
            loglike=self.get_loglike()

            if self.cocenter:
                npars = self.ngauss*4+2
            else:
                npars=self.ngauss*6
            ndata=self.image.size

            chi2=loglike/(-0.5)
            dof=ndata-npars
            chi2per = chi2/dof

            prob = scipy.stats.chisqprob(chi2, dof)

            aic = -2*loglike + 2*npars
            bic = -2*loglike + npars*log(ndata)

            self._stats={'loglike':loglike,
                         'chi2per':chi2per,
                         'dof':dof,
                         'fit_prob':prob,
                         'aic':aic,
                         'bic':bic}

        return copy.deepcopy(self._stats)


    def compare_model(self):
        import images
        model=self.get_model()
        dof=self.image.size - self.ngauss*6
        images.compare_images(self.image, model,
                              label1='image',
                              label2='model',
                              skysig=sqrt(1/self.ivar),
                              dof=dof)

    def compare_normalized_model(self):
        """
        don't use this
        """
        import images
        im = self.get_normalized_model()
        images.compare_images(self.image, im)

    def get_normalized_gmix(self):
        """
        don't use this
        Get the gmix with the best normalization

        Note EM does not give this normalization, a separate
        fitter is used.
        """

        # only runs if norm not already found
        self.find_norm()
        return GMix(self._normalized_gmix.get_pars())

    def get_normalized_model(self):
        """
        don't use this
        Get a model with the best normalization

        Note EM does not give this normalization, a separate
        fitter is used.
        """
        from .render import gmix2image
        # only runs if norm not already found
        self.find_norm()
        return gmix2image(self._normalized_gmix, self.image.shape)

    def get_normalized_loglike(self):
        """
        don't use this
        Get the log likelihood

        Note EM does not give a normalization.  The norm is
        derived using a maximum likelihood fitter.
        """

        # only runs if norm not already found
        self.find_norm()

        ydiff=self._eval_normalized_ydiff([self._norm])
        return (ydiff**2).sum()


    def _eval_normalized_pars(self, counts):
        pars=self.result['gmix'].get_pars()
        psum=0
        ppos=0
        for i in xrange(self.ngauss):
            psum+=pars[ppos]
            ppos += 6

        ppos=0
        for i in xrange(self.ngauss):
            pars[ppos] *= counts/psum
        return pars

    def _eval_normalized_model(self, counts):
        from .render import gmix2image
        pars=self._eval_normalized_pars(counts)

        gmix=GMix(pars)
        mod=gmix2image(gmix, self.image.shape)
        return mod

    def _eval_normalized_ydiff(self, pars):
        counts=pars[0]
        mod=self._eval_normalized_model(counts)
        diff=(mod-self.image)*sqrt(self.ivar)
        return diff.ravel()

    def find_norm(self):
        """
        don't use this, just match the total counts
        Use least squares to find the best normalization

        only runs if norm not already found
        """
        from scipy.optimize import leastsq

        if hasattr(self, '_norm'):
            return

        guess=[self.counts]
        lmres = leastsq(self._eval_normalized_ydiff, guess, full_output=1)

        pars, pcov0, infodict, errmsg, ier = lmres

        if ier == 0:
            # wrong args, this is a bug
            raise ValueError(errmsg)

        if pcov0 is None:
            raise ValueError("bad cov: %s" % errmsg)
        if ier > 4:
            raise ValueError("error: %s" % errmsg)

        norm=pars[0]
        allpars=self._eval_normalized_pars(norm)

        self._normalized_gmix = GMix(allpars)
        self._norm = norm

        #print 'found norm:',norm,"  image sum is:",self.counts


def gmix2image_em(gauss_list, dims, 
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
        return gmix2image_psf_em(gauss_list, psf, dims, 
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


def gmix2image_psf_em(gauss_list, psf_list, dims, 
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


