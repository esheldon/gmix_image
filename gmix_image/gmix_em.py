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
    Create an image from the gaussian input mixture model.

ogrid_image:
    Create an image using the ogrid function from numpy
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
    verbose: bool, optional
        Print out some information for each iteration.

    getters
    ----------
    get_gmix(): get a GMix object
        get a GMix object representing the mixture
    get_dlist(): list of dictionaries
        The fitted gaussian mixture model at the end of the last iteration.
        These have the same entries as the guess parameter with the addition of
        "det", which has the determinant of the covariance matrix.
    get_model(): get a model image
        The image is not normalized to the counts in the original

    get_flags(): number
        A bitmask holding flags for the processing.  Should be zero for
        success.  These flags are defined as attributes to the gmix_image
        module.

            GMIXEM_ERROR_NEGATIVE_DET         1 # determinant <= 0 in covar
            GMIXEM_ERROR_MAXIT                2 # max iteration reached
            GMIXEM_ERROR_NEGATIVE_DET_COCENTER 4 # not used currently

    get_numiter(): number
        The number of iterations used during processing.  Will equal maxiter if
        the maximum iterations was reached.

    get_fdiff(): number
        The fractional difference between the weighted moments for the last two
        iterations.  For convergence this will be less than the input
        tolerance.

    examples
    --------
    import gmix_image

    # initial guesses as a gaussian mixture model
    guess = [{'p':0.4,'row':10,'col':10,'irr':2.5,'irc':0.1,'icc':3.1},
             {'p':0.6,'row':15,'col':17,'irr':1.7,'irc':0.3,'icc':1.5}]

    # create the gaussian mixture
    gm = gmix_image.GMixEM(image, guess, sky=100, maxiter=2000, tol=1.e-6)

    # Work with the results
    flags=gm.get_flags()
    if flags != 0:
        print 'failed with flags:',flags

    print 'number of iterations:',gm.get_numiter()
    print 'fractional diff on last iteration:',gm.get_fdiff()

    dlist = gm.get_dlist()
    print 'center for first guassian:',dlist[0]['row'],dlist[0]['col']

    # run some unit tests
    gmix_image.test.test_em()
    """
    def __init__(self, im, guess, 
                 sky=None,
                 counts=None,
                 maxiter=5000,
                 tol=1.e-6,
                 bound=None,
                 cocenter=False,
                 verbose=False):

        self._image = array(im, ndmin=2, dtype='f8', copy=False)
        self._guess = copy.deepcopy(guess)
        self._sky=sky
        self._counts=counts
        self._maxiter=maxiter
        self._tol=tol
        self._bound=copy.deepcopy(bound)
        self._cocenter = cocenter
        self._verbose=verbose

        if self._sky is None:
            raise ValueError("send sky")
        if self._counts is None:
            self._counts = im.sum()

        verbosity  = 1 if self._verbose else 0
        do_cocenter = 1 if self._cocenter else 0

        super(GMixEM,self).__init__(self._image,
                                    self._sky,
                                    self._counts,
                                    self._guess,
                                    self._maxiter,
                                    self._tol,
                                    bound=self._bound,
                                    cocenter=do_cocenter,
                                    verbose=verbosity)

    def get_gmix(self):
        """
        Get a GMix object representing the result
        """
        return GMix(self.get_dlist())

    def get_model(self):
        """
        Model is not normalized
        """
        from .render import gmix2image

        gmix=self.get_gmix()
        model=gmix2image(gmix, self.image.shape)
        return model


class GMixEMBoot:
    """

    This version can bootstrap because it can generate guesses based on an
    intial guess of the center and size only.  Also it will retry up to a
    specified number of guesses until convergence is found.

    parameters
    ----------
    image: ndarray
        The image to fit. The can be sky subtracted; a constant will be
        added for the EM to ensure all values are positive.
    ngauss: int
        number of gaussians
    cen_guess: [row,col]
        First guess at the center
    sigma_guess: number, optional
        An optional starting guess for the "sigma" of a single
        gausian.  Default sqrt(2) which is fwhm=0.9'' in DES
    ivar: number, optional
        To get a meaningful chi^2 in the returned statistics,
        send a correct ivar.  Default 1
    maxiter: number, optional
        maximum number of iterations, default 5000.  For ngauss > 2
        you may well need that many
    tol: number, optional
        Tolerance in the moments for convergence, default 1.e-6
    cocenter: bool, optional
        If True, force the gaussians to be cocentric.  Default False
    maxtry: number, optional
        Maximum number of retries.  Default 10.

    methods
    -------
    get_result():
        Return a dict with some results and stats
    get_gmix():
        Get a GMix representation of the result
    get_model():
        Get an image of the result, not normalized.
    """
    def __init__(self, image, ngauss, cen_guess,
                 sigma_guess=1.414, # fwhm=0.9'' in des
                 ivar=1.0,
                 maxiter=5000, 
                 tol=1.e-6,
                 cocenter=False,
                 maxtry=10):

        self.image0=image
        self._prep_image() # this sets self.image
        self.counts=self.image.sum()

        self.ngauss=ngauss
        self.cen_guess=cen_guess

        self.sigma_guess=sigma_guess
        self.ivar=ivar

        self.maxiter=maxiter
        self.tol=tol
        self.cocenter=cocenter
        self.maxtry=maxtry

        self._run_em()

    def get_result(self):
        """
        The result, holding a dict list of pars, a GMix representation, and
        some stats about the fit.
        """
        return self.result

    def get_gmix(self):
        """
        Get the GMix representation
        """
        return self.result['gmix']

    def _run_em(self):
        guess0=self._get_guess()

        ntry=self.maxtry
        for i in xrange(ntry):
            guess = self._perturb_dlist(guess0)
            gm = GMixEM(self.image,
                        guess,
                        sky=self.sky, 
                        maxiter=self.maxiter,
                        tol=self.tol,
                        cocenter=self.cocenter)

            flags = gm.get_flags()
            if flags==0:
                break

        if flags != 0:
            print 'em flags:'
            gmix_image.printflags('em',flags)

        self._fitter=gm
        gmix=gm.get_gmix()
        self.result={'gmix':gmix,
                     'flags':flags,
                     'numiter':gm.get_numiter(),
                     'fdiff':gm.get_fdiff(),
                     'ntry':i+1}
        if flags == 0:
            stats=self.get_stats()
            self.result.update(stats)

    def _perturb_dlist(self, dlist_in):
        dlist=copy.deepcopy(dlist_in) 
        for i in xrange(len(dlist)):
            # weirdness with references....
            g=dlist[i]
            g['p']   = g['p']*(1+0.01*srandu())
            g['row'] = g['row'] + 0.5*srandu()
            g['col'] = g['col'] + 0.5*srandu()
            g['irr'] = g['irr']*(1+0.05*srandu())
            g['irc'] = g['irc']*(1+0.05*srandu())
            g['icc'] = g['icc']*(1+0.05*srandu())

            dlist[i]=g

        return dlist

    def _get_guess(self):
        """
        Get a guess as a list of dictionaries.  Guesses
        for higher ngauss are ~turbulence like
        """
        row=self.cen_guess[0]
        col=self.cen_guess[1]
        irr_guess=self.sigma_guess**2

        if self.ngauss==1:
            guess=[{'p':1.0,
                    'row':row,
                    'col':col,
                    'irr':irr_guess,
                    'irc':0.0,
                    'icc':irr_guess}]
        else:
            if self.ngauss==3:
                Texamp=array([0.3,1.0,.6])
                pexamp=array([0.1,0.7,0.22])
            elif self.ngauss==2:
                Texamp=array([12.6,3.8])
                pexamp=array([0.30, 0.70])
            else:
                raise ValueError("ngauss==1,2,3 for now")

            Tfrac=Texamp/Texamp.sum()
            pfrac=pexamp/pexamp.sum()

            guess=[]

            for i in xrange(self.ngauss):
                pi=pfrac[i]
                g={'p':pi,'row':row, 'col':col,
                   'irr':Tfrac[i]*irr_guess,
                   'irc':0.0,
                   'icc':Tfrac[i]*irr_guess}
                guess.append(g)

        return guess

    def _prep_image(self):

        im=self.image0.copy()

        # need no zero pixels and sky value
        im_min = im.min()
        if im_min==0:
            sky=0.001
            im += sky
        elif im_min < 0:
            sky=0.001
            im += (sky-im_min)
        else:
            #sky = im_min
            sky=numpy.median(im)

        self.image=im
        self.sky=sky

    def get_model(self):
        """
        Model is not normalized
        """
        from .render import gmix2image

        gmix=self.result['gmix']
        model=gmix2image(gmix, self.image.shape)
        return model

    def get_loglike(self):
        """
        Get the log likelihood.  The model is forced to have
        the same counts as the image and the articial "sky" is
        added
        """

        model=self.get_model()
        model += self.sky

        model *= self.counts/model.sum()
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

        stats={'loglike':loglike,
               'chi2per':chi2per,
               'dof':dof,
               'fit_prob':prob,
               'aic':aic,
               'bic':bic}

        return stats

class GMixEMPSF:
    """
    deprecated

    Send either cen= or ares= (adaptive moments output)

    This version can "stand alone" because it does an
    initial fit for a single gaussian to bootstrap.  You
    do need to give it guess at the center however, 
    and a size guess will be helpful.
    """
    def __init__(self, image, ivar, ngauss, 
                 cen=None,
                 ares=None,
                 maxiter=5000, 
                 tol=1.e-6,
                 cocenter=False,
                 maxtry_admom=10,
                 maxtry_em=10):
        self.image=image
        self.counts=image.sum()
        self.ivar=ivar
        self.cen_guess=cen
        self.ngauss=ngauss

        self.maxtry_admom=maxtry_admom
        self.maxtry_em=maxtry_em

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

        if flags != 0:
            print 'em flags:'
            gmix_image.printflags('em',flags)



        self._fitter=gm
        gmix=gm.get_gmix()
        self.result={'gmix':gmix,
                     'flags':gm.get_flags(),
                     'numiter':gm.get_numiter(),
                     'fdiff':gm.get_fdiff(),
                     'ntry':i+1}
        if flags == 0:
            stats=self.get_stats()
            self.result.update(stats)

    def _perturb_gmix(self, gmix_in):
        gmix=copy.deepcopy(gmix_in) 
        for i in xrange(len(gmix)):
            # weirdness with references....
            g=gmix[i]
            g['p']   = g['p']*(1+0.01*srandu())
            g['row'] = g['row'] + 0.5*srandu()
            g['col'] = g['col'] + 0.5*srandu()
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
            #sky = im_min
            sky=numpy.median(im)


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

    '''
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
        return GMix(self._normalized_gmix.get_dlist())

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
        '''

def gmix2image_em(gauss_list, dims, 
                  aslist=False, renorm=True, 
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
    aslist:
        Get a list of images.

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


