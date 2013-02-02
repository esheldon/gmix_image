from sys import stderr
import pprint

import numpy
from numpy import zeros, array, where, ogrid, diag, sqrt, isfinite, \
        tanh, arctanh, cos, sin, exp, log
from numpy.linalg import eig

from .util import srandu

from .gmix_em import gmix2image_em
from .render import gmix2image
from .util import total_moms, gmix2pars, print_pars, get_estyle_pars, \
        randomize_e1e2, calculate_some_stats

from .gmix_mcmc import CenPrior

from . import _render

from . import gmix
from .gmix import GMix

GMIXFIT_MAXITER         = 2**0
GMIXFIT_SINGULAR_MATRIX = 2**4
GMIXFIT_NEG_COV_EIG     = 2**5
GMIXFIT_NEG_COV_DIAG    = 2**6
GMIXFIT_NEG_MCOV_DIAG   = 2**7 # the M sub-cov matrix not positive definite
GMIXFIT_MCOV_NOTPOSDEF  = 2**8 # more strict checks on cholesky decomposition
GMIXFIT_CALLS_NOT_CHANGING   = 2**9 # see fmin_cg
GMIXFIT_LOW_S2N = 2**8 # very low S/N for ixx+iyy

class GMixFitSimple:
    """
    This works in g1,g2 space
    """
    def __init__(self, image, ivar, psf, model, ares, **keys):
        # cen1,cen2,e1,e2,T,p
        self.npars=6

        self.image=image
        self.ivar=float(ivar)
        self.psf_gmix=psf
        self.model=model
        self.ares=ares
        self.cen_width=keys.get('cen_width',1.0)

        if self.ares['whyflag']!=0:
            raise ValueError("ares must have whyflag==0")


        self.counts=self.image.sum()
        self.lm_max_try=10

        self.highval=9.999e9

        self.gprior=keys.get('gprior',None)
        self.gprior_like=keys.get('gprior_like',False)

        self._go()

    def get_result(self):
        return self._result

    def _go(self):
        """
        Do a levenberg-marquardt
        """
        from scipy.optimize import leastsq

        ntry=self.lm_max_try
        for i in xrange(1,ntry+1):

            guess=self._get_guess()

            if self.gprior is None or not self.gprior_like:
                lmres = leastsq(self._get_lm_ydiff, guess, full_output=1)
            else:
                lmres = leastsq(self._get_lm_ydiff_gprior, guess, full_output=1)

            res=self._calc_lm_results(lmres)

            if res['flags']==0:
                break

        if i == ntry:
            print "could not find maxlike after",ntry,"tries"

        self._result=res

    def _get_guess(self):
        guess=zeros(self.npars)
        
        Tadmom=self.ares['Irr'] + self.ares['Icc']

        guess[0]=self.ares['wrow'] + 0.01*srandu()
        guess[1]=self.ares['wcol'] + 0.01*srandu()

        if self.gprior is not None:
            g1rand,g2rand=self.gprior.sample2d(1)
        else:
            g1rand=0.1*srandu()
            g2rand=0.1*srandu()
        guess[2]=g1rand
        guess[3]=g2rand

        guess[4] = Tadmom*(1 + 0.1*srandu())
        guess[5] = self.counts*(1 + 0.1*srandu())

        return guess


    def _get_lm_ydiff(self, pars):
        """
        pars are [cen1,cen2,g1,g2,T,counts]
        """

        ydiff = zeros(self.image.size, dtype='f8')

        epars=get_estyle_pars(pars)
        if epars is None:
            ydiff[:] = self.highval
            return ydiff

        gmix=self._get_convolved_gmix(epars)
 
        _render.fill_ydiff(self.image, self.ivar, gmix, ydiff)

        return ydiff

    def _get_lm_ydiff_gprior(self, pars):
        """
        pars are [cen1,cen2,g1,g2,T,counts]
        """

        ydiff = zeros(self.image.size+1, dtype='f8')

        g=sqrt(pars[2]**2 + pars[3]**2)
        gp = self.gprior.prior2d_gabs_scalar(g)

        if gp > 0:
            gp = log(gp)
        else:
            ydiff[:] = self.highval
            return ydiff

        epars=get_estyle_pars(pars)
        if epars is None:
            ydiff[:] = self.highval
            return ydiff

        gmix=self._get_convolved_gmix(epars)
 
        _render.fill_ydiff(self.image, self.ivar, gmix, ydiff)

        ydiff[-1] = gp

        return ydiff



    def _get_convolved_gmix(self, epars):
        """
        epars must be in e1,e2 space
        """
        gmix0=GMix(epars, type=self.model)
        gmix=gmix0.convolve(self.psf_gmix)
        return gmix

    def get_model(self):
        epars=get_estyle_pars(self._result['pars'])
        gmix=self._get_convolved_gmix(epars)
        return gmix2image(gmix, self.image.shape)

    def _compare_model(self,gmix):
        import images
        im=gmix2image(gmix, self.image.shape)
        images.compare_images(self.image, im)

    def _calc_lm_results(self, lmres):

        res={'model':self.model,
             'restype': 'lm'}

        pars, pcov0, infodict, errmsg, ier = lmres

        if ier == 0:
            # wrong args, this is a bug
            raise ValueError(errmsg)

        res['pars'] = pars
        res['pcov0'] = pcov0
        res['numiter'] = infodict['nfev']
        res['errmsg'] = errmsg
        res['ier'] = ier

        res['g']=pars[2:2+2].copy()

        epars=get_estyle_pars(pars)
        gmix=self._get_convolved_gmix(epars)
        stats=calculate_some_stats(self.image, 
                                   self.ivar, 
                                   gmix,
                                   self.npars)
        for k in stats:
            res[k] = stats[k]

        pcov=None
        perr=None
        if pcov0 is not None:
            pcov = self._scale_leastsq_cov(pars, pcov0)

            d=diag(pcov)
            w,=where(d < 0)

            if w.size == 0:
                # only do if non negative
                perr = sqrt(d)

        flags = 0
        if ier > 4:
            flags = 2**(ier-5)

        if pcov is None:
            flags += GMIXFIT_SINGULAR_MATRIX 
        else:
            e,v = eig(pcov)
            weig,=where(e < 0)
            if weig.size > 0:
                flags += GMIXFIT_NEG_COV_EIG 

            wneg,=where(diag(pcov) < 0)
            if wneg.size > 0:
                flags += GMIXFIT_NEG_COV_DIAG 


        res['pcov']=pcov
        res['perr']=perr
        res['flags']=flags

        res['gciv']=None
        res['gsens']=1.0
        res['Tmean']=pars[4]
        res['Terr']=9999.
        res['Ts2n']=0.0
        res['arate']=1.0
        if res['flags']==0:
            res['gcov'] = pcov[2:2+2, 2:2+2]
            # is this right?
            res['Terr'] =perr[4]
            res['Ts2n']=res['Tmean']/res['Terr']
            res['arate']=1.

        return res

    def _scale_leastsq_cov(self, pars, pcov):
        """
        Scale the covariance matrix returned from leastsq; this will
        recover the covariance of the parameters in the right units.
        """
        imsize=self.image.size
        ydiff=self._get_lm_ydiff(pars)
        dof = (self.image.size-len(pars))

        # 0:imsize to remove priors
        s_sq = (ydiff[0:imsize]**2).sum()/dof
        return pcov * s_sq 


class GMixFitCoellip:
    """
    Fit a guassian mixture model to an image.

    The gaussians are co-elliptical.  This parametrization is in terms of a
    Tmax and Tfrac_i.  This parametrization makes it easier to let e1,e2,Tmax
    be free but put priors on the Tfrac values, for example.
    
    Image is assumed to be background-subtracted

    This version does not have an analytical jacobian defined.

    parameters
    ----------
    image: numpy array, dim=2
        A background-subtracted image.
    pixerr: scalar or image
        The error per pixel or an image with the error
        for each pixel.
    prior:
        The guess and the middle of the prior for each parameter.

        for coellip
            [cen1,cen2,e1,e2,T1,T2,..,p1,p2..]

        There are ngauss-1 fi values
    width:
        Width of the prior on each of the above.

    psf: list of dictionaries
        A gaussian mixture model specified as a list of
        dictionaries.  Each dict has these entries
            p: A normalization
            row: center of the gaussian in the first dimension
            col: center of the gaussian in the second dimension
            irr: Covariance matrix element for row*row
            irc: Covariance matrix element for row*col
            icc: Covariance matrix element for col*col

    verbose: bool
    """
    def __init__(self, image, pixerr, prior, width,
                 psf=None, 
                 Tpositive=True,
                 model='coellip',
                 nsub=1,
                 verbose=False):
        self.image=image
        self.pixerr=pixerr
        self.ierr = 1./pixerr
        self.ivar=self.ierr**2
        self.prior=prior
        self.width=width

        self.verbose=verbose

        self.check_prior(prior, width)
        self.set_psf(psf)

        self.ngauss=(len(prior)-4)/2
        self.nsub=nsub

        self.Tpositive=Tpositive

        self.model=model

        self.dofit()

    def check_prior(self, prior, width):
        if len(prior) != len(width):
            raise ValueError("prior and width must be same len")

    def set_psf(self, psf):

        if psf is not None:
            self.psf_gmix = GMix(psf)
            self.psf_pars = gmix2pars(self.psf_gmix)
        else:
            self.psf_gmix = None
            self.psf_pars = None

    def dofit(self):
        """
        Run the fit using LM
        """
        from scipy.optimize import leastsq

        res = leastsq(self.get_ydiff,
                      self.prior,
                      full_output=1)

        self.pars, self.pcov0, self.infodict, self.errmsg, self.ier = res
        if self.ier == 0:
            # wrong args, this is a bug
            raise ValueError(self.errmsg)

        self.numiter = self.infodict['nfev']

        self.pcov=None
        self.perr=None

        if self.pcov0 is not None:
            self.pcov = self.scale_leastsq_cov(self.pars, self.pcov0)

            d=diag(self.pcov)
            w,=where(d < 0)

            if w.size == 0:
                # only do if non negative
                self.perr = sqrt(d)

        self.set_flags()
       
    def set_flags(self):
        flags = 0
        if self.ier > 4:
            flags = 2**(self.ier-5)
            if self.verbose:
                print >>stderr,self.errmsg 

        if self.pcov is None:
            if self.verbose:
                print >>stderr,'singular matrix'
            flags += GMIXFIT_SINGULAR_MATRIX 
        else:
            e,v = eig(self.pcov)
            weig,=where(e < 0)
            if weig.size > 0:
                if self.verbose:
                    import images
                    print >>stderr,'negative covariance eigenvalues'
                    print_pars(self.pars, front='pars: ')
                    print_pars(e,         front='eig:  ')
                    images.imprint(self.pcov,stream=stderr)
                flags += GMIXFIT_NEG_COV_EIG 

            wneg,=where(diag(self.pcov) < 0)
            if wneg.size > 0:
                if self.verbose:
                    import images
                    # only print if we didn't see negative eigenvalue
                    print >>stderr,'negative covariance diagonals'
                    #images.imprint(self.pcov,stream=stderr)
                flags += GMIXFIT_NEG_COV_DIAG 


        self.flags = flags


    def get_ydiff(self, pars):
        """
        Get 1-d vector
            (model-data)/pixerr 
        Also the last len(pars) elements apply the priors
            (pars-prior)/width

        First we apply hard priors on centroid range and determinant(s)
        and demand T,p > 0
        """

        if False:
            print_pars(pars, front="pars: ")
            print_pars(self.psf_pars, front="psf_pars:")
        
        ntot = self.image.size + len(pars)

        if not self.check_hard_priors(pars):
            return zeros(ntot, dtype='f8') + numpy.inf

        ydiff_tot = zeros(ntot, dtype='f8')
        
        model=self._get_model(pars)
        ydiff_tot[0:self.image.size] = (self.image-model).ravel()

        ydiff_tot[0:self.image.size] *= self.ierr

        prior_diff = (self.prior-pars)/self.width
        ydiff_tot[self.image.size:] = prior_diff
        return ydiff_tot


    def check_hard_priors(self, pars):
        wbad,=where(isfinite(pars) == False)
        if wbad.size > 0:
            if self.verbose:
                print >>stderr,'NaN in pars',pars
            return False

        e1=pars[2]
        e2=pars[3]
        e = sqrt(e1**2 + e2**2)
        if (abs(e1) >= 1) or (abs(e2) >= 1) or (e >= 1):
            if self.verbose:
                print >>stderr,'ellip >= 1'
            return False

        gmix_obj=self.pars2gmix(pars)
        gmix=gmix_obj.get_dlist()

        # overall determinant and centroid
        g0 = gmix[0]
        det = g0['irr']*g0['icc'] - g0['irc']**2
        if (det <= 0 
                or pars[0] < 0 or pars[0] > (self.image.shape[0]-1)
                or pars[1] < 0 or pars[1] > (self.image.shape[1]-1)):
            if self.verbose:
                print >>stderr,'bad det or centroid'
            return False

        for g in gmix:
            if self.Tpositive:
                T = g['irr']+g['icc']
                if T <= 0:
                    if self.verbose:
                        print 'bad T or Tfrac: ',T
                    return False
            det=g['irr']*g['icc']-g['irc']**2
            if det <= 0:
                if self.verbose:
                    print_pars(Tfvals,front='bad det: ')
                return False
            if g['p'] <= 0:
                if self.verbose:
                    print 'bad p: ',g['p']
                return False

        """
        if self.psf_gmix:
            gc=gmix_obj.convolve(self.psf_gmix)
            dl=gc.get_dlist()
            for g in dl:
                det=g['irr']*g['icc']-g['irc']**2
                if det <= 0:
                    if self.verbose:
                        print_pars(Tfvals,front='bad det: ')
                    return False
        """

        return True

    def _get_model(self, epars):
        gmix=self._get_gmix(epars)
        model=gmix2image(gmix, self.image.shape, nsub=self.nsub)
        return model

    def _get_gmix(self, epars):
        if self.psf_gmix is not None:
            gmix=self._get_convolved_gmix(epars)
        else:
            gmix=self.pars2gmix(epars)
        return gmix

    def _get_convolved_gmix(self, epars):
        """
        epars must be in e1,e2 space
        """
        gmix0=GMix(epars, type=self.model)
        gmix=gmix0.convolve(self.psf_gmix)
        return gmix




    def pars2gmix(self, pars):
        from . import gmix
        if self.model=='gdev':
            return gmix.GMixDev(pars)
        elif self.model=='gexp':
            return gmix.GMixExp(pars)
        elif self.model=='coellip-Tfrac':
            return gmix.GMix(pars, type=gmix.GMIX_COELLIP_TFRAC)
        elif self.model=='coellip':
            return gmix.GMixCoellip(pars)
        else:
            raise ValueError("bad model: '%s'" % self.model)

    '''
    def scale_leastsq_cov(self, pars, pcov):
        """
        Scale the covariance matrix returned from leastsq; this will
        recover the covariance of the parameters in the right units.
        """
        dof = (self.image.size-len(pars))
        s_sq = (self.get_ydiff(pars)**2).sum()/dof
        return pcov * s_sq 
    '''

    def scale_leastsq_cov(self, pars, pcov):
        """
        Scale the covariance matrix returned from leastsq; this will
        recover the covariance of the parameters in the right units.
        """
        imsize=self.image.size
        ydiff=self.get_ydiff(pars)
        dof = (self.image.size-len(pars))

        # 0:imsize to remove priors
        s_sq = (ydiff[0:imsize]**2).sum()/dof
        return pcov * s_sq 



    def get_model(self):
        return self._get_model(self.pars)

    def get_stats(self):
        gmix=self.get_gmix()
        ivar=self.ierr**2
        stats=calculate_some_stats(self.image, ivar, gmix, len(self.pars))
        return stats

    def get_flags(self):
        return self.flags

    def get_s2n(self):
        """
        This is a raw S/N including all pixels and
        no weighting
        """
        if isinstance(self.ierr, numpy.ndarray):
            raise ValueError("Implement S/N for error image")
        model = self.get_model()
        msum = model.sum()
        s2n = msum/sqrt(model.size)*self.ierr
        return s2n

    def get_weighted_s2n(self):
        """
        weighted S/N including all pixels
        """
        if isinstance(self.ierr, numpy.ndarray):
            raise ValueError("Implement S/N for error image")
        model = self.get_model()

        w2sum=(model**2).sum()
        sum=(model*self.image).sum()

        s2n = sum/sqrt(w2sum)*self.ierr
        return s2n


    def get_numiter(self):
        return self.numiter

    def get_chi2(self):
        ydiff = self.get_ydiff(self.pars)
        return (ydiff**2).sum()
    def get_dof(self):
        return self.image.size-len(self.pars)

    def get_chi2per(self):
        chi2=self.get_chi2()
        return chi2/(self.image.size-len(self.pars))

    def get_gmix(self):
        return self.pars2gmix(self.pars)

    def get_pars(self):
        return self.pars
    def get_perr(self):
        return self.perr
    def get_pcov(self):
        return self.pcov

    def get_result(self):
        gmix=self.get_gmix()
        npars=len(self.prior)
        stats=calculate_some_stats(self.image, 
                                   self.ivar, 
                                   gmix,
                                   npars)

        res={'pars':self.pars,
             'pcov':self.pcov,
             'perr':self.perr,
             'flags':self.flags,
             'numiter':self.numiter}
        res.update(stats)
        return res

    gmix = property(get_gmix)




def quick_fit_psf_coellip(image, skysig, ngauss, ares=None, cen=None):
    """
    Quick fit using GMixFitCoellip.  Guesses look somewhat like turbulent for
    ngauss > 1
    """

    if cen is None and ares is None:
        raise ValueError("send ares= or cen=")
    if ares is None:
        import admom
        ares = admom.admom(image,
                             cen[0],
                             cen[1],
                             sigsky=skysig,
                             guess=2.,
                             nsub=1)

    if ares['whyflag'] != 0:
        return None

    counts=image.sum()
    npars=2*ngauss+4

    if ngauss==1:

        prior=zeros(npars)
        width=zeros(npars) + 100

        Tpsf=ares['Irr']+ares['Icc']

        prior[0]=ares['wrow']
        prior[1]=ares['wcol']
        prior[2]=ares['e1']
        prior[3]=ares['e2']
        prior[4]=ares['Irr'] + ares['Icc']
        prior[5]=counts

        # randomize
        prior[0] += 0.01*srandu()
        prior[1] += 0.01*srandu()

        e1start=prior[2]
        e2start=prior[3]
        prior[2],prior[3] = randomize_e1e2(e1start,e2start)

        prior[4:npars] = prior[4:npars]*(1+0.05*srandu(2*ngauss))

        gm = GMixFitCoellip(image, skysig, prior,width, Tpositive=True)


    elif ngauss==2:

        Texamp=array([12.6,3.8])
        pexamp=array([0.30, 0.70])

        Tfrac=Texamp/Texamp.sum()
        pfrac=pexamp/pexamp.sum()

        prior=zeros(npars)
        width=zeros(npars) + 100

        Tpsf=ares['Irr']+ares['Icc']

        prior[0]=ares['wrow']
        prior[1]=ares['wcol']
        prior[2]=ares['e1']
        prior[3]=ares['e2']
        prior[4:4+2] = Tpsf*Tfrac
        prior[6:6+2] = counts*pfrac

        # randomize
        prior[0] += 0.01*srandu()
        prior[1] += 0.01*srandu()

        e1start=prior[2]
        e2start=prior[3]
        prior[2],prior[3] = randomize_e1e2(e1start,e2start)

        prior[4:npars] = prior[4:npars]*(1+0.05*srandu(2*ngauss))

        gm = GMixFitCoellip(image, skysig, prior,width, Tpositive=True)

    elif ngauss==3:
        # these are good for guessing, but the final answer is
        # often a bit off from here
        Texamp=array([0.46,5.95,2.52])
        pexamp=array([0.1,0.7,0.22])

        Tfrac=Texamp/Texamp.sum()
        pfrac=pexamp/pexamp.sum()


        prior=zeros(npars)
        width=zeros(npars) + 100

        Tpsf=ares['Irr']+ares['Icc']

        prior[0]=ares['wrow']
        prior[1]=ares['wcol']
        prior[2]=ares['e1']
        prior[3]=ares['e2']

        prior[4:4+3] = Tpsf*Tfrac
        prior[7:7+3] = counts*pfrac

        # randomize
        prior[0] += 0.01*srandu()
        prior[1] += 0.01*srandu()
        e1start=prior[2]
        e2start=prior[3]
        prior[2],prior[3] = randomize_e1e2(e1start,e2start)


        prior[4:npars] = prior[4:npars]*(1+0.05*srandu(2*ngauss))

        gm = GMixFitCoellip(image, skysig, prior,width, Tpositive=True)
    else:
        raise ValueError("bad ngauss: %s" % ngauss)

    return gm




def pars2gmix_coellip_Tfrac(pars):
    """
    Convert a parameter array as used for the LM code into a gaussian mixture
    model.  This is for the case using T fractions, easier for priors.

    [cen1,cen2,e1,e2,Tmax,Tri,pi]
    """
    ngauss = (len(pars)-4)/2
    gmix=[]

    row=pars[0]
    col=pars[1]
    e1 = pars[2]
    e2 = pars[3]
    Tmax = pars[4]

    for i in xrange(ngauss):
        d={}

        if i == 0:
            T = Tmax
        else:
            Tfrac = pars[4+i]
            T = Tmax*Tfrac

        p = pars[4+ngauss+i]
        
        d['p'] = p
        d['row'] = row
        d['col'] = col
        d['irr'] = (T/2.)*(1-e1)
        d['irc'] = (T/2.)*e2
        d['icc'] = (T/2.)*(1+e1)
        gmix.append(d)

    return gmix




def pars2gmix_coellip_pick(pars, ptype='Tfrac'):
    """
    Convert a parameter array as used for the LM code into a gaussian mixture
    model.  This is for the case of co-elliptical gaussians.
    """
    if ptype=='Tfrac':
        return pars2gmix_coellip_Tfrac(pars)
    else:
        raise ValueError("ptype should be in ['Tfrac']")


def eta2ellip(eta):
    """
    This is not eta from BJ02
    """
    return (1.+tanh(eta))/2.
def ellip2eta(ellip):
    """
    This is not eta from BJ02
    """
    return arctanh( 2*ellip-1 )

def get_ngauss_coellip(pars):
    return (len(pars)-4)/2


