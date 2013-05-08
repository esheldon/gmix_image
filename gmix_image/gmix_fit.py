from sys import stderr
import inspect
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
from .gmix import GMix, GMixCoellip

LM_MAX_TRY=5

HIGHVAL=9.999e9

GMIXFIT_MAXITER         = 2**0
GMIXFIT_SINGULAR_MATRIX = 2**4
GMIXFIT_NEG_COV_EIG     = 2**5
GMIXFIT_NEG_COV_DIAG    = 2**6
GMIXFIT_NEG_MCOV_DIAG   = 2**7 # the M sub-cov matrix not positive definite
GMIXFIT_MCOV_NOTPOSDEF  = 2**8 # more strict checks on cholesky decomposition
GMIXFIT_CALLS_NOT_CHANGING   = 2**9 # see fmin_cg
GMIXFIT_LOW_S2N = 2**10 # very low S/N for ixx+iyy

notfinite_bit=11
GMIXFIT_NOTFINITE = 2**notfinite_bit # very low S/N for ixx+iyy

# failure before true fit begins, e.g. in _fit_round_fixcen
GMIXFIT_EARLY_FAILURE = 2**30

def run_leastsq(func, guess):
    from scipy.optimize import leastsq
    try:
        lmres = leastsq(func, guess, full_output=1)
    except ValueError as e:
        serr=str(d)
        if 'NaNs' in serr or 'infs' in serr:
            pars=numpy.zeros(len(guess))
            pars[:] = -9999
            cov0=numpy.zeros( (len(guess),len(guess)))
            cov0[:,:] = 9999
            lmres=(pars,cov0,{'nfev':-1},
                   "not finite",notfinite_bit+5)
        else:
            raise e

    return lmres

class GMixFitSimple:
    """
    6 parameter models.  works in g1,g2 space
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
        self.lm_max_try=keys.get('lm_max_try',LM_MAX_TRY)

        self.gprior=keys.get('gprior',None)
        self.gprior_like=keys.get('gprior_like',False)

        self._go()

    def get_result(self):
        return self._result

    def _go(self):
        """
        Do a levenberg-marquardt
        """

        ntry=self.lm_max_try

        for i in xrange(1,ntry+1):

            guess=self._get_guess()

            if self.gprior is None or not self.gprior_like:
                lmres = run_leastsq(self._get_lm_ydiff, guess)
            else:
                lmres = run_leastsq(self._get_lm_ydiff_gprior, guess)

            res=self._calc_lm_results(lmres)

            if res['flags']==0:
                break

        if res['flags'] != 0:
            mess="could not find maxlike after %s tries" % ntry
            print >>stderr,'%s.%s: %s' % (self.__class__,inspect.stack()[0][3],mess)

        res['ntry'] = i
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
            ydiff[:] = HIGHVAL
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
            ydiff[:] = HIGHVAL
            return ydiff

        epars=get_estyle_pars(pars)
        if epars is None:
            ydiff[:] = HIGHVAL
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
            gcov=pcov[2:2+2, 2:2+2]
            res['gcov'] = gcov
            res['gerr'] = sqrt(diag(gcov))
            res['Terr']  = perr[4]
            res['Ts2n']  = res['Tmean']/res['Terr']
            res['arate'] = 1.

            epars=get_estyle_pars(pars)
            gmix=self._get_convolved_gmix(epars)
            stats=calculate_some_stats(self.image, 
                                       self.ivar, 
                                       gmix,
                                       self.npars)

            res.update(stats)

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


class GMixFitMultiBase:
    """
    Much needs to over-ridden
    """
    def __init__(self, **keys):
        """

        This init needs to be over-ridden.  You need to set the number of
        parameters and the lists, e.g. npars.  Below are some examples.  But
        you do want to call this with the keywords

        self.npars=6
        self.imlist=self._get_imlist(imlist)
        self.wtlist=self._get_imlist(wtlist)

        self.jacoblist=jacoblist
        self.psflist=psflist
        self.cen0=cen0  # starting center and center of coord system
        self.model=model

        self._set_im_wt_sums()

        self.nimage=len(self.imlist)
        self.imsize=self.imlist[0].size
        self.totpix=self.nimage*self.imsize

        self._fit_round_fixcen()
        self._fit_full()
        """
        raise RuntimeError("over-ride")

    def get_result(self):
        """
        This is the full result structure made by _calc_lm_results
        """
        return self._result

    def _get_gmix_list(self, pars):
        raise RuntimeError("over-ride")

    def _get_lm_ydiff_gmix_list(self, gmix_list):
        """
        Take in a list of full gmix objects and calculate
        the full ydiff vector across them all
        """
        imsize=self.imsize
        ydiffall = zeros(self.totpix, dtype='f8')
        ydiff = zeros(imsize, dtype='f8')

        for i in xrange(self.nimage):
            im=self.imlist[i]
            wt=self.wtlist[i]
            jacob=self.jacoblist[i]
            gmix=gmix_list[i]
 
            # center of coord system is always the starting center
            _render.fill_ydiff_wt_jacob(im,
                                        wt,
                                        jacob['dudrow'],
                                        jacob['dudcol'],
                                        jacob['dvdrow'],
                                        jacob['dvdcol'],
                                        self.cen0[0], # coord system center
                                        self.cen0[1],
                                        gmix,
                                        ydiff)
            ydiffall[i*imsize:(i+1)*imsize] = ydiff[:]

        return ydiffall

    def _get_convolved_gmix(self, epars, psf):
        """
        Not all children will use this. epars must be in e1,e2 space
        """
        gmix0=GMix(epars, type=self.model)
        gmix=gmix0.convolve(psf)
        return gmix

    def _get_imlist(self, imlist0):
        """
        Make a new list with images that are C contiguous
        and type 'f8' native
        """
        imlist=[]
        for im0 in imlist0:
            im=numpy.array(im0, dtype='f8', order='C', copy=False)
            imlist.append(im)
        return imlist

    def _set_im_wt_sums(self):
        """
        median of the counts across all input images
        """
        clist=numpy.zeros(len(self.imlist))
        wtsum=0.0
        for i,im in enumerate(self.imlist):
            clist[i] = im.sum()
            wtsum += self.wtlist[i].sum()
        
        self.counts=numpy.median(clist)
        self.wtsum=wtsum
        #self.lowest_psum = -5.0*sqrt(1./wtsum)

    def _calc_lm_results(self, lmres):
        """
        This should be totally generic; you need to
        implement _get_gmix_list(pars)
        """
        res={'model':self.model, 'restype': 'lm'}

        pars, pcov0, infodict, errmsg, ier = lmres

        if ier == 0:
            # wrong args, this is a bug
            raise ValueError(errmsg)

        res['pars'] = pars
        res['pcov0'] = pcov0
        res['numiter'] = infodict['nfev']
        res['errmsg'] = errmsg
        res['ier'] = ier

        pcov=None
        perr=None
        gmix_list=None

        if pcov0 is not None:
            gmix_list=self._get_gmix_list(pars)
            pcov = self._scale_leastsq_cov(gmix_list, pcov0)

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

        if res['flags']==0:
            if gmix_list is None:
                gmix_list=self._get_gmix_list(pars)

            stats=self._get_stats(gmix_list)
            res.update(stats)

        return res

    def _get_stats(self, gmix_list):
        from math import log, sqrt
        from . import render
        import scipy.stats

        npars=self.npars

        s2n_numer=0.
        s2n_denom=0.
        loglike=0.

        for i in xrange(self.nimage):
            im=self.imlist[i]
            wt=self.wtlist[i]
            jacob=self.jacoblist[i]
            gmix=gmix_list[i]
 
            tres=render._render.loglike_wt_jacob(im,
                                                 wt,
                                                 jacob['dudrow'],
                                                 jacob['dudcol'],
                                                 jacob['dvdrow'],
                                                 jacob['dvdcol'],
                                                 self.cen0[0], # coord system center
                                                 self.cen0[1],
                                                 gmix)

            tloglike,ts2n_numer,ts2n_denom,tflags=tres

            s2n_numer += ts2n_numer
            s2n_denom += ts2n_denom
            loglike += tloglike
            
        if s2n_denom > 0:
            s2n=s2n_numer/sqrt(s2n_denom)
        else:
            s2n=0.0

        chi2=loglike/(-0.5)
        dof=self.totpix-npars
        chi2per = chi2/dof

        prob = scipy.stats.chisqprob(chi2, dof)

        aic = -2*loglike + 2*npars
        bic = -2*loglike + npars*log(self.totpix)

        return {'s2n_w':s2n,
                'loglike':loglike,
                'chi2per':chi2per,
                'dof':dof,
                'fit_prob':prob,
                'aic':aic,
                'bic':bic}

    def _check_lists(self,*args):
        llen=len(args[0])
        for l in args:
            if len(l) != llen:
                raise ValueError("all lists must be same length")

    def _scale_leastsq_cov(self, gmix_list, pcov):
        """
        Scale the covariance matrix returned from leastsq; this will
        recover the covariance of the parameters in the right units.
        """
        ydiff = self._get_lm_ydiff_gmix_list(gmix_list)
        dof   = self.totpix-self.npars

        s_sq = (ydiff**2).sum()/dof
        return pcov * s_sq 


class GMixFitMultiSimple(GMixFitMultiBase):
    """
    6 parameter models.  works in g1,g2 space.
    Work in sky coords

    Note the starting center will at the same location in each of the
    images and equal cen0.
    """
    def __init__(self, imlist, wtlist, jacoblist, psflist, cen0, model,
                 **keys):

        self.lm_max_try=keys.get('lm_max_try',LM_MAX_TRY)

        self.npars=6
        self.imlist=self._get_imlist(imlist)
        self.wtlist=self._get_imlist(wtlist)
        self.jacoblist=jacoblist
        self.psflist=psflist
        self._check_lists(self.imlist,self.wtlist,self.jacoblist,self.psflist)

        self.cen0=cen0  # starting center and center of coord system
        self.model=model

        self._set_im_wt_sums()

        self.nimage=len(self.imlist)
        self.imsize=self.imlist[0].size
        self.totpix=self.nimage*self.imsize

        self._fit_round_fixcen()
        self._fit_full()

    def get_gmix(self):
        """
        Get the best fit gmix, not convolved
        """
        pars=self._result['pars']
        epars=get_estyle_pars(pars)
        gmix=GMix(epars, type=self.model)
        return gmix

    def _fit_full(self):
        """
        Do a levenberg-marquardt
        """

        if self._rfc_res['flags'] != 0:
            self._result={'flags':self._rfc_res['flags'],
                          'model':self.model,
                          'restype':'lm',
                          'errmsg':'round fixcen fit failed',
                          'numiter':0,
                          'ntry':0}
            return

        ntry=self.lm_max_try
        for i in xrange(1,ntry+1):
            guess=self._get_guess()

            lmres = run_leastsq(self._get_lm_ydiff_full, guess)

            res=self._calc_lm_results(lmres)

            if res['flags']==0:
                break

        if res['flags'] != 0:
            mess="could not find maxlike after %s tries" % ntry
            print >>stderr,'%s.%s: %s' % (self.__class__,inspect.stack()[0][3],mess)

        res['ntry'] = i
        self._result=res

        if 'pars' in res:
            self._add_extra_results()

    def _add_extra_results(self):
        res=self._result
        res['g']=res['pars'][2:2+2].copy()
        res['gcov']=None
        res['gsens']=1.0
        res['Tmean']=res['pars'][4]
        res['Terr']=9999.
        res['Ts2n']=0.0
        res['arate']=1.0
        if res['flags']==0:
            gcov=res['pcov'][2:2+2, 2:2+2].copy()
            res['gcov'] = gcov
            res['gerr'] = sqrt(diag(gcov))
            res['Terr']  = res['perr'][4]
            res['Ts2n']  = res['Tmean']/res['Terr']
            res['arate'] = 1.

    def _fit_round_fixcen(self):
        """
        An initial fit for a round model, forced to be at the starting center

        we fit only for parameters [T,counts]
        """

        ntry=self.lm_max_try
        for i in xrange(1,ntry+1):

            guess=self._get_round_fixcen_guess()
            lmres = run_leastsq(self._get_lm_ydiff_round_fixcen, guess)

            pars, pcov0, infodict, errmsg, ier = lmres
            if ier==0:
                raise ValueError(errmsg)

            if pcov0 is not None and ier <= 4:
                break

        if pcov0 is None or ier > 4:
            mess="could not find maxlike after %s tries" % ntry
            print >>stderr,'%s.%s: %s' % (self.__class__,inspect.stack()[0][3],mess)
            flags=GMIXFIT_EARLY_FAILURE 
        else:
            flags=0

        pcov=None
        if pcov0 is not None:
            epars=self._pars2convert(pars)
            gmix_list=self._get_gmix_list(epars)
            pcov=self._scale_leastsq_cov(gmix_list,pcov0)

        self._rfc_res={'flags':flags,
                       'numiter':infodict['nfev'],
                       'ntry':i,
                       'pars':pars,
                       'pcov0':pcov0,
                       'pcov':pcov,
                       'errmsg':errmsg}

    def get_rfc_result(self):
        return self._rfc_res

    def _get_gmix_list(self, pars):
        """
        Get a list of gmix objects, each convolved with
        the psf in the individual images
        """
        gmix_list=[]
        for psf in self.psflist:
            gmix=self._get_convolved_gmix(pars, psf)
            gmix_list.append(gmix)

        return gmix_list 

    def _pars2convert(self, pars2):
        """
        Put center in uv space at 0,0 which corresponds to cen0 in pixel space

        Set ellipticity to 0,0
        """
        pars=numpy.array([0.0,
                          0.0,
                          0.0,
                          0.0,
                          pars2[0],
                          pars2[1]], dtype='f8')
        return pars

    def _get_lm_ydiff_round_fixcen(self, pars2):
        """
        pars are [T,counts]
        """
        pars=self._pars2convert(pars2)
        return self._get_lm_ydiff_epars(pars)

    def _get_lm_ydiff_full(self, pars):
        """
        pars are [T,counts]
        """
        epars=get_estyle_pars(pars)
        return self._get_lm_ydiff_epars(epars)
    
    def _get_lm_ydiff_epars(self, epars):
        #if epars is None or epars[-1] < self.lowest_psum:

        if epars is None:
            ydiffall = zeros(self.totpix, dtype='f8')
            ydiffall[:] = -HIGHVAL
            return ydiffall

        gmix_list=self._get_gmix_list(epars)

        # inherited
        return self._get_lm_ydiff_gmix_list(gmix_list)

    def _get_round_fixcen_guess(self):
        guess=numpy.array([4.0, self.counts], dtype='f8')
        guess = guess*(1. + 0.1*srandu(2))
        return guess

    def _get_guess(self):
        if self._rfc_res['flags'] != 0:
            raise ValueError("can't fit when round fixcen fails")

        guess=zeros(self.npars)
        
        T0 = self._rfc_res['pars'][0]
        counts0 = self._rfc_res['pars'][1]

        # guess center 0,0 in uv plane
        guess[0]=0.5*srandu()
        guess[1]=0.5*srandu()

        g1rand=0.1*srandu()
        g2rand=0.1*srandu()
        guess[2]=g1rand
        guess[3]=g2rand

        guess[4] = T0*(1 + 0.1*srandu())
        guess[5] = counts0*(1 + 0.1*srandu())

        return guess

class GMixFitMultiMatch(GMixFitMultiSimple):
    """
    fit to the input gaussian mixture, only letting the total flux vary .  The
    center of the model should be relative the (0,0) coordinate center in uv
    space, which corresponds to cen0, the coord system center in pixel coords.
    Make sure cen0 is the same as used for the reference fit!

    You can enter any GMix object.  

    Note the center of the coord system will be at the same location in each
    of the images and equal cen0.  The center in the uv plan will be taken
    from gmix and will not change
    """
    def __init__(self, imlist, wtlist, jacoblist, psflist, cen0, gmix0,
                 **keys):

        self.lm_max_try=keys.get('lm_max_try',LM_MAX_TRY)
        self.npars=1

        self.imlist=self._get_imlist(imlist)
        self.wtlist=self._get_imlist(wtlist)
        self.jacoblist=jacoblist
        self.psflist=psflist
        self.gmix0=gmix0.copy()
        self._check_lists(self.imlist,self.wtlist,self.jacoblist,self.psflist)

        self.cen0=cen0  # starting center and center of coord system

        # we will fix this and only reset the fluxes as we go
        self._set_gmix_list()

        self.nimage=len(self.imlist)
        self.imsize=self.imlist[0].size
        self.totpix=self.nimage*self.imsize
        self.model='amponly'

        self._dofit()


    def _dofit(self):
        """
        Do a levenberg-marquardt
        """

        ntry=self.lm_max_try
        for i in xrange(1,ntry+1):

            guess=self._get_guess()

            lmres = run_leastsq(self._get_lm_ydiff_flux, guess)

            res=self._calc_lm_results(lmres)

            if res['flags']==0:
                break

        if res['flags'] != 0:
            mess="could not find maxlike after %s tries" % ntry
            print >>stderr,'%s.%s: %s' % (self.__class__,inspect.stack()[0][3],mess)

        res['ntry'] = i
        self._result=res

        self._add_extra_results()

    def _add_extra_results(self):
        res=self._result
        res['F']=res['pars'][0]
        res['Ferr']=9999.
        if res['flags']==0:
            res['Ferr']  = res['perr'][0]

    def _get_lm_ydiff_flux(self, pars1):
        """
        pars1 are [counts]
        """
        #if pars1[0] < self.lowest_psum:
        #    ydiffall = zeros(self.totpix, dtype='f8')
        #    ydiffall[:] = -HIGHVAL
        #    return ydiffall

        self._set_psum(pars1)
        return self._get_lm_ydiff_gmix_list(self.gmix_list)

    def _set_psum(self, pars1):
        psum=pars1[0]
        for g in self.gmix_list:
            g.set_psum(psum)
    def _get_gmix_list(self, pars):
        """
        called by the generic code calc_lm_results
        from the base class
        """
        self._set_psum(pars)
        return self.gmix_list
    def _set_gmix_list(self):
        gmix_list=[]
        for psf in self.psflist:
            g = self.gmix0.convolve(psf)
            gmix_list.append(g)
        self.gmix_list=gmix_list


    def _get_guess(self):
        psum=self.gmix0.get_psum()
        guess=numpy.array([psum],dtype='f8')
        guess[0] = guess[0]*(1.+0.05*srandu())
        return guess


class GMixFitMultiCModel(GMixFitMultiBase):
    """
    fit to the input gaussian mixture, only letting the total flux vary .  The
    center of the model should be relative the (0,0) coordinate center in uv
    space, which corresponds to cen0, the coord system center in pixel coords.
    Make sure cen0 is the same as used for the reference fit!

    You can enter any GMix object.  

    Note the center of the coord system will be at the same location in each
    of the images and equal cen0.  The center in the uv plan will be taken
    from gmix and will not change
    """
    def __init__(self, imlist, wtlist, jacoblist, psflist, cen0,
                 gmix_exp, gmix_dev,
                 **keys):

        self.lm_max_try=keys.get('lm_max_try',LM_MAX_TRY)
        self.npars=1

        self._check_lists(imlist,wtlist,jacoblist,psflist)

        self.imlist=self._get_imlist(imlist)
        self.wtlist=self._get_imlist(wtlist)
        self.jacoblist=jacoblist
        self.psflist=psflist

        self.gmix_exp=gmix_exp.copy()
        self.gmix_dev=gmix_dev.copy()

        self.cen0=cen0  # starting center and center of coord system

        # we will fix this and only reset the fluxes as we go
        self._set_gmix_lists()

        self.nimage=len(self.imlist)
        self.imsize=self.imlist[0].size
        self.totpix=self.nimage*self.imsize
        self.model='composite'

        self._dofit()

    def _dofit(self):
        """
        Do a levenberg-marquardt
        """

        ntry=self.lm_max_try
        for i in xrange(1,ntry+1):

            guess=self._get_guess()

            lmres = run_leastsq(self._get_ydiff, guess)

            res=self._calc_lm_results(lmres)

            if res['flags']==0:
                break

        if res['flags'] != 0:
            mess="could not find maxlike after %s tries" % ntry
            print >>stderr,'%s.%s: %s' % (self.__class__,inspect.stack()[0][3],mess)

        res['ntry'] = i
        self._result=res

        self._add_extra_results()

    def _add_extra_results(self):
        res=self._result
        res['fracdev']=res['pars'][0]
        res['fracdev_err']=9999.
        if res['flags']==0:
            res['fracdev_err']  = res['perr'][0]

    def _get_ydiff(self, pars1):
        """
        pars1 are [frac_dev]
        """

        if pars1[0] < 0 or pars1[0] > 1:
            ydiffall = zeros(self.totpix, dtype='f8')
            ydiffall[:] = HIGHVAL
            return ydiffall

        gmix_list=self._get_gmix_list(pars1)
        return self._get_lm_ydiff_gmix_list(gmix_list)


    def _get_gmix_list(self, pars):
        """
        called by the generic code calc_lm_results
        from the base class
        """
        fracdev=pars[0]
        gmix_list=[]
        for i in xrange(len(self.exp_list)):
            exp_flux = (.1-fracdev)*self.exp_fluxes[i]
            dev_flux = fracdev*self.dev_fluxes[i]

            gexp=self.exp_list[i].copy()
            gdev=self.dev_list[i].copy()

            gexp.set_psum(exp_flux)
            gdev.set_psum(dev_flux)

            exp_dlist=gexp.get_dlist()
            dev_dlist=gdev.get_dlist()

            dlist=exp_dlist + dev_dlist

            gm = GMix(dlist)

            gmix_list.append(gm)

        return gmix_list

    def _set_gmix_lists(self):
        exp_list=[]
        dev_list=[]
        exp_fluxes=[]
        dev_fluxes=[]

        for psf in self.psflist:
            gexp = self.gmix_exp.convolve(psf)
            gdev = self.gmix_dev.convolve(psf)
            exp_list.append(gexp)
            dev_list.append(gdev)

            exp_fluxes.append( gexp.get_psum() )
            dev_fluxes.append( gdev.get_psum() )

        self.exp_list=exp_list
        self.dev_list=dev_list

        self.exp_fluxes=exp_fluxes
        self.dev_fluxes=dev_fluxes

    def _get_guess(self):
        guess=numpy.array([0.5],dtype='f8')
        guess[0] = guess[0]*(1.+0.1*srandu())
        return guess

class GMixFitMultiPSFFlux(GMixFitMultiBase):
    """

    Fit for a psfflux and centroid, taking the PSFs models from all the images
    and applying a total flux for them and fitting to the data

    the psf model is that derived from GMixFitPSFJacob, which is currently
    limited to 'coellip'

    cen0 is the pixel coords corresponding to sky coords center, and 0,0 will
    thus be taken as the starting point for the centroid fitting

    the median counts from the images is used for starting guess on
    the flux
    """
    def __init__(self, 
                 imlist,
                 wtlist,
                 jacoblist,
                 gmix_list,
                 cen0, 
                 **keys):

        self.lm_max_try=keys.get('lm_max_try',LM_MAX_TRY)
        self.npars=3

        self.imlist=self._get_imlist(imlist)
        self.wtlist=self._get_imlist(wtlist)
        self.jacoblist=jacoblist
        self._check_lists(self.imlist,self.wtlist,self.jacoblist)

        self.cen0=cen0  # starting center and center of coord system
        self._copy_gmix_list(gmix_list)

        self.nimage=len(self.imlist)
        self.imsize=self.imlist[0].size
        self.totpix=self.nimage*self.imsize

        self.model='psf'
        self._set_im_wt_sums()

        self._dofit()

    def _dofit(self):
        """
        Do a levenberg-marquardt
        """

        ntry=self.lm_max_try
        for i in xrange(1,ntry+1):

            guess=self._get_guess()

            lmres = run_leastsq(self._get_ydiff, guess)

            res=self._calc_lm_results(lmres)

            if res['flags']==0:
                break

        if res['flags'] != 0:
            mess="could not find maxlike after %s tries" % ntry
            print >>stderr,'%s.%s: %s' % (self.__class__,inspect.stack()[0][3],mess)

        res['ntry'] = i
        self._result=res
        self._add_extra_results()

    def _get_ydiff(self, pars3):
        """
        pars3 are [rowcen,colcen,counts]
        """

        #if pars3[2] < self.lowest_psum:
        #    ydiffall = zeros(self.totpix, dtype='f8')
        #    ydiffall[:] = -HIGHVAL
        #    return ydiffall

        self._set_cen_psum(pars3)
        return self._get_lm_ydiff_gmix_list(self.gmix_list)

    def _set_cen_psum(self, pars3):
        new_row=pars3[0]
        new_col=pars3[1]
        new_flux=pars3[2]

        for gm in self.gmix_list:
            gm.set_cen(new_row, new_col)
            gm.set_psum(new_flux)

    def _get_guess(self):
        guess=numpy.zeros(3)
        guess[0:0+2] = 0.5*srandu(2)
        guess[2] = self.counts*(1.+0.05*srandu())
        return guess


    def _add_extra_results(self):
        res=self._result
        res['row'] = res['pars'][0]
        res['col'] = res['pars'][1]
        res['F']=res['pars'][2]

        res['rowerr']=9999.
        res['colerr']=9999.
        res['Ferr']=9999.
        if res['flags']==0:
            res['rowerr']=res['perr'][0]
            res['colerr']=res['perr'][1]
            res['Ferr']  = res['perr'][2]



    def _get_gmix_list(self, pars3):
        """
        called by the generic code calc_lm_results
        from the base class
        """
        self._set_cen_psum(pars3)
        return self.gmix_list

    def _copy_gmix_list(self, glist):
        
        gmix_list=[]
        for g in glist:
            gmix_list.append(g.copy())

        self.gmix_list=gmix_list


class GMixFitPSFJacob(GMixFitMultiSimple):
    """
    Override the parts that require a psf or assume a simple model

    Fit a coelliptical multi-gaussian model

    set cen0 to initial guess in pixel space; it will be the
    center of the coordinate system
    """
    def __init__(self, image, ivar, jacobian, cen0, ngauss, **keys):
        if ngauss > 3:
            raise ValueError("support ngauss>3")

        self.lm_max_try=keys.get('lm_max_try',LM_MAX_TRY)

        self.ngauss=ngauss
        self.model='coellip'

        self.npars=2*ngauss + 4
        self.image=numpy.array(image, dtype='f8', order='C', copy=False)
        self.ivar=float(ivar)

        self.jacobian=jacobian
        self.cen0=cen0  # starting center and center of coord system

        self.imsize=self.image.size
        self.totpix=self.imsize
        self.counts=self.image.sum()

        # lowest allowed amplitide
        #self.lowest_psum = -5.0*sqrt(self.ivar*self.imsize)
        self.pstart=4+self.ngauss

        self._fit_round_fixcen()
        self._fit_full()

    def _get_lm_ydiff_round_fixcen(self, pars2):
        """
        pars are [T,counts]
        """

        # since we are doign coellip, this means fitting
        # a single round gaussian, see GMixFitMultiSimple
        epars=self._pars2convert(pars2)
        return self._get_lm_ydiff_epars(epars)

    def _get_lm_ydiff_full(self, pars):
        epars=get_estyle_pars(pars)
        return self._get_lm_ydiff_epars(epars)
 
    def _get_lm_ydiff_epars(self, epars):
        #if (epars is None 
        #        or epars[self.pstart:].sum() < self.lowest_psum):

        if epars is None:
            ydiff = zeros(self.totpix, dtype='f8')
            ydiff[:] = -HIGHVAL
            return ydiff

        gmix=GMixCoellip(epars)
        return self._get_lm_ydiff_gmix(gmix)


    def _get_lm_ydiff_gmix(self, gmix):
        """
        Take in a list of full gmix objects and calculate
        the full ydiff vector across them all
        """
        ydiff = zeros(self.imsize, dtype='f8')

        # center of coord system is always the starting center
        _render.fill_ydiff_jacob(self.image,
                                 self.ivar,
                                 self.jacobian['dudrow'],
                                 self.jacobian['dudcol'],
                                 self.jacobian['dvdrow'],
                                 self.jacobian['dvdcol'],
                                 self.cen0[0], # coord system center
                                 self.cen0[1],
                                 gmix,
                                 ydiff)

        return ydiff

    def _scale_leastsq_cov(self, gmix_list, pcov):
        """
        Scale the covariance matrix returned from leastsq; this will
        recover the covariance of the parameters in the right units.
        """
        gmix=gmix_list[0]
        ydiff = self._get_lm_ydiff_gmix(gmix)
        dof   = self.totpix-self.npars

        s_sq = (ydiff**2).sum()/dof
        return pcov * s_sq 


    def _get_stats(self, gmix_list):
        from math import log, sqrt
        from . import render
        import scipy.stats

        npars=self.npars

        gmix=gmix_list[0]

        tres=render._render.loglike_jacob(self.image,
                                          self.ivar,
                                          self.jacobian['dudrow'],
                                          self.jacobian['dudcol'],
                                          self.jacobian['dvdrow'],
                                          self.jacobian['dvdcol'],
                                          self.cen0[0], # coord system center
                                          self.cen0[1],
                                          gmix)

        loglike,s2n_numer,s2n_denom,flags=tres
            
        if s2n_denom > 0:
            s2n=s2n_numer/sqrt(s2n_denom)
        else:
            s2n=0.0

        chi2=loglike/(-0.5)
        dof=self.totpix-npars
        chi2per = chi2/dof

        prob = scipy.stats.chisqprob(chi2, dof)

        aic = -2*loglike + 2*npars
        bic = -2*loglike + npars*log(self.totpix)

        return {'s2n_w':s2n,
                'loglike':loglike,
                'chi2per':chi2per,
                'dof':dof,
                'fit_prob':prob,
                'aic':aic,
                'bic':bic}


    def _get_guess(self):
        if self._rfc_res['flags'] != 0:
            raise ValueError("can't fit when round fixcen fails")

        npars=self.npars
        ngauss=self.ngauss
        guess=zeros(npars)
        
        T0 = self._rfc_res['pars'][0]
        counts0 = self._rfc_res['pars'][1]

        # guess center 0,0 in uv plane
        guess[0]=0.0
        guess[1]=0.0

        g1rand=0.1*srandu()
        g2rand=0.1*srandu()
        guess[2]=g1rand
        guess[3]=g2rand

        if ngauss==1:
            guess[4] = T0*(1 + 0.05*srandu())
            guess[5] = counts0*(1 + 0.05*srandu())
        else:
            if ngauss==2:
                Texamp=array([12.6,3.8])
                pexamp=array([0.30, 0.70])

                Tfrac=Texamp/Texamp.sum()
                pfrac=pexamp/pexamp.sum()

            elif ngauss==3:
                Texamp=array([0.46,5.95,2.52])
                pexamp=array([0.1,0.7,0.22])

                Tfrac=Texamp/Texamp.sum()
                pfrac=pexamp/pexamp.sum()
            else:
                raise ValueError("support ngauss>3")

            guess[4:4+ngauss] = T0*Tfrac 
            guess[4+ngauss:] = counts0*pfrac

            guess[4:npars] = guess[4:npars]*(1+0.05*srandu(2*ngauss))

        return guess

    def _get_gmix_list(self, pars):
        """
        called by the generic code calc_lm_results
        from the base class
        """
        epars=get_estyle_pars(pars)
        gmix=GMixCoellip(epars)
        return [gmix]


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
                 estyle='e',
                 verbose=False):
        self.image=image
        self.pixerr=pixerr
        self.ierr = 1./pixerr
        self.ivar=self.ierr**2
        self.prior=prior
        self.width=width
        self.estyle=estyle

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

        res = run_leastsq(self.get_ydiff, self.prior)

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

    def _get_model(self, pars):
        gmix=self._get_gmix(pars)
        model=gmix2image(gmix, self.image.shape, nsub=self.nsub)
        return model

    def _get_gmix(self, pars):
        if self.psf_gmix is not None:
            gmix=self._get_convolved_gmix(pars)
        else:
            gmix=self.pars2gmix(pars)
        return gmix

    def _get_convolved_gmix(self, pars):
        gmix0=self.pars2gmix(pars)
        gmix=gmix0.convolve(self.psf_gmix)
        return gmix


    def pars2gmix(self, pars):
        from . import gmix

        if self.estyle=='g':
            epars=get_estyle_pars(pars)
        else:
            epars=pars

        if self.model=='gdev':
            return gmix.GMixDev(epars)
        elif self.model=='gexp':
            return gmix.GMixExp(epars)
        elif self.model=='coellip-Tfrac':
            return gmix.GMix(epars, type=gmix.GMIX_COELLIP_TFRAC)
        elif self.model=='coellip':
            return gmix.GMixCoellip(epars)
        else:
            raise ValueError("bad model: '%s'" % self.model)


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

        Tmean=gmix.get_T()
        res={'pars':self.pars,
             'pcov':self.pcov,
             'perr':self.perr,
             'Tmean':Tmean,
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

    prior=zeros(npars)
    width=zeros(npars) + 1.e6
    if ngauss==1:

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

def test_simple(s2n=100.,
                psf_sigma=2.0,
                sigma=4.0,
                counts=100.,
                model='gexp',
                s2n_method='matched'):
    import fimage
    import admom

    Tpsf=2*psf_sigma**2
    T=2*sigma**2

    g1=0.0
    g2=0.0

    # centers are actuall ignored when creating the convolved
    # image
    pars=numpy.array([-1., -1., g1, g2, T, counts])
    epars=get_estyle_pars(pars)
    gmix=GMix(epars,type=model)

    e1psf=-0.05
    e2psf=-0.07
    gmix_psf_prepix=GMix([-1., -1., e1psf, e2psf, Tpsf, 1.0],type='coellip')

    ci = fimage.convolved.ConvolverGMix(gmix, gmix_psf_prepix)
    cin = fimage.convolved.NoisyConvolvedImage(ci, s2n, 1.e8,
                                               s2n_method=s2n_method)

    ivar=1./cin['skysig']**2


    cen=ci['cen']

    ngauss_psf=2
    gm_psf=quick_fit_psf_coellip(cin.psf, cin['skysig_psf'], ngauss_psf,
                                 cen=cen)
    gmix_psf=gm_psf.get_gmix()

    image = cin.image
    ares = admom.admom(image,
                       cen[0],
                       cen[1],
                       sigsky=cin['skysig'],
                       guess=2.,
                       nsub=1)

    # starting guess in pixel coords, origin in uv space
    gm=GMixFitSimple(image,
                     ivar,
                     gmix_psf,
                     model,
                     ares)
    return gm.get_result()

def test_multi(s2n=100.,
               sigma=4.0,
               psf_s2n=1.e8,
               psf_sigma=2.0,
               psf_ngauss_fit=2,
               g1=0.0,
               g2=0.0,
               counts=100., 
               model='gexp',
               nimages=10,
               s2n_method='matched',
               scale=0.27,
               theta=20.0,
               randpsf=False):
    import fimage
    import math
    from math import cos,sin
    from lensing.util import rotate_shape

    thetarad=math.radians(theta)
    ctheta=cos(thetarad)
    stheta=sin(thetarad)
    c2theta=cos(2*thetarad)
    s2theta=sin(2*thetarad)

    # rotate backwards to the pixel coords
    g1pix,g2pix=rotate_shape(g1,g2,-theta)

    s2n_per=s2n/sqrt(nimages)

    # for simulation, in pixels
    Tpsf0_pix=2*(psf_sigma/scale)**2
    Tpix=2*(sigma/scale)**2
    counts_pix=counts/(scale*scale)

    # centers are actually ignored when creating the convolved
    # image
    pars=numpy.array([-1., -1., g1pix, g2pix, Tpix, counts_pix])
    epars=get_estyle_pars(pars)
    gmix=GMix(epars,type=model)

    imlist=[]
    wtlist=[]
    psflist=[]
    jacoblist=[]

    s2n_uw_sum=0.
    aperture=1.5/scale # 1.5 arcsec diameter
    rad=aperture/2.

    for i in xrange(nimages):
        jacob={'dudrow':ctheta*scale, 'dudcol':-stheta*scale,
               'dvdrow':stheta*scale, 'dvdcol': ctheta*scale}
        if randpsf:
            e1psf=0.1*srandu()
            e2psf=0.1*srandu()
            Tpsf=Tpsf0_pix*(1.+0.1*srandu())
            gmix_psf_nopix=GMix([0., 0., e1psf, e2psf, Tpsf, 1.0],type='turb')
        else:
            e1psf=-0.05
            e2psf=-0.07
            Tpsf=Tpsf0_pix
            gmix_psf_nopix=GMix([0., 0., e1psf, e2psf, Tpsf, 1.0],type='turb')

        ci = fimage.convolved.ConvolverGMix(gmix, gmix_psf_nopix)
        cin = fimage.convolved.NoisyConvolvedImage(ci, s2n_per, psf_s2n,
                                                   s2n_method=s2n_method)
        s2n_uw_sum += fimage.noise.get_s2n_uw_aperture(ci.image, cin['skysig'],
                                                       ci['cen'], rad)

        psf_ivar=1./cin['skysig_psf']**2
        gm_psf=GMixFitPSFJacob(cin.psf, psf_ivar, jacob, cin['cen'], psf_ngauss_fit)
        gmix_psf=gm_psf.get_gmix()

        wt=cin.image.copy()
        wt = 0.0*wt + 1./cin['skysig']**2

        imlist.append(cin.image) 
        wtlist.append(wt)
        psflist.append(gmix_psf)
        jacoblist.append(jacob)
        
    s2n_uw15_per = s2n_uw_sum/nimages
    s2n_uw15 = s2n_uw15_per*sqrt(nimages)

    # starting guess in pixel coords, origin in uv space
    cen0=ci['cen']

    gm=GMixFitMultiSimple(imlist,
                          wtlist,
                          jacoblist,
                          psflist,
                          cen0,
                          model)
    return gm.get_result(), s2n_uw15

def _make_data(jacob, gmix, Tpsf0_pix, s2n, psf_s2n, psf_ngauss_fit,rad,
              s2n_method='matched'):
    import fimage
    e1psf=0.1*srandu()
    e2psf=0.1*srandu()
    Tpsf=Tpsf0_pix*(1.+0.1*srandu())
    gmix_psf_nopix=GMix([-1., -1., e1psf, e2psf, Tpsf, 1.0],type='turb')

    ci = fimage.convolved.ConvolverGMix(gmix, gmix_psf_nopix)
    cin = fimage.convolved.NoisyConvolvedImage(ci, s2n, psf_s2n,
                                                s2n_method=s2n_method)
    s2n_uw = fimage.noise.get_s2n_uw_aperture(ci.image, cin['skysig'],
                                              ci['cen'], rad)

    psf_ivar=1./cin['skysig_psf']**2
    gm_psf=GMixFitPSFJacob(cin.psf, psf_ivar, jacob, cin['cen'], psf_ngauss_fit)
    gmix_psf=gm_psf.get_gmix()

    wt=cin.image.copy()
    wt = 0.0*wt + 1./cin['skysig']**2

    return cin.image, wt, gmix_psf, s2n_uw, cin['cen']

                
def test_multi_color(s2n=100.,
                     sigma=4.0,
                     psf_s2n=1.e8,
                     psf_sigma=2.0,
                     psf_ngauss_fit=2,
                     g1=0.0,
                     g2=0.0,
                     counts1=100., 
                     counts2=250.,
                     model='gexp',
                     nimages=10,
                     scale=0.27,
                     sigratio=0.9,
                     eratio=0.9,
                     eoffset=0.01,
                     s2n_method='matched'):
    import fimage

    s2n_per=s2n/sqrt(nimages)

    sigma_2 = sigratio*sigma

    # for simulation, in pixels
    Tpsf0_pix=2*(psf_sigma/scale)**2
    Tpix=2*(sigma/scale)**2
    counts1_pix=counts1/(scale*scale)
    counts2_pix=counts2/(scale*scale)

    Tpix_2 = 2*(sigma_2/scale)**2

    g1_2 = eratio*g1+eoffset
    g2_2 = eratio*g2-eoffset

    # centers are actually ignored when creating the convolved
    # image
    pars1=numpy.array([-1., -1., g1, g2, Tpix, counts1_pix])
    pars2=numpy.array([-1., -1., g1_2, g2_2, Tpix_2, counts2_pix])

    epars1=get_estyle_pars(pars1)
    epars2=get_estyle_pars(pars2)
    gmix1=GMix(epars1,type=model)
    gmix2=GMix(epars2,type=model)

    imlist1=[]
    wtlist1=[]
    psflist1=[]
    jacoblist1=[]

    imlist2=[]
    wtlist2=[]
    psflist2=[]
    jacoblist2=[]

    s2n_uw_sum=0.
    aperture=1.5/scale # 1.5 arcsec diameter
    rad=aperture/2.
    for i in xrange(nimages):

        # image 1
        jacob1={'dudrow':scale, 'dudcol':0.0,
                'dvdrow':0.0,   'dvdcol':scale}
        im1, wt1, gmix_psf1,s2n_uw1,cen1=\
                _make_data(jacob1,gmix1,Tpsf0_pix, s2n_per,psf_s2n,
                           psf_ngauss_fit,rad, s2n_method=s2n_method)
        jacob2={'dudrow':scale, 'dudcol':0.0,
                'dvdrow':0.0,   'dvdcol':scale}
        im2, wt2, gmix_psf2,s2n_uw2,cen2=\
                _make_data(jacob2,gmix2,Tpsf0_pix, s2n_per,psf_s2n,
                           psf_ngauss_fit,rad)

        s2n_uw_sum += s2n_uw1

        imlist1.append(im1) 
        wtlist1.append(wt1)
        psflist1.append(gmix_psf1)
        jacoblist1.append(jacob1)

        imlist2.append(im2) 
        wtlist2.append(wt2)
        psflist2.append(gmix_psf2)
        jacoblist2.append(jacob2)

    s2n_uw15_per = s2n_uw_sum/nimages
    s2n_uw15 = s2n_uw15_per*sqrt(nimages)

    # starting guess in pixel coords, origin in uv space
    # all are the same for this simple test
    gm1=GMixFitMultiSimple(imlist1,
                           wtlist1,
                           jacoblist1,
                           psflist1,
                           cen1,
                           model)
    res1=gm1.get_result()

    if res1['flags'] != 0:
        return {}, {}, -1

    gm2=GMixFitMultiMatch(imlist2,
                          wtlist2,
                          jacoblist2,
                          psflist2,
                          cen2,
                          gm1.get_gmix())
    res2=gm2.get_result()
    return res1,res2,s2n_uw15


def test_psfflux_star(s2n=100.,
                      sigma=2.0,
                      counts=100., 
                      nimages=10,
                      scale=0.27,
                      ngauss_fit=2):
    """
    test recover of the PSFFlux from a turb PSF using a coelliptical gauss fit

    The psf shape is fit from a higher s/n image, then an image at the
    requested s/n per SE image is created and used for the fit

    """
    import fimage

    s2n_per=s2n/sqrt(nimages)

    # for simulation, in pixels
    sigma_pix=sigma/scale
    Tpix=2*(sigma_pix)**2
    counts_pix=counts/(scale*scale)

    padding=5.
    dim=2*5*sigma_pix
    if (dim % 2) == 0:
        dim += 1
    dims = [dim]*2
    cen=[(dim-1)/2]*2

    imlist=[]
    wtlist=[]
    psflist=[]
    jacoblist=[]

    aperture=1.5/scale # 1.5 arcsec diameter
    rad=aperture/2.
    for i in xrange(nimages):
        jacob={'dudrow':scale, 'dudcol':0.0,
               'dvdrow':0.0,   'dvdcol':scale}

        e1=0.1*srandu()
        e2=0.1*srandu()
        T=Tpix*(1.+0.1*srandu())
        gmix_nopix=GMix([cen[0], cen[1], e1, e2, T, counts_pix],type='turb')

        im=gmix2image(gmix_nopix, dims, nsub=16)
        #im_lown,skysig_lown=fimage.noise.add_noise_admom(im, 1.e6)
        #im_highn,skysig_highn=fimage.noise.add_noise_admom(im, s2n_per)
        im_lown,skysig_lown=fimage.noise.add_noise_matched(im, 1.e6, cen)
        im_highn,skysig_highn=fimage.noise.add_noise_matched(im, s2n_per, cen)

        # get psf model from low noise image
        ivar_lown=1./skysig_lown**2
        gm_psf=GMixFitPSFJacob(im_lown, ivar_lown, jacob, cen, ngauss_fit)

        gmix_psf=gm_psf.get_gmix()

        ivar_highn=1./skysig_highn**2
        wt=im_highn.copy()
        wt = 0.0*wt + ivar_highn

        imlist.append(im_highn) 
        wtlist.append(wt)
        psflist.append(gmix_psf)
        jacoblist.append(jacob)

    # starting guess in pixel coords, origin in uv space
    gm=GMixFitMultiPSFFlux(imlist,
                           wtlist,
                           jacoblist,
                           psflist,
                           cen)
    return gm.get_result()


def test_cmodel(s2n=100.,
                exp_sigma=4.0,
                dev_sigma=7.0,
                g1exp=0.0,
                g2exp=0.0,
                g1dev=0.0,
                g2dev=0.0,

                counts=100.,
                fracdev_true=0.4,

                psf_s2n=1.e8,
                psf_sigma=2.0,
                psf_ngauss_fit=2,
                nimages=10,
                scale=0.27,
                s2n_method='matched'):
    import fimage

    s2n_per=s2n/sqrt(nimages)

    # for simulation, in pixels
    Tpsf0_pix=2*(psf_sigma/scale)**2

    exp_Tpix=2*(exp_sigma/scale)**2
    dev_Tpix=2*(dev_sigma/scale)**2

    counts_pix=counts/(scale*scale)
    exp_counts_pix=(1.-fracdev_true)*counts_pix
    dev_counts_pix=fracdev_true*counts_pix

    # centers are actually ignored when creating the convolved
    # image
    exp_pars=numpy.array([-1., -1., g1exp, g2exp, exp_Tpix, exp_counts_pix])
    dev_pars=numpy.array([-1., -1., g1dev, g2dev, dev_Tpix, dev_counts_pix])

    exp_epars=get_estyle_pars(exp_pars)
    dev_epars=get_estyle_pars(dev_pars)
    exp_gmix=GMix(exp_epars,type='exp')
    dev_gmix=GMix(dev_epars,type='dev')
    

    exp_dlist=exp_gmix.get_dlist()
    dev_dlist=dev_gmix.get_dlist()
    dlist=exp_dlist + dev_dlist
    #pprint.pprint(dlist)

    gmix = GMix(dlist)

    imlist=[]
    wtlist=[]
    psflist=[]
    jacoblist=[]

    s2n_uw_sum=0.
    aperture=1.5/scale # 1.5 arcsec diameter
    rad=aperture/2.
    for i in xrange(nimages):
        jacob={'dudrow':scale, 'dudcol':0.0,
               'dvdrow':0.0,   'dvdcol':scale}

        e1psf=0.1*srandu()
        e2psf=0.1*srandu()
        Tpsf=Tpsf0_pix*(1.+0.1*srandu())
        gmix_psf_nopix=GMix([-1., -1., e1psf, e2psf, Tpsf, 1.0],type='turb')

        ci = fimage.convolved.ConvolverGMix(gmix, gmix_psf_nopix)
        cin = fimage.convolved.NoisyConvolvedImage(ci, s2n_per, psf_s2n,
                                                   s2n_method=s2n_method)
        #import images
        #images.multiview(ci.image)
        #images.view(cin.image)
        #stop
        s2n_uw_sum += fimage.noise.get_s2n_uw_aperture(ci.image, cin['skysig'],
                                                       ci['cen'], rad)

        psf_ivar=1./cin['skysig_psf']**2
        gm_psf=GMixFitPSFJacob(cin.psf, psf_ivar, jacob, cin['cen'], psf_ngauss_fit)
        gmix_psf=gm_psf.get_gmix()

        wt=cin.image.copy()
        wt = 0.0*wt + 1./cin['skysig']**2

        imlist.append(cin.image) 
        wtlist.append(wt)
        psflist.append(gmix_psf)
        jacoblist.append(jacob)


    s2n_uw15_per = s2n_uw_sum/nimages
    s2n_uw15 = s2n_uw15_per*sqrt(nimages)

    cen0=ci['cen']

    defres=( {}, {}, {}, -1 )
    # starting guess in pixel coords, origin in uv space
    # all are the same for this simple test
    gm_expfit=GMixFitMultiSimple(imlist,
                                 wtlist,
                                 jacoblist,
                                 psflist,
                                 cen0,
                                 'gexp')
    exp_res=gm_expfit.get_result()
    if exp_res['flags'] != 0:
        return defres

    gm_devfit=GMixFitMultiSimple(imlist,
                                 wtlist,
                                 jacoblist,
                                 psflist,
                                 cen0,
                                 'gdev')
    dev_res=gm_devfit.get_result()
    if dev_res['flags'] != 0:
        print dev_res['errmsg']
        return defres


    exp_gmix=gm_expfit.get_gmix()
    dev_gmix=gm_devfit.get_gmix()

    gm=GMixFitMultiCModel(imlist,
                          wtlist,
                          jacoblist,
                          psflist,
                          cen0,
                          exp_gmix,
                          dev_gmix)

    res=gm.get_result()
    return exp_res,dev_res,res,s2n_uw15


