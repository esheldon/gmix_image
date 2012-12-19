import math
import numpy
from numpy import sqrt, log, log10, zeros, \
        where, array, diag, median
import esutil as eu
from esutil.random import srandu
from esutil.misc import wlog
from esutil.random import LogNormal

from .util import print_pars, randomize_e1e2
from .gmix import GMix, GMixExp, GMixDev, GMixCoellip, gmix2pars

from . import render
from .render import gmix2image

from lensing.shear import Shear

LOWVAL=-9999.9e9

class MixMC:
    def __init__(self, image, ivar, psf, cenprior, T, gprior, model,
                 **keys):
        """
        mcmc sampling of posterior.

        parameters
        ----------
        image:
            sky subtracted image as a numpy array
        ivar:
            1/(Error per pixel)**2
        psf:
            The psf gaussian mixture
        cenprior:
            The center prior object.
        T:
            Starting value for ixx+iyy of main component
            or a LogNormal object
        gprior:
            The prior on the g1,g2 surface.
        model:
            Type of model, gexp, gdev, gauss, coellip

        nwalkers: optional
            Number of walkers, default 20
        nstep: optional
            Number of steps in MCMC chain, default 200
        burnin: optional
            Number of burn in steps, default 400
        mca_a: optional
            For affine invariant chain, default 2
        iter: optional
            Iterate until acor is OK, default True
        start_pars: optional
            Starting central positions for parameters.  Only the
            g1,g2 and total flux values are used.
        """
        
        self.make_plots=False

        # cen1,cen2,e1,e2,T,p
        self.npars=6

        self.image=image
        self.ivar=float(ivar)
        self._set_psf(psf)
        self.cenprior=cenprior
        self.T=T
        self.gprior=gprior
        self.model=model

        self.nwalkers=keys.get('nwalkers',20)
        self.nstep=keys.get('nstep',200)
        self.burnin=keys.get('burnin',400)
        self.mca_a=keys.get('mca_a',2.0)
        self.iter=keys.get('iter',True)
        self.start_pars=keys.get('start_pars',None)


        if isinstance(T, LogNormal):
            self.T_is_prior=True
        else:
            self.T_is_prior=False

        self.tpars=zeros(6,dtype='f8')

        self._go()

    def get_result(self):
        return self._result

    def get_maxlike_model(self):
        """
        Get the model representing the maximum likelihood point in the chain
        Is this useful?
        """
        w=self.lnprobs.argmax()
        pars = self.trials[w,:].copy()

        e1,e2,ok=g1g2_to_e1e2(pars[2],pars[3])
        if not ok:
            raise ValueError("bad e1,e2")

        pars[2],pars[3]=e1,e2
        gmix=self._get_convolved_gmix(pars)
        model=gmix2image(gmix, self.image.shape)
        return model


    def _go(self):
        import emcee

        self.sampler = emcee.EnsembleSampler(self.nwalkers, 
                                             self.npars, 
                                             self._calc_lnprob,
                                             a=self.mca_a)
        
        self._do_trials()

        self.trials  = self.sampler.flatchain

        lnprobs = self.sampler.lnprobability.reshape(self.nwalkers*self.nstep)
        self.lnprobs = lnprobs - lnprobs.max()

        # get the expectation values, sensitivity and errors
        self._calc_result()

        if self.make_plots:
            self._doplots()

    def _do_trials(self):
        sampler=self.sampler
        guess=self._get_guess()

        pos, prob, state = sampler.run_mcmc(guess, self.burnin)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, self.nstep)
        if self.iter:
            while True:
                try:
                    acor=sampler.acor
                    tau = (sampler.acor/self.burnin).max()
                    if tau > 0.1:
                        wlog("tau",tau,"greater than 0.1")
                    else:
                        break
                except:
                    # something went wrong with acor, run some more
                    pass
                pos, prob, state = sampler.run_mcmc(pos, self.nstep)


    def _calc_lnprob(self, pars):
        """
        pars are [g1,g2,T]

        """
        # hard priors first

        T=pars[4]

        g1,g2=pars[2],pars[3]

        e1,e2,ok = g1g2_to_e1e2(g1,g2)
        if not ok:
            return LOWVAL

        if T < 0:
            return LOWVAL

        self.tpars[0] = pars[0]
        self.tpars[1] = pars[1]
        self.tpars[2] = e1
        self.tpars[3] = e2
        self.tpars[4] = T
        self.tpars[5:] = pars[5:]

        logprob = self._get_loglike_c(self.tpars)

        gp = self._get_lngprior(g1,g2)
        logprob += gp

        cp = self.cenprior.lnprob(pars[0:2])
        logprob += cp

        if self.T_is_prior:
            Tp = self.T.lnprob(T)
            logprob += Tp

        return logprob

 
    def _get_loglike_c(self, pars):
        gmix=self._get_convolved_gmix(pars)

        loglike,s2n,flags=\
            render._render.loglike(self.image, 
                                   gmix,
                                   self.ivar)

        if flags != 0:
            return LOWVAL
        return loglike

    def _get_lngprior(self, g1, g2):
        g=sqrt(g1**2 + g2**2)
        gp = self.gprior.prior_gabs_scalar(g)
        if gp > 0:
            gp = log(gp)
        else:
            gp=LOWVAL
        return gp

    def _get_convolved_gmix(self,pars):
        """
        This should have T linear
        """
        if self.model=='gexp':
            gmix0=GMixExp(pars)
        elif self.model=='gdev':
            gmix0=GMixDev(pars)
        elif self.model=='gauss':
            gmix0=GMixCoellip(pars)
        elif self.model=='coellip':
            raise ValueError("use NGaussFitter")
        else:
            raise ValueError("bad model: '%s'" % self.model)
        gmix=gmix0.convolve(self.psf_gmix)
        return gmix


    def _calc_result(self):
        """
        We marginalize over all parameters but g1,g2, which
        are index 0 and 1 in the pars array
        """
        import mcmc

        g=zeros(2)
        gcov=zeros((2,2))
        gsens = zeros(2)

        g1vals=self.trials[:,2]
        g2vals=self.trials[:,3]

        prior = self.gprior(g1vals,g2vals)
        dpri_by_g1 = self.gprior.dbyg1(g1vals,g2vals)
        dpri_by_g2 = self.gprior.dbyg2(g1vals,g2vals)

        psum = prior.sum()

        # prior is already in the distribution of
        # points.  This is simpler for most things but
        # for sensitivity we need a factor of (1/P)dP/de

        pars,pcov = mcmc.extract_stats(self.trials)
        #g, gcov = mcmc.extract_stats(self.trials[:,2:2+2])

        g[:] = pars[2:4]
        gcov[:,:] = pcov[2:4, 2:4]

        g1diff = g[0]-g1vals
        g2diff = g[1]-g2vals

        w,=where(prior > 0)
        if w.size == 0:
            raise ValueError("no prior values > 0!")

        gsens[0]= 1.-(g1diff[w]*dpri_by_g1[w]/prior[w]).mean()
        gsens[1]= 1.-(g2diff[w]*dpri_by_g2[w]/prior[w]).mean()
 
        arates = self.sampler.acceptance_fraction
        arate = arates.mean()

        # weighted s/n based on the most likely point
        s2n,loglike,chi2per,dof,prob=self._calculate_maxlike_stats()

        Tmean=pars[4]
        Terr=sqrt(pcov[4,4])
        Ts2n=pars[4]/sqrt(pcov[4,4])

        self._result={'model':self.model,
                      'g':g,
                      'gcov':gcov,
                      'gsens':gsens,
                      'pars':pars,
                      'pcov':pcov,
                      'Tmean':Tmean,
                      'Terr':Terr,
                      'Ts2n':Ts2n,
                      'arate':arate,
                      's2n_w':s2n,
                      'loglike':loglike,
                      'chi2per':chi2per,
                      'dof':dof,
                      'fit_prob':prob}

    def _calculate_maxlike_stats(self):
        """
        Stats Based on the most likely point
        """
        import scipy.stats

        w=self.lnprobs.argmax()
        pars = self.trials[w,:].copy()
        e1,e2,ok=g1g2_to_e1e2(pars[2],pars[3])
        if not ok:
            raise ValueError("bad e1,e2")
        pars[2],pars[3]=e1,e2

        gmix=self._get_convolved_gmix(pars)

        loglike,s2n,flags=\
            render._render.loglike(self.image, 
                                   gmix,
                                   self.ivar)
        chi2=loglike/(-0.5)
        dof=self.image.size-pars.size
        chi2per = chi2/dof

        prob = scipy.stats.chisqprob(chi2, dof)
        return s2n, loglike, chi2per, dof, prob

    def _get_guess(self):
        guess=zeros( (self.nwalkers,self.npars) )

        guess[:,0]=self.cenprior.cen[0] + 0.01*srandu(self.nwalkers)
        guess[:,1]=self.cenprior.cen[1] + 0.01*srandu(self.nwalkers)

        if self.start_pars is not None:
            sh=Shear(e1=self.start_pars[2],e2=self.start_pars[3])
            for i in xrange(self.nwalkers):
                e1s,e2s=randomize_e1e2(sh.g1,sh.g2,width=0.05)
                guess[i,2] = e1s
                guess[i,3] = e2s
        else:
            # (0,0) with some scatter
            guess[:,2]=0.1*srandu(self.nwalkers)
            guess[:,3]=0.1*srandu(self.nwalkers)

        # guess for T is self.T with scatter
        if self.T_is_prior:
            T=self.T.mean
            guess[:,4] = T*(1 + 0.1*srandu(self.nwalkers))
        else:
            T=self.T
            guess[:,4] = T*(1 + 0.1*srandu(self.nwalkers))

        # first guess at amp is the total flux
        if self.start_pars is not None:
            pguess=self.start_pars[5]
            guess[:,5] = pguess*(1+0.1*srandu(self.nwalkers))
        else:
            imtot=self.image.sum()
            guess[:,5] = imtot*(1 + 0.1*srandu(self.nwalkers))

        return guess

    def _set_psf(self, psf):
        if psf is not None:
            self.psf_gmix = GMix(psf)
            self.psf_pars = gmix2pars(self.psf_gmix)
        else:
            self.psf_gmix = None
            self.psf_pars = None


    def _doplots(self):

        import mcmc
        import biggles
        biggles.configure("default","fontsize_min",1.2)
        tab=biggles.Table(6,2)

        cen1vals=self.trials[:,0]
        cen2vals=self.trials[:,1]
        Tvals=self.trials[:,4]
        g1vals=self.trials[:,2]
        g2vals=self.trials[:,3]
        g1lab=r'$g_1$'
        g2lab=r'$g_2$'

        ampvals=self.trials[:,5]

        ind=numpy.arange(g1vals.size)

        burn_cen=biggles.FramedPlot()
        cen1p=biggles.Curve(ind, cen1vals, color='blue')
        cen2p=biggles.Curve(ind, cen2vals, color='red')
        cen1p.label=r'$x_1$'
        cen2p.label=r'$x_2$'
        burn_cen.add(cen1p)
        burn_cen.add(cen2p)
        key=biggles.PlotKey(0.9,0.9,[cen1p,cen2p],halign='right')
        burn_cen.add(key)
        burn_cen.ylabel='cen'

        burn_g1=biggles.FramedPlot()
        burn_g1.add(biggles.Curve(ind, g1vals))
        burn_g1.ylabel=r'$g_1$'

        burn_g2=biggles.FramedPlot()
        burn_g2.add(biggles.Curve(ind, g2vals))
        burn_g2.ylabel=r'$g_2$'

        burn_T=biggles.FramedPlot()
        burn_T.add(biggles.Curve(ind, Tvals))
        burn_T.ylabel='T'

        burn_amp=biggles.FramedPlot()
        burn_amp.add(biggles.Curve(ind, ampvals))
        burn_amp.ylabel='Amplitide'



        likep = biggles.FramedPlot()
        likep.add( biggles.Curve(ind, self.lnprobs) )
        likep.ylabel='ln( prob )'


        g = self._result['g']
        gcov = self._result['gcov']
        errs = sqrt(diag(gcov))

        res=self.get_result()
        print 's2n weighted maxlike:',res['s2n_w']
        print 'chi^2/dof: %.3f/%i = %f' % (res['chi2per']*res['dof'],res['dof'],res['chi2per'])
        print 'prob:',res['fit_prob']
        print 'acceptance rate:',res['arate']
        print 'T:  %.16g +/- %.16g' % (Tvals.mean(), Tvals.std())

        print_pars(self._result['pars'])
        print_pars(sqrt(diag(self._result['pcov'])))
        print 'g1: %.16g +/- %.16g' % (g[0],errs[0])
        print 'g2: %.16g +/- %.16g' % (g[1],errs[1])
        print 'median g1:  %.16g ' % median(g1vals)
        print 'g1sens:',self._result['gsens'][0]
        print 'g2sens:',self._result['gsens'][1]

        g0=self._result['g0']
        err0=sqrt(diag(self._result['gcov0']))
        print 'g1_0: %.16g +/- %.16g' % (g0[0],err0[0])
        print 'g2_0: %.16g +/- %.16g' % (g0[1],err0[1])

        cenw = cen1vals.std()
        cen_bsize=cenw*0.2
        hplt_cen0 = eu.plotting.bhist(cen1vals,binsize=cen_bsize,
                                      color='blue',
                                      show=False)
        hplt_cen = eu.plotting.bhist(cen2vals,binsize=cen_bsize,
                                     color='red',
                                     show=False, plt=hplt_cen0)
        hplt_cen.add(key)

        bsize1=g1vals.std()*0.2 #errs[0]*0.2
        bsize2=g2vals.std()*0.2 # errs[1]*0.2
        hplt_g1 = eu.plotting.bhist(g1vals,binsize=bsize1,
                                  show=False)
        hplt_g2 = eu.plotting.bhist(g2vals,binsize=bsize2,
                                  show=False)

        Tsdev = Tvals.std()
        Tbsize=Tsdev*0.2
        #hplt_T = eu.plotting.bhist(Tvals,binsize=Tbsize,
        #                          show=False)

        logTvals=log10(Tvals)
        Tsdev = logTvals.std()
        Tbsize=Tsdev*0.2
        hplt_T = eu.plotting.bhist(logTvals,binsize=Tbsize,
                                   show=False)



        amp_sdev = ampvals.std()
        amp_bsize=amp_sdev*0.2
        hplt_amp = eu.plotting.bhist(ampvals,binsize=amp_bsize,
                                     show=False)



        hplt_cen.xlabel='center'
        hplt_g1.xlabel=g1lab
        hplt_g2.xlabel=g2lab
        hplt_T.xlabel=r'$log_{10}T$'
        hplt_amp.xlabel='Amplitude'

        tab[0,0] = burn_cen
        tab[1,0] = burn_g1
        tab[2,0] = burn_g2
        tab[3,0] = burn_T
        tab[4,0] = burn_amp

        tab[0,1] = hplt_cen
        tab[1,1] = hplt_g1
        tab[2,1] = hplt_g2
        tab[3,1] = hplt_T
        tab[4,1] = hplt_amp
        tab[5,0] = likep
        tab.show()

        if False: 
            import images
            nx = ny = 40
            levels=8
            h2d = eu.stat.histogram2d(Tvals, g1vals, nx=nx, ny=ny,more=True)
            images.view(h2d['hist'], type='cont',
                        xdr=[h2d['xcenter'][0], h2d['xcenter'][-1]],
                        ydr=[h2d['ycenter'][0], h2d['ycenter'][-1]],
                        xlabel='T', ylabel='g1', levels=levels)
            h2d = eu.stat.histogram2d(Tvals, g2vals, nx=nx, ny=ny,more=True)
            images.view(h2d['hist'], type='cont',
                        xdr=[h2d['xcenter'][0], h2d['xcenter'][-1]],
                        ydr=[h2d['ycenter'][0], h2d['ycenter'][-1]],
                        xlabel='T', ylabel='g2', levels=levels)

        key=raw_input('hit a key (q to quit): ')
        if key=='q':
            stop


class CenPrior:
    def __init__(self, cen, sigma):
        self.cen=cen
        self.sigma=sigma
        self.sigma2=[s**2 for s in sigma]

    def lnprob(self, pos):
        lnprob0 = -0.5*(self.cen[0]-pos[0])**2/self.sigma2[0]
        lnprob1 = -0.5*(self.cen[1]-pos[1])**2/self.sigma2[1]
        return lnprob0 + lnprob1


def g1g2_to_e1e2(g1, g2):
    """
    This version without exceptions

    returns e1,e2,okflag
    """
    g = math.sqrt(g1**2 + g2**2)
    if g >= 1.:
        return LOWVAL,LOWVAL,False

    if g == 0:
        return 0.,0.,True
    e = math.tanh(2*math.atanh(g))
    if e >= 1.:
        return LOWVAL,LOWVAL,False

    fac = e/g
    e1, e2 = fac*g1, fac*g2
    return e1,e2,True


