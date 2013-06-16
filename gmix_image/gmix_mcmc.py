"""
admom

for now
    if admom_s2n > thresh:
        LMFitterSimple
    else:
        MixMC

"""
from sys import stderr
import math
import numpy
from numpy import sqrt, log, log10, zeros, \
        where, array, diag, median
from numpy.linalg import eig

from . import util
from .util import print_pars, randomize_e1e2, \
        calculate_some_stats, srandu
from .priors import CenPrior
from .gmix import GMix

from . import render
from .render import gmix2image


LOWVAL=-9999.9e9


def _get_as_list(arg):
    if arg is None:
        return None

    if isinstance(arg,list):
        return arg
    else:
        return [arg]


def _check_lists(*args):
    llen=len(args[0])
    for l in args:
        if l is None:
            continue
        if len(l) != llen:
            raise ValueError("all lists must be same length")


class MixMCSimple:
    def __init__(self, image, weight, psf, gprior, T_guess, counts_guess, cen_guess, model, **keys):
        """
        mcmc sampling of posterior, simple model.

        Two modes of operation - send a center guess and admom will
        be run internally, or send ares=, with wrow,wcol,Irr,Irc,Icc

        parameters
        ----------
        image:
            sky subtracted image as a numpy array, or list of such
        weight:
            1/(Error per pixel)**2, or an image of such or even
            a list of those corresponding to the image input
        psf:
            The psf gaussian mixture as a GMix object, or list of such
        cen:
            The center guess.  Ignored if ares= is sent.
        gprior:
            The prior on the g1,g2 surface.
        model:
            Type of model, gexp, gdev, gauss

        jacob: optional
            A dictionary holding the jacobian, or list of such

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
        cen_width: bool
            Use this as a width on the prior,
            with the center set the adaptive moments solution.
            Default is broad, 1.0
        """
        
        self.keys=keys

        self.make_plots=keys.get('make_plots',False)
        self.do_pqr=keys.get('do_pqr',True)
        self.when_prior = keys.get('when_prior',"during")

        # cen1,cen2,e1,e2,T,p
        self.npars=6

        self.im_list=_get_as_list(image)
        self.wt_list=_get_as_list(weight)
        self.psf_list=_get_as_list(psf)
        self._set_jacob_list(**keys)
        _check_lists(self.im_list, self.wt_list, self.psf_list,self.jacob_list)

        self.nimage=len(self.im_list)
        self.imsize=self.im_list[0].size
        self.totpix=self.nimage*self.imsize

        self.model=model

        self.gprior=gprior
        self.T_guess=T_guess
        self.counts_guess=counts_guess

        self.cen_guess=cen_guess

        self.cen_prior=keys.get('cen_prior',None)
        if self.cen_prior is None:
            self.cen_width=keys.get('cen_width',1.0)
            self.cen_prior=CenPrior(self.cen_guess, [self.cen_width]*2)

        self.Tprior=keys.get('Tprior',None)
        self.counts_prior=keys.get('counts_prior',None)

        self.nwalkers=keys.get('nwalkers',20)
        self.nstep=keys.get('nstep',200)
        self.burnin=keys.get('burnin',400)
        self.draw_gprior=keys.get('draw_gprior',True)
        self.mca_a=keys.get('mca_a',2.0)
        self.doiter=keys.get('iter',True)

        self._set_im_sums()

        self._go()

    def get_result(self):
        return self._result

    def get_gmix(self):
        pars=self._result['pars']
        try:
            gmix=GMix(pars, type=self.model)
        except ValueError:
            gmix=None
        return gmix

    def _go(self):
        self.sampler=self._do_trials()

        self.trials  = self.sampler.flatchain

        lnprobs = self.sampler.lnprobability.reshape(self.nwalkers*self.nstep)
        self.lnprobs = lnprobs - lnprobs.max()

        # get the expectation values, sensitivity and errors
        self._calc_result()

        if self.make_plots:
            self._doplots()

    def _get_sampler(self):
        import emcee
        sampler = emcee.EnsembleSampler(self.nwalkers, 
                                        self.npars, 
                                        self._calc_lnprob,
                                        a=self.mca_a)
        return sampler

    def _do_trials(self):

        sampler = self._get_sampler()
        guess=self._get_guess()
        pos, prob, state = sampler.run_mcmc(guess, self.burnin)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, self.nstep)


        if self.doiter:
            tau_max=0.10

            i=0
            while True:
                i+=1
                try:
                    acor=sampler.acor
                    tau = (sampler.acor/self.nstep).max()
                    if tau > tau_max:
                        print >>stderr,"tau",tau,"greater than",tau_max,"iter:",i
                    else:
                        break
                except:
                    print >>stderr,"err in acor, retrying"

                sampler.reset()
                pos, prob, state = sampler.run_mcmc(pos, self.nstep)

            self._tau=tau
        return sampler

    def _calc_lnprob(self, pars):

        g=sqrt(pars[2]**2 + pars[3]**2)
        if g >= 1:
            return LOWVAL

        gmix_list=self._get_gmix_list(pars)
        if gmix_list is None:
            return LOWVAL

        logprob = self._get_loglike_c(gmix_list)

        logprob += self._get_priors(pars)

        return logprob

 
    def _get_loglike_c(self, gmix_list):
        loglike = 0.0

        for i in xrange(len(self.im_list)):
            im=self.im_list[i]
            wt=self.wt_list[i]
            jacob=self.jacob_list[i]
            gmix=gmix_list[i]

            res=self._dolike_one(im,wt,jacob,gmix)

            loglike += res[0]
            flags = res[3]

            if flags != 0:
                return LOWVAL


        return loglike

    def _get_priors(self, pars):
        logprob=0.0

        if self.when_prior=="during":
            gp = self._get_lngprior(pars[2], pars[3])
            logprob += gp

        cp = self.cen_prior.lnprob(pars[0:2])
        logprob += cp

        if self.Tprior is not None:
            try:
                Tp = self.Tprior.lnprob(pars[4])
                logprob += Tp
            except ValueError:
                return LOWVAL

        if self.counts_prior is not None:
            try:
                cp = self.counts_prior.lnprob(pars[5])
                logprob += cp
            except ValueError:
                return LOWVAL

        return logprob

    def _get_lngprior(self, g1, g2):
        g=sqrt(g1**2 + g2**2)
        gp = self.gprior.prior2d_gabs_scalar(g)
        if gp > 0:
            gp = log(gp)
        else:
            gp=LOWVAL
        return gp




    def _dolike_one(self, im, wt, jacob, gmix):
        if not isinstance(wt,numpy.ndarray):
            if jacob is None:
                res=render._render.loglike(im,
                                           gmix,
                                           wt)
            else:
                res=render._render.loglike_jacob(im,
                                                 wt,
                                                 jacob['dudrow'],
                                                 jacob['dudcol'],
                                                 jacob['dvdrow'],
                                                 jacob['dvdcol'],
                                                 jacob['row0'],
                                                 jacob['col0'],
                                                 gmix)
        else:
            if jacob is None:
                raise RuntimeError("implement loglike_wt without jacob")
            else:
                res=render._render.loglike_wt_jacob(im,
                                                    wt,
                                                    jacob['dudrow'],
                                                    jacob['dudcol'],
                                                    jacob['dvdrow'],
                                                    jacob['dvdcol'],
                                                    jacob['row0'],
                                                    jacob['col0'],
                                                    gmix)
        return res


    def _get_gmix_list(self, pars):
        """
        Get a list of gmix objects, each convolved with
        the psf in the individual images
        """
        gmix_list=[]
        for psf in self.psf_list:
            gmix=self._get_convolved_gmix(pars, psf)
            if gmix is None:
                return None

            gmix_list.append(gmix)

        return gmix_list 

    def _get_convolved_gmix(self, pars, psf):
        try:
            gmix0=GMix(pars, type=self.model)
            gmix=gmix0.convolve(psf)
        except ValueError:
            gmix=None
        return gmix




    def _calc_result(self):
        """
        We marginalize over all parameters but g1,g2, which
        are index 0 and 1 in the pars array
        """

        pars,pcov,g,gcov,gsens=self._get_trial_stats()
 
        arates = self.sampler.acceptance_fraction
        arate = arates.mean()

        gmix_list=self._get_gmix_list(pars)
        if gmix_list is None:
            stats={}
        else:
            stats=self._get_fit_stats(gmix_list)

        Tmean,Terr=self._get_T_stats(pars,pcov)
        Ts2n=Tmean/Terr

        flux,flux_err=self._get_flux_stats(pars,pcov)
        Fs2n=flux/flux_err

        self._result={'flags':0,
                      'model':self.model,
                      'g':g,
                      'gcov':gcov,
                      'gsens':gsens,
                      'pars':pars,
                      'perr':sqrt(diag(pcov)),
                      'pcov':pcov,
                      'Tmean':Tmean,
                      'Terr':Terr,
                      'Ts2n':Ts2n,
                      'flux':flux,
                      'flux_err':flux_err,
                      'flux_s2n':Fs2n,
                      'arate':arate}

        if self.do_pqr:
            P,Q,R = self._get_PQR()
            self._result['P']=P
            self._result['Q']=Q
            self._result['R']=R

        self._result.update(stats)

    def _get_T_stats(self, pars, pcov):
        """
        Simple model
        """
        return pars[4], sqrt(pcov[4,4])

    def _get_flux_stats(self, pars, pcov):
        """
        Simple model
        """
        return pars[5], sqrt(pcov[5,5])

    def _get_fit_stats(self, gmix_list):
        from math import log, sqrt
        from . import render
        import scipy.stats

        npars=self.npars

        s2n_numer=0.
        s2n_denom=0.
        loglike=0.

        for i in xrange(self.nimage):
            im=self.im_list[i]
            wt=self.wt_list[i]
            jacob=self.jacob_list[i]
            gmix=gmix_list[i]
 
            res=self._dolike_one(im,wt,jacob,gmix)

            loglike += res[0]
            s2n_numer += res[1]
            s2n_denom += res[2]
            
        if s2n_denom > 0:
            s2n=s2n_numer/sqrt(s2n_denom)
        else:
            s2n=0.0

        dof=self._get_dof()
        eff_npix=self._get_effective_npix()

        chi2=loglike/(-0.5)
        chi2per = chi2/dof

        prob = scipy.stats.chisqprob(chi2, dof)

        aic = -2*loglike + 2*npars
        bic = -2*loglike + npars*log(eff_npix)

        return {'s2n_w':s2n,
                'loglike':loglike,
                'chi2per':chi2per,
                'dof':dof,
                'fit_prob':prob,
                'aic':aic,
                'bic':bic}

    def _get_effective_npix(self):
        """
        Because of the weight map, each pixel gets a different weight in the
        chi^2.  This changes the effective degrees of freedom.  The extreme
        case is when the weight is zero; these pixels are essentially not used.

        We replace the number of pixels with

            eff_npix = sum(weights)maxweight
        """
        if isinstance( self.wt_list[0], numpy.ndarray):
            if not hasattr(self, 'eff_npix'):
                wtmax = 0.0
                wtsum = 0.0
                for wt in self.wt_list:
                    this_wtmax = wt.max()
                    if this_wtmax > wtmax:
                        wtmax = this_wtmax

                    wtsum += wt.sum()

                self.eff_npix=wtsum/wtmax
        else:
            self.eff_npix=self.totpix

        if self.eff_npix <= 0:
            self.eff_npix=1.e-6

        return self.eff_npix

    def _get_dof(self):
        """
        Effective def based on effective number of pixels
        """
        eff_npix=self._get_effective_npix()
        dof = eff_npix-self.npars
        if dof <= 0:
            dof = 1.e-6
        return dof


    def _get_trial_stats(self):
        import mcmc

        g1vals=self.trials[:,2]
        g2vals=self.trials[:,3]

        prior = self.gprior(g1vals,g2vals)
        dpri_by_g1 = self.gprior.dbyg1(g1vals,g2vals)
        dpri_by_g2 = self.gprior.dbyg2(g1vals,g2vals)

        psum = prior.sum()

        if self.when_prior=="during":
            # prior is already in the distribution of
            # points.  This is simpler for most things but
            # for sensitivity we need a factor of (1/P)dP/de
            pars,pcov = mcmc.extract_stats(self.trials)
        else:
            pars,pcov = mcmc.extract_stats(self.trials,weights=prior)

        g = pars[2:4].copy()
        gcov = pcov[2:4, 2:4].copy()

        g1diff = g[0]-g1vals
        g2diff = g[1]-g2vals

        gsens = zeros(2)
        if self.when_prior=="during":
            w,=where(prior > 0)
            if w.size == 0:
                raise ValueError("no prior values > 0!")

            gsens[0]= 1.-(g1diff[w]*dpri_by_g1[w]/prior[w]).mean()
            gsens[1]= 1.-(g2diff[w]*dpri_by_g2[w]/prior[w]).mean()
        else:
            gsens[0]= 1.-(g1diff*dpri_by_g1).sum()/psum
            gsens[1]= 1.-(g2diff*dpri_by_g2).sum()/psum

        return pars, pcov, g, gcov, gsens

    def _get_PQR(self):
        """
        get the marginalized P,Q,R from Bernstein & Armstrong

        Note if the prior is already in our mcmc chain, so we need to divide by
        the prior everywhere.  Because P*J=P at shear==0 this means P is always
        1

        """
        import lensing

        g1=self.trials[:,2]
        g2=self.trials[:,3]

        P,Q,R = self.gprior.get_pqr(g1,g2)

        if self.when_prior=="during":
            P,Q,R = self._fix_pqr_for_during(g1,g2,P,Q,R)

        P = P.mean()
        Q = Q.mean(axis=0)
        R = R.mean(axis=0)

        return P,Q,R

    def _fix_pqr_for_during(self, g1, g2, P, Q, R):
        prior = self.gprior(g1,g2)
        w,=numpy.where(prior > 0)
        if w.size == 0:
            raise ValueError("no prior values > 0!")

        P = P[w]
        Q = Q[w,:]
        R = R[w,:,:]

        pinv = 1/prior[w]
        P *= pinv[w]
        Q[:,0] *= pinv[w]
        Q[:,1] *= pinv[w]

        R[:,0,0] *= pinv[w]
        R[:,0,1] *= pinv[w]
        R[:,1,0] *= pinv[w]
        R[:,1,1] *= pinv[w]

        return P, Q, R

    def get_maxprob_pars(self):
        wmax=self.lnprobs.argmax()
        max_pars = self.trials[wmax,:].copy()
        return max_pars

    def get_maxprob_model(self):
        max_pars=self.get_maxprob_pars()
        gmix=self._get_convolved_gmix(max_pars)
        model=gmix2image(gmix,self.image.shape)
        return model

    def _set_jacob_list(self, **keys):
        jlist = keys.get("jacob",None)
        if jlist is None:
            jlist=[None]*len(self.im_list)
        else:
            if not isinstance(jlist,list):
                jlist=[jlist]

        self.jacob_list=jlist        

    def _set_im_sums(self):
        """
        median of the counts across all input images
        """
        clist=numpy.zeros(len(self.im_list))
        for i,im in enumerate(self.im_list):
            clist[i] = im.sum()
        
        self.counts_median=numpy.median(clist)

    def _get_guess(self):
        """
        Note for model coellip this only does one gaussian
        """

        counts_guess=self.counts_guess
        if counts_guess  is None:
            counts_guess=self.counts_median


        guess=zeros( (self.nwalkers,self.npars) )

        guess[:,0]=self.cen_prior.cen[0] + 0.01*srandu(self.nwalkers)
        guess[:,1]=self.cen_prior.cen[1] + 0.01*srandu(self.nwalkers)

        if self.draw_gprior:
            g1rand,g2rand=self.gprior.sample2d(self.nwalkers)
            guess[:,2] = g1rand
            guess[:,3] = g2rand
        else:
            gtot = 0.9*numpy.random.random(self.nwalkers)
            theta=numpy.random.random()*numpy.pi
            g1rand = gtot*numpy.cos(2*theta)
            g2rand = gtot*numpy.sin(2*theta)
            guess[:,2]=g1rand
            guess[:,3]=g2rand
            # (0,0) with some scatter
            #guess[:,2]=0.1*srandu(self.nwalkers)
            #guess[:,3]=0.1*srandu(self.nwalkers)

        guess[:,4] = self.T_guess*(1 + 0.1*srandu(self.nwalkers))
        guess[:,5] = counts_guess*(1 + 0.1*srandu(self.nwalkers))

        self._guess=guess
        return guess


    def _doplots(self):
        import biggles
        import esutil as eu

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

        flux=res['pars'][5]
        flux_err=sqrt(res['pcov'][5,5])
        print 'flux: %g +/- %g' % (flux,flux_err)
        print 'T:  %g +/- %g' % (Tvals.mean(), Tvals.std())
        print 's2n weighted:',res['s2n_w']
        print 'acceptance rate:',res['arate'],'mca_a',self.mca_a

        print_pars(self._result['pars'])
        print_pars(sqrt(diag(self._result['pcov'])))
        #print 'g1sens:',self._result['gsens'][0]
        #print 'g2sens:',self._result['gsens'][1]
        #print 'g1: %.16g +/- %.16g' % (g[0],errs[0])
        #print 'g2: %.16g +/- %.16g' % (g[1],errs[1])
        print 'chi^2/dof: %.3f/%i = %f' % (res['chi2per']*res['dof'],res['dof'],res['chi2per'])
        print 'probrand:',res['fit_prob']

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

        w,=numpy.where(Tvals > 0)
        if w.size > 0:
            logTvals=log10(Tvals[w])
            Tsdev = logTvals.std()
            Tbsize=Tsdev*0.2
            hplt_T = eu.plotting.bhist(logTvals,binsize=Tbsize,
                                       show=False)

            hplt_T.xlabel=r'$log_{10}T$'
        else:
            hplt_T = None

        amp_sdev = ampvals.std()
        amp_bsize=amp_sdev*0.2
        hplt_amp = eu.plotting.bhist(ampvals,binsize=amp_bsize,
                                     show=False)



        hplt_cen.xlabel='center'
        hplt_g1.xlabel=g1lab
        hplt_g2.xlabel=g2lab
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

        self.tab=tab

        prompt=self.keys.get('prompt',True)
        if prompt:
            tab.show()
            key=raw_input('hit a key (q to quit): ')
            if key=='q':
                stop
            print

class MixMCMatch(MixMCSimple):
    def __init__(self, image, weight, psf, gmix0, **keys):
        """
        mcmc sampling of posterior, simple model, fixing structural
        parameters.

        parameters
        ----------
        image:
            sky subtracted image as a numpy array, or list of such
        weight:
            1/(Error per pixel)**2, or an image of such or even
            a list of those corresponding to the image input
        psf:
            The psf gaussian mixture as a GMix object, or list of such
        gmix0:
            GMix for shape

        jacob: optional
            A dictionary holding the jacobian, or list of such

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
        """
        
        self.keys=keys
        self.make_plots=keys.get('make_plots',False)

        self.npars=1

        self.im_list=_get_as_list(image)
        self.wt_list=_get_as_list(weight)
        self.psf_list=_get_as_list(psf)
        self._set_jacob_list(**keys)
        _check_lists(self.im_list, self.wt_list, self.psf_list,self.jacob_list)

        self.gmix0=gmix0

        self.nimage=len(self.im_list)
        self.imsize=self.im_list[0].size
        self.totpix=self.nimage*self.imsize

        self.model='amponly'

        self.start_counts=gmix0.get_psum()

        self.nwalkers=keys.get('nwalkers',10)
        self.nstep=keys.get('nstep',100)
        self.burnin=keys.get('burnin',100)
        self.mca_a=keys.get('mca_a',2.0)
        self.doiter=keys.get('iter',True)

        #self._set_im_sums()

        self._set_gmix_list()

        self._go()

    def _calc_lnprob(self, pars):
        gmix_list=self._get_gmix_list(pars)
        if gmix_list is None:
            return LOWVAL

        # this is in the base class
        logprob = self._get_loglike_c(gmix_list)
        return logprob

    def _set_gmix_list(self):
        gmix_list=[]
        for psf in self.psf_list:
            g = self.gmix0.convolve(psf)
            gmix_list.append(g)
        self.gmix_list=gmix_list

    def _set_psum(self, pars1):
        psum=pars1[0]
        for g in self.gmix_list:
            g.set_psum(psum)

    def _get_gmix_list(self, pars):
        """
        called by the generic code 
        """
        self._set_psum(pars)
        return self.gmix_list

    def _calc_result(self):
        import mcmc

        pars,pcov = mcmc.extract_stats(self.trials)
 
        arates = self.sampler.acceptance_fraction
        arate = arates.mean()

        gmix_list=self._get_gmix_list(pars)
        stats=self._get_fit_stats(gmix_list)

        flux = pars[0]
        if pcov[0,0] > 0:
            perr = sqrt(diag(pcov))
            flux_err = sqrt(pcov[0,0])
        else:
            flux_err = abs(LOWVAL)
            perr = numpy.array([flux_err])

        Fs2n=flux/flux_err

        self._result={'flags':0,
                      'model':self.model,
                      'pars':pars,
                      'perr':sqrt(diag(pcov)),
                      'pcov':pcov,
                      'flux':flux,
                      'flux_err':flux_err,
                      'Fs2n':Fs2n,
                      'arate':arate}

        self._result.update(stats)


    def _get_guess(self):
        """
        Note for model coellip this only does one gaussian
        """

        guess=zeros( (self.nwalkers,self.npars) )

        if self.start_counts == 0:
            guess[:,0]=self.start_counts + srandu(self.nwalkers)
        else:
            guess[:,0]=self.start_counts*(1.0 + 0.1*srandu(self.nwalkers))

        self._guess=guess
        return guess


    def _doplots(self):
        import mcmc
        import biggles
        import esutil as eu

        tab=biggles.Table(2,self.npars)

        flux = self.trials[:,0]
        fmean=self._result['flux']
        ferr=self._result['flux_err']

        burn_plt=biggles.FramedPlot()
        ind=numpy.arange(self.trials.shape[0])
        burn_plt.add(biggles.Curve(ind, flux))

        burn_plt.ylabel='Amplitude'

        bsize=ferr*0.2
        hplt = eu.plotting.bhist(flux,binsize=bsize, show=False)
        hplt.xlabel='Amplitude'

        tab[0,0] = burn_plt
        tab[1,0] = hplt

        print 'flux: %g +/- %g' % (fmean,ferr)
        print 'arate:',self._result['arate']

        self.tab=tab

        prompt=self.keys.get('prompt',True)
        if prompt:
            tab.show()
            key=raw_input('hit a key (q to quit): ')
            if key=='q':
                stop
            print


class MixMCBD(MixMCSimple):
    """
    bulge+disk, coelliptical
    """
    def __init__(self, image, ivar, psf, gprior, **keys):

        self.keys=keys

        self.make_plots=keys.get('make_plots',False)
        self.do_pqr=keys.get('do_pqr',True)
        self.when_prior = "after"

        # cen1,cen2,e1,e2,Ti,pi
        self.model='bd'
        self.npars=8

        self.image=image
        self.ivar=float(ivar)

        self.psf_gmix=psf

        self.gprior=gprior

        # doesn't currently make sense
        self.Tprior=None

        self.nwalkers=keys.get('nwalkers',20)
        self.nstep=keys.get('nstep',200)
        self.burnin=keys.get('burnin',400)
        self.draw_gprior=keys.get('draw_gprior',True)
        self.mca_a=keys.get('mca_a',2.0)
        self.doiter=keys.get('iter',True)
        
        self.cen_guess=keys.get('cen',None)
        self.ares=keys.get('ares',None)

        self.cen_width=keys.get('cen_width',1.0)

        if self.cen_guess is None and self.ares is None:
            raise ValueError("send cen= or ares=")
        if self.ares is not None and self.ares['whyflag']!=0:
            raise ValueError("If you enter ares it must have "
                             "whyflag==0")

        self.counts=self.image.sum()

        self._go()

    def _get_guess(self):
        if self.ares is None:
            self.ares=self._run_admom(self.image, self.ivar, 
                                      self.cen_guess, 8.0)

        cen=[self.ares['wrow'],self.ares['wcol']]
        self.cen_prior=CenPrior(cen, [self.cen_width]*2)

        T0=self.ares['Irr'] + self.ares['Icc']

        guess=zeros( (self.nwalkers,self.npars) )

        guess[:,0]=self.cen_prior.cen[0] + 0.01*srandu(self.nwalkers)
        guess[:,1]=self.cen_prior.cen[1] + 0.01*srandu(self.nwalkers)

        if self.draw_gprior:
            g1rand,g2rand=self.gprior.sample2d(self.nwalkers)
            guess[:,2] = g1rand
            guess[:,3] = g2rand
        else:
            # (0,0) with some scatter
            guess[:,2]=0.1*srandu(self.nwalkers)
            guess[:,3]=0.1*srandu(self.nwalkers)

        counts0=self.counts

        fac=1
        if self.make_plots:
            print 'T0 guess:',fac*T0
        """
        guess[:,4] = fac*T0*( 1 + 0.1*srandu(self.nwalkers) )
        guess[:,5] = fac*T0*( 1 + 0.1*srandu(self.nwalkers) )

        guess[:,6] = 0.5*counts0*( 1 + 0.1*srandu(self.nwalkers) )
        guess[:,7] = 0.5*counts0*( 1 + 0.1*srandu(self.nwalkers) )
        """

        guess[:,4] = fac*T0*( 1 + 0.4*srandu(self.nwalkers) )
        guess[:,5] = fac*T0*( 1 + 0.4*srandu(self.nwalkers) )

        guess[:,6] = counts0*(0.2+0.6*numpy.random.random(self.nwalkers))
        guess[:,7] = counts0*(0.2+0.6*numpy.random.random(self.nwalkers))

        self._guess=guess
        return guess


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

        if self.when_prior=="during":
            pars,pcov = mcmc.extract_stats(self.trials)
        else:
            pars,pcov = mcmc.extract_stats(self.trials,weights=prior)


        g[:] = pars[2:4]
        gcov[:,:] = pcov[2:4, 2:4]

        g1diff = g[0]-g1vals
        g2diff = g[1]-g2vals

        if self.when_prior=="during":
            w,=where(prior > 0)
            if w.size == 0:
                raise ValueError("no prior values > 0!")

            gsens[0]= 1.-(g1diff[w]*dpri_by_g1[w]/prior[w]).mean()
            gsens[1]= 1.-(g2diff[w]*dpri_by_g2[w]/prior[w]).mean()
        else:
            gsens[0]= 1.-(g1diff*dpri_by_g1).sum()/psum
            gsens[1]= 1.-(g2diff*dpri_by_g2).sum()/psum

 
        arates = self.sampler.acceptance_fraction
        arate = arates.mean()

        max_pars=self.get_maxprob_pars()
        gmix=self._get_convolved_gmix(max_pars)

        stats=calculate_some_stats(self.image, 
                                   self.ivar, 
                                   gmix,
                                   self.npars)

        cdiag = diag(pcov)

        Fvals=pars[[6,7]].copy()
        Fcov = pcov[6:6+2, 6:6+2].copy()

        flux=Fvals.sum()
        flux_err=sqrt( Fcov[0,0] + Fcov[1,1] + 2*Fcov[0,1] )

        Tvals=pars[[4,5]].copy()
        Tcov=pcov[4:4+2, 4:4+2].copy()

        Tmean=(Tvals*Fvals).sum()/flux
        Terr = sqrt(  Fvals[0]**2*Tcov[0,0]
                    + Fvals[1]**2*Tcov[1,1]
                    + 2*Fvals[0]*Fvals[1]*Tcov[0,1] )


        Ts2n=Tmean/Terr
        Fs2n=flux/flux_err

        self._result={'flags':0,
                      'model':self.model,
                      'g':g,
                      'gcov':gcov,
                      'gsens':gsens,
                      'pars':pars,
                      'perr':sqrt(diag(pcov)),
                      'pcov':pcov,
                      'Tmean':Tmean,
                      'Terr':Terr,
                      'Ts2n':Ts2n,
                      'flux':flux,
                      'flux_err':flux_err,
                      'Fs2n':Fs2n,
                      'arate':arate}

        if self.do_pqr:
            P,Q,R = self._get_PQR()
            self._result['P']=P
            self._result['Q']=Q
            self._result['R']=R

        for k in stats:
            self._result[k] = stats[k]

    def _doplots(self):
        import mcmc
        import biggles
        import esutil as eu

        res=self.get_result()

        biggles.configure('screen','width',1100)
        biggles.configure('screen','height',1100)
        biggles.configure("default","fontsize_min",1.2)

        tab=biggles.Table(self.npars,2)

        labels=[r'$cen_1$',r'$cen_2$',
                r'$g_1$',r'$g_2$',
                r'$T_{exp}$',r'$T_{dev}$',
                r'$F_{exp}',r'$F_{dev}$']

        ind=numpy.arange(self.trials.shape[0])
        for i in xrange(self.npars):
            vals = self.trials[:,i]

            burn_plt=biggles.FramedPlot()
            burn_plt.add(biggles.Curve(ind, vals))

            burn_plt.ylabel=labels[i]

            bsize=vals.std()*0.2
            hplt = eu.plotting.bhist(vals,binsize=bsize, show=False)
            hplt.xlabel=labels[i]

            tab[i,0] = burn_plt
            tab[i,1] = hplt

        Tmeans = self.trials[:,4:4+2].mean(axis=0)
        Fmeans = self.trials[:,6:6+2].mean(axis=0)

        Tfracs=Tmeans/Tmeans.sum()
        Ffracs=Fmeans/Fmeans.sum()

        print 'flux: %g +/- %g' % (res['flux'],res['flux_err'])
        print 'T:    %g +/- %g' % (res['Tmean'],res['Terr'])
        Fcov=res['pcov'][6:6+2, 6:6+2]

        print 'flux cov:',Fcov
        print 'Ffracs:',Ffracs
        print 'Tvals:',Tmeans
        print 'Tfracs:',Tfracs
        print 'exp Tmin,max:',self.trials[:,4].min(), self.trials[:,4].max()
        print 'dev Tmin,max:',self.trials[:,5].min(), self.trials[:,5].max()
        print 'arate:',self._result['arate']

        prompt=self.keys.get('prompt',True)
        if prompt:
            tab.show()
            key=raw_input('hit a key (q to quit): ')
            if key=='q':
                stop
            print


class MixMCCoellip(MixMCSimple):
    def __init__(self, image, ivar, psf, gprior, ngauss, **keys):

        self.keys=keys

        self.make_plots=keys.get('make_plots',False)
        self.do_pqr=keys.get('do_pqr',True)
        self.when_prior = "after"

        # cen1,cen2,e1,e2,Ti,pi
        self.model='coellip'
        self.ngauss=ngauss
        self.npars=2*ngauss+4

        self.image=image
        self.ivar=float(ivar)

        self.psf_gmix=psf

        self.gprior=gprior

        # doesn't currently make sense
        self.Tprior=None

        self.nwalkers=keys.get('nwalkers',20)
        self.nstep=keys.get('nstep',200)
        self.burnin=keys.get('burnin',400)
        self.draw_gprior=keys.get('draw_gprior',True)
        self.mca_a=keys.get('mca_a',2.0)
        self.doiter=keys.get('iter',True)
        
        self.cen_guess=keys.get('cen',None)
        self.ares=keys.get('ares',None)

        self.cen_width=keys.get('cen_width',1.0)

        if self.cen_guess is None and self.ares is None:
            raise ValueError("send cen= or ares=")
        if self.ares is not None and self.ares['whyflag']!=0:
            raise ValueError("If you enter ares it must have "
                             "whyflag==0")

        self.counts=self.image.sum()

        self._go()

    def _get_guess(self):
        if self.ares is None:
            self.ares=self._run_admom(self.image, self.ivar, 
                                      self.cen_guess, 8.0)

        cen=[self.ares['wrow'],self.ares['wcol']]
        self.cen_prior=CenPrior(cen, [self.cen_width]*2)

        T0=self.ares['Irr'] + self.ares['Icc']

        guess=zeros( (self.nwalkers,self.npars) )

        guess[:,0]=self.cen_prior.cen[0] + 0.01*srandu(self.nwalkers)
        guess[:,1]=self.cen_prior.cen[1] + 0.01*srandu(self.nwalkers)

        if self.draw_gprior:
            g1rand,g2rand=self.gprior.sample2d(self.nwalkers)
            guess[:,2] = g1rand
            guess[:,3] = g2rand
        else:
            # (0,0) with some scatter
            guess[:,2]=0.1*srandu(self.nwalkers)
            guess[:,3]=0.1*srandu(self.nwalkers)

        ngauss=self.ngauss
        counts0=self.counts
        if ngauss==1:
            guess[:,4] = T0*(1 + 0.05*srandu(self.nwalkers))
            guess[:,5] = counts0*(1 + 0.05*srandu(self.nwalkers))
        else:
            # these are tuned ~ for exp
            if ngauss==2:
                #Texamp=array([12.6,3.8])
                #pexamp=array([0.30, 0.70])
                Texamp=array([0.77, 0.23])
                pexamp=array([0.44, 0.53])

                Tfrac=Texamp/Texamp.sum()
                pfrac=pexamp/pexamp.sum()

            elif ngauss==3:
                #Texamp=array([0.46,5.95,2.52])
                #pexamp=array([0.1,0.7,0.22])
                #Texamp = array([1.0,0.3,0.06])
                #pexamp = array([0.26,0.55,0.18])

                Texamp =array([0.79,  0.2,  0.03])
                pexamp = array([0.44, 0.48, 0.08])

                Tfrac=Texamp/Texamp.sum()
                pfrac=pexamp/pexamp.sum()
            else:
                raise ValueError("support ngauss>3")

            fac=1.5
            if self.make_plots:
                print 'Tguess:',fac*T0*Tfrac
            for i in xrange(ngauss):
                guess[:,4+i] = fac*T0*Tfrac[i]*(1. + 0.1*srandu(self.nwalkers))
                guess[:,4+ngauss+i] = counts0*pfrac[i]*(1. + 0.1*srandu(self.nwalkers))

        self._guess=guess
        return guess


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

        pars,pcov = mcmc.extract_stats(self.trials,weights=prior)

        g[:] = pars[2:4]
        gcov[:,:] = pcov[2:4, 2:4]

        g1diff = g[0]-g1vals
        g2diff = g[1]-g2vals

        gsens[0]= 1.-(g1diff*dpri_by_g1).sum()/psum
        gsens[1]= 1.-(g2diff*dpri_by_g2).sum()/psum
 
        arates = self.sampler.acceptance_fraction
        arate = arates.mean()

        max_pars=self.get_maxprob_pars()
        gmix=self._get_convolved_gmix(max_pars)

        stats=calculate_some_stats(self.image, 
                                   self.ivar, 
                                   gmix,
                                   self.npars)

        cdiag = diag(pcov)
        Fvals=pars[4+self.ngauss:]
        flux=Fvals.sum()
        Fcovs = cdiag[4+self.ngauss:]
        flux_err=sqrt( Fcovs.sum() )

        Tvals=pars[4:4+self.ngauss]
        Tmean=(Tvals*Fvals).sum()/flux

        Tcovs = cdiag[4:4+self.ngauss]
        Fvals2 = Fvals**2
        Terr2 = (Tcovs*Fvals2).sum()/Fvals2.sum()
        Terr = sqrt(Terr2)

        Ts2n=Tmean/Terr
        Fs2n=flux/flux_err

        self._result={'flags':0,
                      'model':self.model,
                      'g':g,
                      'gcov':gcov,
                      'gsens':gsens,
                      'pars':pars,
                      'perr':sqrt(diag(pcov)),
                      'pcov':pcov,
                      'Tmean':Tmean,
                      'Terr':Terr,
                      'Ts2n':Ts2n,
                      'flux':flux,
                      'flux_err':flux_err,
                      'Fs2n':Fs2n,
                      'arate':arate}

        if self.do_pqr:
            P,Q,R = self._get_PQR()
            self._result['P']=P
            self._result['Q']=Q
            self._result['R']=R

        for k in stats:
            self._result[k] = stats[k]

    def _doplots(self):
        import mcmc
        import biggles
        import esutil as eu

        biggles.configure('screen','width',1100)
        biggles.configure('screen','height',1100)
        biggles.configure("default","fontsize_min",1.2)

        ngauss=self.ngauss
        tab=biggles.Table(self.npars,2)

        labels=[r'$cen_1$',r'$cen_2$',r'$g_1$',r'$g_2$']
        Tlabs = [r'$T_{%s}$' % (i+1) for i in xrange(ngauss)]
        Flabs = [r'$F_{%s}$' % (i+1) for i in xrange(ngauss)]

        labels += Tlabs
        labels += Flabs

        ind=numpy.arange(self.trials.shape[0])
        for i in xrange(self.npars):
            vals = self.trials[:,i]

            burn_plt=biggles.FramedPlot()
            burn_plt.add(biggles.Curve(ind, vals))

            burn_plt.ylabel=labels[i]

            bsize=vals.std()*0.2
            hplt = eu.plotting.bhist(vals,binsize=bsize, show=False)
            hplt.xlabel=labels[i]

            tab[i,0] = burn_plt
            tab[i,1] = hplt

        Tmeans = self.trials[:,4:4+self.ngauss].mean(axis=0)
        Fmeans = self.trials[:,4+self.ngauss:].mean(axis=0)

        Tfracs=Tmeans/Tmeans.sum()
        Ffracs=Fmeans/Fmeans.sum()

        print 'arate:',self._result['arate']
        print 'Tvals:',Tmeans
        print 'Tfracs:',Tfracs
        print 'Ffracs:',Ffracs

        prompt=self.keys.get('prompt',True)
        if prompt:
            tab.show()
            key=raw_input('hit a key (q to quit): ')
            if key=='q':
                stop
            print


class MixMCPSF:
    def __init__(self, image, ivar, model, **keys):
        """
        mcmc sampling of posterior.

        Note the ellipticities are in e space not g space

        Two modes of operation - send a center guess and admom will
        be run internally, or send ares=, with wrow,wcol,Irr,Irc,Icc

        parameters
        ----------
        image:
            sky subtracted image as a numpy array
        ivar:
            1/(Error per pixel)**2
        cen:
            The center guess.  Ignored if ares= is sent.
        model:
            Type of model, gexp, gdev, gauss

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
        ares: optional
            The output from a run of admom.  The whyflag
            field must be zero.
        """
        
        self.keys=keys

        self.make_plots=False

        # cen1,cen2,e1,e2,T,p
        self.npars=6

        self.image=image
        self.ivar=float(ivar)
        self.model=model

        self.nwalkers=keys.get('nwalkers',20)
        self.nstep=keys.get('nstep',200)
        self.burnin=keys.get('burnin',400)
        self.mca_a=keys.get('mca_a',2.0)
        self.doiter=keys.get('iter',True)
        
        self.cen_guess=keys.get('cen',None)
        self.ares=keys.get('ares',None)

        if self.cen_guess is None and self.ares is None:
            raise ValueError("send cen= or ares=")
        if self.ares is not None and self.ares['whyflag']!=0:
            raise ValueError("If you enter ares it must have "
                             "whyflag==0")

        self.counts=self.image.sum()


        self._go()

    def get_result(self):
        return self._result

    def get_gmix(self):
        # these are not maxlike pars, but expectation value pars
        return self._get_gmix(self._result['pars'])

    def _get_gmix(self, pars):
        if self.model=='gauss':
            type='coellip'
        else:
            type=self.model

        try:
            gmix=GMix(pars, type=type)
        except ValueError:
            gmix=None
        return gmix


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
        if self.doiter:
            while True:
                try:
                    acor=sampler.acor
                    tau = (sampler.acor/self.nstep).max()
                    if tau > 0.1:
                        print "tau",tau,"greater than 0.1"
                    else:
                        break
                except:
                    # something went wrong with acor, run some more
                    pass
                pos, prob, state = sampler.run_mcmc(pos, self.nstep)


    def _calc_lnprob(self, pars):
        g=sqrt(pars[2]**2 + pars[3]**2)
        if g >= 1:
            return LOWVAL

        gmix=self._get_gmix(pars)
        if gmix is None:
            return LOWVAL

        logprob = self._get_loglike_c(gmix)

        cp = self.cen_prior.lnprob(pars[0:2])
        logprob += cp

        return logprob

 
    def _get_loglike_c(self, gmix):

        loglike,s2n_numer,s2n_denom,flags=\
            render._render.loglike(self.image, gmix, self.ivar)

        if flags != 0:
            return LOWVAL

        return loglike


    def _calc_result(self):
        """
        We marginalize over all parameters but g1,g2, which
        are index 0 and 1 in the pars array
        """
        import mcmc

        pars,pcov = mcmc.extract_stats(self.trials)
 
        arates = self.sampler.acceptance_fraction
        arate = arates.mean()

        wmax=self.lnprobs.argmax()
        max_pars = self.trials[wmax,:].copy()
        gmix=self._get_gmix(max_pars)

        if gmix is None:
            stats={}
        else:
            stats=calculate_some_stats(self.image, 
                                       self.ivar, 
                                       gmix,
                                       self.npars)

        Tmean=pars[4]
        Terr=sqrt(pcov[4,4])
        Ts2n=pars[4]/sqrt(pcov[4,4])

        self._result={'flags':0,
                      'model':self.model,
                      'pars':pars,
                      'perr':sqrt(diag(pcov)),
                      'pcov':pcov,
                      'Tmean':Tmean,
                      'Terr':Terr,
                      'Ts2n':Ts2n,
                      'arate':arate}

        for k in stats:
            self._result[k] = stats[k]

    def _run_admom(self, image, ivar, cen, Tguess):
        import admom

        ntry=10
        for i in xrange(ntry):
            ares = admom.admom(image,
                               cen[0],
                               cen[1],
                               sigsky=sqrt(1/ivar),
                               guess=Tguess/2,
                               nsub=1)
            if ares['whyflag']==0:
                break
        if i==(ntry-1):
            raise ValueError("admom failed %s times" % ntry)

        return ares


    def _get_guess(self):

        if self.ares is None:
            self.ares=self._run_admom(self.image, self.ivar, 
                                      self.cen_guess, 8.0)

        
        cen=[self.ares['wrow'],self.ares['wcol']]
        self.cen_prior=CenPrior(cen, [1.]*2)

        Tadmom=self.ares['Irr'] + self.ares['Icc']

        guess=zeros( (self.nwalkers,self.npars) )

        guess[:,0]=self.cen_prior.cen[0] + 0.01*srandu(self.nwalkers)
        guess[:,1]=self.cen_prior.cen[1] + 0.01*srandu(self.nwalkers)

        # (0,0) with some scatter
        guess[:,2]=0.1*srandu(self.nwalkers)
        guess[:,3]=0.1*srandu(self.nwalkers)

        guess[:,4] = Tadmom*(1 + 0.1*srandu(self.nwalkers))
        guess[:,5] = self.counts*(1 + 0.1*srandu(self.nwalkers))

        return guess


    def _doplots(self):
        import mcmc
        import biggles
        import esutil as eu

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
        burn_g1.ylabel=r'$e_1$'

        burn_g2=biggles.FramedPlot()
        burn_g2.add(biggles.Curve(ind, g2vals))
        burn_g2.ylabel=r'$e_2$'

        burn_T=biggles.FramedPlot()
        burn_T.add(biggles.Curve(ind, Tvals))
        burn_T.ylabel='T'

        burn_amp=biggles.FramedPlot()
        burn_amp.add(biggles.Curve(ind, ampvals))
        burn_amp.ylabel='Amplitide'



        likep = biggles.FramedPlot()
        likep.add( biggles.Curve(ind, self.lnprobs) )
        likep.ylabel='ln( prob )'


        res=self.get_result()
        print 's2n weighted:',res['s2n_w']
        print 'acceptance rate:',res['arate']
        print 'T:  %.16g +/- %.16g' % (Tvals.mean(), Tvals.std())

        print_pars(self._result['pars'])
        print_pars(sqrt(diag(self._result['pcov'])))
        print 'chi^2/dof: %.3f/%i = %f' % (res['chi2per']*res['dof'],res['dof'],res['chi2per'])
        print 'probrand:',res['fit_prob']

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

        prompt=self.keys.get('prompt',True)
        if prompt:
            tab.show()
            key=raw_input('hit a key (q to quit): ')
            if key=='q':
                stop
            print



