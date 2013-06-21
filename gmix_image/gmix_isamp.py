from sys import stderr
import numpy
from numpy import sqrt, diag
from . import gmix_fit
from .gmix_fit import GMIXFIT_HUGE_ERRORS
from .gmix_mcmc import MixMCSimple, _get_as_list, _check_lists
from .util import print_pars

class GMixIsampSimple(MixMCSimple):
    def __init__(self, image, weight, psf, gprior, T_guess, counts_guess, cen_guess, model, **keys):
        self.keys=keys

        self.make_plots=keys.get('make_plots',False)
        self.do_pqr=keys.get('do_pqr',True)

        # cen1,cen2,e1,e2,T,p
        self.npars=6
        self.when_prior="during"

        self.im_list=_get_as_list(image)
        self.nimage=len(self.im_list)

        #self.wt_list=_get_as_list(weight)
        self._set_wt_list(weight)
        self.psf_list=_get_as_list(psf)
        self._set_jacob_list(**keys)
        _check_lists(self.im_list, self.wt_list, self.psf_list,self.jacob_list)

        self.imsize=self.im_list[0].size
        self.totpix=self.nimage*self.imsize

        self.model=model

        self.gprior=gprior
        self.T_guess=T_guess
        self.counts_guess=counts_guess
        self.cen_guess=cen_guess
        self.g_guess=keys.get('g_guess',None)

        self.cen_prior=keys.get('cen_prior',None)
        if self.cen_prior is None:
            self.cen_width=keys.get('cen_width',1.0)
            self.cen_prior=CenPrior(self.cen_guess, [self.cen_width]*2)

        self.T_prior=keys.get('T_prior',None)
        self.counts_prior=keys.get('counts_prior',None)

        self.nsample=keys.get('nsample',500)

        self._set_im_sums()

        self._go()

    def _go(self):
        """
        Do a max likelihood fit and then use importance sampling
        to sample the posterior
        """

        # self._lm_result is created
        gm=self._do_maxlike_fit()

        defres = {'flags':self._lm_result['flags'],
                  'lm_result':self._lm_result}

        if self._lm_result['flags'] != 0:
            self._result=defres
        else:
            # self.trials and self.lnprobs are created
            #print_pars(self._lm_result['pars'],front='lm pars:')
            #print_pars(self._lm_result['perr'],front='lm perr:')
            if not self._do_isample():
                defres['flags'] = GMIXFIT_HUGE_ERRORS 
                self._result=defres
                return

            self._calc_result()

        if self.make_plots:
            self._doplots()

    def _do_maxlike_fit(self):
        guess=self._get_lm_guess()
        gm=gmix_fit.GMixFitMultiSimple(self.im_list,
                                       self.wt_list,
                                       self.jacob_list,
                                       self.psf_list,
                                       self.model,

                                       g_guess=self.g_guess,
                                       T_guess=self.T_guess,
                                       counts_guess=self.counts_guess,
                                       cen_guess=self.cen_guess,

                                       gprior=self.gprior,
                                       cen_prior=self.cen_prior,
                                       T_prior=self.T_prior,
                                       counts_prior=self.counts_prior)


        self._lm_result=gm.get_result()

    def _get_lm_guess(self):
        guess=numpy.zeros(self.npars)

        T0 = self.T_guess
        counts0 = self.counts_guess

        if self.g_guess is None:
            gtot = 0.9*numpy.random.random()
            theta=numpy.random.random()*numpy.pi
            g1rand = gtot*numpy.cos(2*theta)
            g2rand = gtot*numpy.sin(2*theta)
            guess[2]=g1rand
            guess[3]=g2rand
        else:
            guess[2:2+2] = self.g_guess

        guess[4] = T0
        guess[5] = counts0

        return guess


    def _do_isample(self):
        """
        Sample from approximate posterior based on likelihood
        and errors.  This gets the trials and probs_approx

        Then evaluate the actual probs to get prob, and weight
        the trials by prob/probs_approx in our statistics

        """
        from math import exp

        # first draw from a distribution representing the
        # max like point plus errors
        trials, probs_approx = self._sample_dist()
        if trials is None:
            # happens when get a crazy fit
            return False

        # now evaluate all these to get the true prob
        iweights = numpy.zeros(self.nsample)
        #probs = numpy.zeros(self.nsample)

        lnprobs = numpy.zeros(self.nsample)
        lnprob_max=None
        for i in xrange(self.nsample):
            #if probs_approx[i]==0:
            #    print 'WARNING: found zero'
            pars = trials[i,:]
            lnprobs[i]=self._calc_lnprob(pars)

        lnprobs = lnprobs - lnprobs.max()
        #print lnprobs

        probs = numpy.exp(lnprobs)

        iweights = probs/probs_approx

        self.trials=trials
        self.probs=probs
        self.probs_approx=probs_approx
        self.iweights=iweights
        self.iweights_sum=iweights.sum()
        self.iweights_sum_inv= 1./self.iweights_sum

        return True

    def _sample_dist(self):
        """
        Represent the n-dimensional distribution as a multi-variate Cauchy
        distribution.  Draw from it and evaluate the probability.  Also
        multiply by our priors.
        """

        return self._sample_dist_gaussian()

    def _sample_dist_gaussian(self):
        from esutil.stat import cholesky_sample

        defres=None,None
        mess="    crazy covariance, cannot do isample"

        pars=self._lm_result['pars']
        cov=self._lm_result['pcov']

        # twice as wide
        cov_sample=cov*4

        if cov_sample[2,2] > 1 or cov_sample[3,3] > 1:
            print >>stderr,mess
            return defres

        try:
            trials = cholesky_sample(cov_sample, self.nsample,
                                     means=pars)
        except numpy.linalg.linalg.LinAlgError:
            print >>stderr,mess
            return defres

        pvals = self._eval_par_gauss(trials, pars, cov_sample)

        return trials, pvals

    def _eval_par_gauss(self, locations, pars, cov):
        from numpy import dot

        npars=pars.size

        pars2d = numpy.atleast_2d(pars)
        cinv = numpy.linalg.inv(cov)

        ones=numpy.ones( (npars,1) )

        ldiff = locations-pars2d
        arg   = -0.5 * dot(dot(ldiff, cinv) * ldiff, ones)[:, 0]
        return numpy.exp(arg)


    def _calc_result(self):
        """
        We marginalize over all parameters but g1,g2, which
        are index 0 and 1 in the pars array
        """

        pars,pcov,g,gcov,gsens=self._get_trial_stats()
 
        #print_pars(pars,front="is pars: ")
        #print_pars(sqrt(diag(pcov)),front="is perr: ")

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
                      'arate':1.0}

        if self.do_pqr:
            P,Q,R = self._get_PQR()
            self._result['P']=P
            self._result['Q']=Q
            self._result['R']=R

        self._result.update(stats)


    def _get_trial_stats(self):
        import mcmc

        g1vals=self.trials[:,2]
        g2vals=self.trials[:,3]

        prior = self.gprior(g1vals,g2vals)

        w,=numpy.where(prior > 0)
        if w.size == 0:
            raise ValueError("no prior values > 0!")

        dpri_by_g1 = self.gprior.dbyg1(g1vals,g2vals)
        dpri_by_g2 = self.gprior.dbyg2(g1vals,g2vals)

        psum = prior.sum()

        # prior is already in the distribution of points.  This is simpler for
        # most things but for lensfit sensitivity we need a factor of
        # (1/P)dP/de

        pars,pcov = mcmc.extract_stats(self.trials, weights=self.iweights)

        g = pars[2:4].copy()
        gcov = pcov[2:4, 2:4].copy()

        g1diff = g[0]-g1vals
        g2diff = g[1]-g2vals

        gsens = numpy.zeros(2)
        gsens[0]= 1.-(g1diff[w]*dpri_by_g1[w]/prior[w]).mean()
        gsens[1]= 1.-(g2diff[w]*dpri_by_g2[w]/prior[w]).mean()

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

        P,Q,R,w = self._fix_pqr_for_during(g1,g2,P,Q,R)

        iweights = self.iweights[w]

        Qw = Q.copy()
        Qw[:,0] *= iweights
        Qw[:,1] *= iweights

        Rw = R.copy()
        Rw[:,0,0] *= iweights
        Rw[:,0,1] *= iweights
        Rw[:,1,0] *= iweights
        Rw[:,1,1] *= iweights

        Psum = (P*iweights).sum()
        Qsum = Qw.sum(axis=0)
        Rsum = Rw.sum(axis=0)
        
        iwsum = self.iweights_sum_inv
        P *= iwsum
        Q *= iwsum
        R *= iwsum
        
        P = P.mean()
        Q = Q.mean(axis=0)
        R = R.mean(axis=0)

        return P,Q,R

    def _doplots(self):
        import biggles
        import esutil as eu

        res=self.get_result()
        if 's2n_w' not in res:
            print >>stderr,'not plotting, answer not found'
            return

        biggles.configure("default","fontsize_min",1.2)
        tab=biggles.Table(5,1)

        cen1vals=self.trials[:,0]
        cen2vals=self.trials[:,1]
        Tvals=self.trials[:,4]
        g1vals=self.trials[:,2]
        g2vals=self.trials[:,3]
        g1lab=r'$g_1$'
        g2lab=r'$g_2$'

        ampvals=self.trials[:,5]

        g = res['g']
        gcov = res['gcov']
        errs = sqrt(diag(gcov))


        flux=res['pars'][5]
        flux_err=sqrt(res['pcov'][5,5])
        print 'flux: %g +/- %g' % (flux,flux_err)
        print 'T:  %g +/- %g' % (Tvals.mean(), Tvals.std())
        print 's2n weighted:',res['s2n_w']

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
            logTvals=numpy.log10(Tvals[w])
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

        tab[0,0] = hplt_cen
        tab[1,0] = hplt_g1
        tab[2,0] = hplt_g2
        tab[3,0] = hplt_T
        tab[4,0] = hplt_amp

        self.tab=tab

        prompt=self.keys.get('prompt',True)
        if prompt:
            tab.show()
            key=raw_input('hit a key (q to quit): ')
            if key=='q':
                stop
            print

