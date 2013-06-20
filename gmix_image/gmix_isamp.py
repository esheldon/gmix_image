from .gmix_mcmc import MixMCSimple

class GMixIsampSimple(MixMCSimple):
    def __init__(self, image, weight, psf, gprior, T_guess, counts_guess, cen_guess, model, **keys):
        self.keys=keys

        self.make_plots=keys.get('make_plots',False)
        self.do_pqr=keys.get('do_pqr',True)

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
        self.g_guess=keys.get('g_guess',None)

        self.cen_prior=keys.get('cen_prior',None)
        if self.cen_prior is None:
            self.cen_width=keys.get('cen_width',1.0)
            self.cen_prior=CenPrior(self.cen_guess, [self.cen_width]*2)

        self.Tprior=keys.get('Tprior',None)
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
        if self._lm_result['flags'] != 0:
            self._result={'flags':self._lm_result['flags'],
                          'lm_result':self._lm_result}
        else:
            # self.trials and self.lnprobs are created
            self._do_isample()
            self._calc_result()
        """
        self.trials  = self.sampler.flatchain

        lnprobs = self.sampler.lnprobability.reshape(self.nwalkers*self.nstep)
        self.lnprobs = lnprobs - lnprobs.max()

        # get the expectation values, sensitivity and errors
        self._calc_result()

        """
        if self.make_plots:
            self._doplots()

    def _do_maxlike_fit(self):
        guess=self._get_lm_guess()
        gm=GMixFitMultiSimple(self.im_list,
                              self.wt_list,
                              self.jacob_list,
                              self.psf_list,
                              self.model,

                              g_guess=self.g_guess,
                              T_guess=self.T_guess,
                              counts_guess=self.counts_guess,
                              cen_guess=self.cen_guess,

                              gprior=self.gprior,
                              cen_prior=self.cen_prior
                              T_prior=elf.T_prior,
                              counts_prior=self.counts_prior)


        self._lm_result=gm.get_result()

    def _do_isample(self):
        """
        Sample from approximate posterior based on likelihood
        and errors.  This gets the trials and probs_approx

        Then evaluate the actual probs to get prob, and weight
        the trials by prob/probs_approx in our statistics

        """
        # first draw from a distribution representing the
        # max like point plus errors
        trials, probs_approx = self._sample_dist()

        # now evaluate all these to get the true prob
        iweights = numpy.zeros(self.nsample)
        probs = numpy.zeros(self.nsample)

        for i in xrange(self.nsample):
            pars = trials[i,:]

            prob = self._calc_lnprob(pars)

            probs[i] = prob
            iweights[i] = prob/probs_approx[i]

        self.trials=trials
        self.probs=probs
        self.probs_approx=probs_approx
        self.iweights=iweights
        self.iweights_sum=iweights.sum()
        self.iweights_sum_inv= 1./self.iweights_sum

    def _sample_dist(self):
        """
        Represent the n-dimensional distribution as a multi-variate Cauchy
        distribution.  Draw from it and evaluate the probability.  Also
        multiply by our priors.
        """
        from esutil.random import Cauchy

        pars=self._lm_result['pars']
        cov=self._lm_result['pcov']

        dist=Cauchy(pars, cov)

        trials = dist.sample(self.nsample)
        pvals = dist(trials)

        return trials, pvals


    def _get_trial_stats(self):
        import mcmc

        g1vals=self.trials[:,2]
        g2vals=self.trials[:,3]

        prior = self.gprior(g1vals,g2vals)

        w,=where(prior > 0)
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

        gsens = zeros(2)
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

        P,Q,R = self._fix_pqr_for_during(g1,g2,P,Q,R)

        iweights = self.iweights

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
        
        return P,Q,R


