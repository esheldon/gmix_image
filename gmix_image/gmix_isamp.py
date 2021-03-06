from sys import stderr
import numpy
from numpy import sqrt, diag
from . import gmix_fit
from .gmix_fit import GMIXFIT_CHOLESKY, GMIXFIT_CRAZY_COV
from .gmix_mcmc import MixMCSimple, _get_as_list, _check_lists
from .util import print_pars

from . import priors

import time

class GMixIsampSimple(MixMCSimple):
    def __init__(self,
                 image,
                 weight,
                 psf,

                 cen_prior,
                 gprior,
                 T_prior,
                 counts_prior,

                 prior_samples,

                 guess,
                 model, **keys):
        self.keys=keys

        self.make_plots=keys.get('make_plots',False)
        self.do_pqr=keys.get('do_pqr',True)

        # cen1,cen2,e1,e2,T,p
        self.npars=6
        self.when_prior="during"
        self.minfrac=0.8

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

        self.cen_prior=cen_prior
        self.gprior=gprior
        self.T_prior=T_prior
        self.counts_prior=counts_prior

        self.guess=guess

        self.nsample=keys.get('nsample',500)

        self.verbose=keys.get('verbose',False)

        #self._presample_gprior()
        #self._presample_prior()
        self.prior_samples=prior_samples

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
            if not self._do_isample():
                defres['flags'] = GMIXFIT_CHOLESKY 
                self._result=defres
                return

            if not self._calc_result():
                defres['flags'] = GMIXFIT_CRAZY_COV
                self._result=defres
                return

        if self.make_plots:
            self._doplots()

    def _do_maxlike_fit(self):
        """
        If we get a crazy covariance matrix (large g errors)
        we repeat
        """
        nretry=10
        for i in xrange(nretry):
            gm=gmix_fit.GMixFitMultiSimple(self.im_list,
                                           self.wt_list,
                                           self.jacob_list,
                                           self.psf_list,
                                           self.model,

                                           g_guess=self.guess[2:4],
                                           T_guess=self.guess[4],
                                           counts_guess=self.guess[5],
                                           cen_guess=self.guess[0:2],

                                           gprior=self.gprior,
                                           cen_prior=self.cen_prior,
                                           T_prior=self.T_prior,
                                           counts_prior=self.counts_prior,
                                           lm_max_try=1)

            res=gm.get_result()
            if res['flags']==0:
                pcov=res['pcov']
                if pcov[2,2] < 1 and pcov[3,3] < 1:
                    break
                else:
                    res['flags'] = GMIXFIT_CRAZY_COV


        self._lm_result=res


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

        nsample=self.nsample

        """
        w,=numpy.where(probs_approx == 0)
        if w.size!=0:
            raise ValueError("zero prob approx: %d/%d" % (w.size,nsample))
        """
        w,=numpy.where(probs_approx > 0)
        if w.size == 0:
            return False

        trials = trials[w,:]
        probs_approx=probs_approx[w]
        nsample=w.size

        usefrac = float(nsample)/self.nsample
        if usefrac < self.minfrac:
            mess=('    require %g of samples good, '
                  'but only found %g. Not continuing')
            print >>stderr,mess % (self.minfrac,usefrac)
            return False

        # now evaluate all these to get the true prob
        iweights = numpy.zeros(nsample)

        lnprobs = numpy.zeros(nsample)
        lnprob_max=None
        for i in xrange(nsample):
            pars = trials[i,:]
            lnprobs[i]=self._calc_lnprob(pars)

        lnprobs = lnprobs - lnprobs.max()
        #print lnprobs

        probs = numpy.exp(lnprobs)

        iweights = probs/probs_approx
        wgood,=numpy.where( numpy.isfinite(iweights) )
        if wgood.size != nsample:
            print >>stderr,"    nan found"
            return False

        self.trials=trials
        self.probs=probs
        self.probs_approx=probs_approx
        self.iweights=iweights
        self.iweights_sum=iweights.sum()
        self.iweights_sum_inv= 1./self.iweights_sum

        return True

    def _sample_dist(self):
        return self._sample_dist_with_gauss()
        #return self._sample_dist_prior_only()

    def _sample_dist_prior_only(self):
        import esutil as eu
        from esutil.random import NormalND, LogNormal

        nwalkers=20
        burnin=25

        pars=self._lm_result['pars']
        cov=self._lm_result['pcov']*4

        errs = numpy.sqrt( numpy.diag(cov) ) 

        w,=numpy.where(errs < 0.01)
        if w.size > 0:
            errs[w] = 0.01

        all_samp=self.prior_samples['samples']

        nsample=self.nsample
        nsig=4
        while True:
            w,=numpy.where(  (all_samp[:,0] > (pars[0]-nsig*errs[0]) )
                           & (all_samp[:,0] < (pars[0]+nsig*errs[0]) )
                           & (all_samp[:,1] > (pars[1]-nsig*errs[1]) )
                           & (all_samp[:,1] < (pars[1]+nsig*errs[1]) )
                           & (all_samp[:,2] > (pars[2]-nsig*errs[2]) )
                           & (all_samp[:,2] < (pars[2]+nsig*errs[2]) )
                           & (all_samp[:,3] > (pars[3]-nsig*errs[3]) )
                           & (all_samp[:,3] < (pars[3]+nsig*errs[3]) )
                           & (all_samp[:,4] > (pars[4]-nsig*errs[4]) )
                           & (all_samp[:,4] < (pars[4]+nsig*errs[4]) )
                           & (all_samp[:,5] > (pars[5]-nsig*errs[5]) )
                           & (all_samp[:,5] < (pars[5]+nsig*errs[5]) ) )
            if w.size < nsample:
                nsig+=0.5
                print >>stderr,'    found',w.size,'increasing sigma to',nsig
            else:
                randi  = eu.numpy_util.randind(w.size, nwalkers)
                randi  = w[randi]
                break

        samples = all_samp[randi,:]
        probs = self.prior_samples['prob'][randi]
        return samples,probs


    def _sample_dist_with_gauss(self):
        from esutil.random import NormalND, LogNormal

        nwalkers=20
        burnin=25

        pars=self._lm_result['pars']
        cov=self._lm_result['pcov']*4

        cen_prior = priors.CenPrior(pars[0:2], [cov[0,0], cov[1,1]])

        g1_sigma = numpy.sqrt(cov[2,2])
        g2_sigma = numpy.sqrt(cov[3,3])

        if g1_sigma < 0.001:
            g1_sigma=0.001
        if g2_sigma < 0.001:
            g2_sigma=0.001

        g_dist_gauss = NormalND( pars[2:4], [g1_sigma,g2_sigma])
        gg_prior = priors.GPriorTimesGauss(self.gprior, g_dist_gauss)

        Tg=pars[4]
        T_sigma = numpy.sqrt(cov[4,4])
        if Tg < 0.01:
            Tg=0.01
        if T_sigma < 0.01:
            T_sigma=0.01

        counts_g=pars[5]
        counts_sigma = numpy.sqrt(cov[5,5])
        if counts_g < 0.01:
            counts_g=0.01
        if counts_sigma < 0.001:
            counts_sigma=0.001

        T_prior = LogNormal(Tg, T_sigma)
        counts_prior = LogNormal(counts_g, counts_sigma)
 
        comb=priors.CombinedPriorSimple(cen_prior,
                                        gg_prior,
                                        T_prior,
                                        counts_prior)

        g1rand,g2rand=self._get_gstart(g_dist_gauss,nwalkers)
        if g1rand is None:
            # we will just use the lm fit
            return None,None

        start=numpy.zeros( (nwalkers,self.npars) )

        start[:,0:2] = cen_prior.sample(nwalkers)


        start[:,2] = g1rand
        start[:,3] = g2rand

        start[:,4] = T_prior.sample(nwalkers)
        start[:,5] = counts_prior.sample(nwalkers)

        sampler = comb.sample(start,
                              self.nsample, 
                              burnin=burnin,
                              nwalkers=nwalkers,
                              get_sampler=True)

        prand = sampler.flatchain
        lnp = sampler.lnprobability
        lnp = lnp.reshape(lnp.shape[0]*lnp.shape[1])
        probs = numpy.exp(lnp)
        return prand,probs

    def _get_gstart(self,g_dist_gauss, nwalkers):
        """
        Get good starts for the walkers
        """
        import esutil as eu
        g1vals_pre = self.prior_samples['g1']
        g2vals_pre = self.prior_samples['g2']

        # just for starts; doesn't have to be perfect
        nsig_g = 3

        it1=1
        it2=1
        while True:
            g1_range = numpy.array([g_dist_gauss.mean[0]-nsig_g*g_dist_gauss.sigma[0],
                                    g_dist_gauss.mean[0]+nsig_g*g_dist_gauss.sigma[0]])
            g2_range = numpy.array([g_dist_gauss.mean[1]-nsig_g*g_dist_gauss.sigma[1],
                                    g_dist_gauss.mean[1]+nsig_g*g_dist_gauss.sigma[1]])

            g1_range.clip(-1.0,1.0,g1_range)
            g2_range.clip(-1.0,1.0,g2_range)

            if g1_range[0] == g1_range[1]:
                print >>stderr,'    bad g1 range',g1_range
                print >>stderr,'    sigma(g1):',g_dist_gauss.sigma[0]
                if it1==1:
                    g_dist_gauss.sigma[0] = 0.01
                    it1 += 1
                else:
                    nsig += 1
                continue

            if g2_range[0] == g2_range[1]:
                print >>stderr,'    bad g2 range',g2_range
                print >>stderr,'    sigma(g2):',g_dist_gauss.sigma[1]
                if it2==1:
                    g_dist_gauss.sigma[1] = 0.01
                    it2 += 1
                else:
                    nsig += 1
                continue

            break


        while True:
            w,=numpy.where(  (g1vals_pre > g1_range[0])
                           & (g1vals_pre < g1_range[1])
                           & (g2vals_pre > g2_range[0])
                           & (g2vals_pre < g2_range[1]) )
            if w.size < nwalkers:
                print >>stderr,'    found',w.size,'expanding range'
                #print >>stderr,'    ',g1_range
                #print >>stderr,'    ',g2_range
                # there just aren't enough in this range, so expand it
                g1_range[0]-=0.01
                g1_range[1]+=0.01
                g2_range[0]-=0.01
                g2_range[1]+=0.01

                g1_range.clip(-1.0,1.0,g1_range)
                g2_range.clip(-1.0,1.0,g2_range)

                #print >>stderr,'    ',g1_range
                #print >>stderr,'    ',g2_range
            else:
                randi  = eu.numpy_util.randind(w.size, nwalkers)
                randi  = w[randi]
                g1rand = g1vals_pre[randi]
                g2rand = g2vals_pre[randi]
                break

        return g1rand,g2rand


    def _presample_gprior(self):
        #npre=self.keys.get('n_pre_sample',100000)
        npre=self.keys.get('n_pre_sample',10000)
        self.g1vals_pre,self.g2vals_pre = self.gprior.sample2d(npre)



    def _calc_result(self):
        """
        We marginalize over all parameters but g1,g2, which
        are index 0 and 1 in the pars array
        """

        sres=self._get_trial_stats()
        if sres is None:
            return False
        pars,pcov,g,gcov,gsens=sres
 
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
                      'arate':1.0,
                      'lm_result':self._lm_result}

        if self.do_pqr:
            pqr_res = self._get_PQR()
            if pqr_res is None:
                return False

            P,Q,R = pqr_res
            self._result['P']=P
            self._result['Q']=Q
            self._result['R']=R

        self._result.update(stats)

        return True

    def _get_trial_stats(self):
        import mcmc

        g1vals=self.trials[:,2]
        g2vals=self.trials[:,3]

        prior = self.gprior(g1vals,g2vals)

        w,=numpy.where(prior > 0)
        if w.size == 0:
            print_pars(self._lm_result['pars'],front='lm pars:', stream=stderr)
            print_pars(self._lm_result['perr'],front='lm perr:', stream=stderr)
            print >>stderr,"no prior values > 0!"
            return None

        dpri_by_g1 = self.gprior.dbyg1(g1vals,g2vals)
        dpri_by_g2 = self.gprior.dbyg2(g1vals,g2vals)

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

        pqr_res = self._fix_pqr_for_during(g1,g2,P,Q,R)
        if pqr_res is None:
            return None
        P,Q,R,w = pqr_res

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

