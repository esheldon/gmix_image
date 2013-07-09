from . import prob
from .gmix_mcmc import MixMCSimple, _get_as_list, _check_lists

class MixEtaSimple(object):
    """
    For now requiring eta_prior also sent, for derivatives etc.
    """
    def __init__(self, im_list, wt_list, psf_list, guess, config, **keys):
        self.npars=6
        self.keys=keys

        # temporary
        self.eta_prior=keys['eta_prior']
        self.when_prior="during"

        self.im_list=_get_as_list(im_list)
        self.nimage=len(self.im_list)

        self._set_wt_list(wt_list)
        self._set_jacob_list(**keys)
        self.psf_list=_get_as_list(psf_list)

        _check_lists(self.im_list, self.wt_list, self.psf_list,self.jacob_list)

        self.guess=guess

        self.config=config
        self.model=config['model']

        self.nwalkers = guess.shape[0]

        if self.guess.shape[1] != self.npars:
            raise ValueError("guess must be length %d" % self.npars)

        self.burnin=config['burnin']
        self.nstep=config['nstep']

        self.imsize=self.im_list[0].size
        self.totpix=self.nimage*self.imsize
        self.mca_a=config['mca_a']
        self.doiter = config['iter']

        self.make_plots=config['make_plots']
        self.do_pqr=config['do_pqr']

        self.prob_obj = prob.Prob(self.im_list,
                                  self.wt_list,
                                  self.jacob_list,
                                  self.psf_list,
                                  self.config)

        self._go()

    def _get_guess(self):
        return self.guess

    def _calc_lnprob(self, eta_pars):
        lnprob,s2n_numer,s2n_denom,flags = self.prob_obj.get_lnprob(eta_pars)
        return lnprob

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

