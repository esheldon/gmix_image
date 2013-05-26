import numpy
from numpy import sqrt, cos, sin, exp, pi, zeros,  \
        random, where, array
import math


class GPrior(object):
    """
    This is the base class.  You need to over-ride a few of
    the functions, see below
    """
    def __init__(self, pars):
        self.pars=pars
        self.maxval = self(0., 0.)

        # sub-class may want to over-ride this, see GPriorExp
        self.gmax=1.0

    def __call__(self, g1, g2):
        """
        Get the 2d prior
        """
        g = sqrt(g1**2 + g2**2)
        return self.prior2d_gabs(g)

    def prior2d_gabs(self, g):
        """
        Get the 2d prior for the input |g| value(s)
        """
        raise RuntimeError("over-ride")

    def prior2d_gabs_scalar(self, g):
        """
        Get the 2d prior for the input |g| scalar value
        """
        raise RuntimeError("over-ride")


    def prior1d(self, g):
        """
        Get the 1d prior for an input |g| value(s).
        """
        return 2*pi*g*self.prior2d_gabs(g)


    def dbyg1(self, g1, g2, h=1.e-6):
        """
        Derivative with respect to g1 at the input g1,g2 location

        Uses central difference and a small enough step size
        to use just two points
        """
        ff = self(g1+h/2, g2)
        fb = self(g1-h/2, g2)

        return (ff - fb)/h

    def dbyg2(self, g1, g2, h=1.e-6):
        """
        Derivative with respect to g2 at the input g1,g2 location

        Uses central difference and a small enough step size
        to use just two points
        """
        ff = self(g1, g2+h/2)
        fb = self(g1, g2-h/2)
        return (ff - fb)/h


    def get_pqr(self, g1in, g2in, h=1.e-6):
        """
        Evaluate 
            P
            Q
            R
        From Bernstein & Armstrong

        P is this prior times the jacobian at shear==0

        Q is the gradient of P*J evaluated at shear==0

            [ d(P*J)/dg1, d(P*J)/dg2]_{g=0}

        R is grad of grad of P*J at shear==0
            [ d(P*J)/dg1dg1  d(P*J)/dg1dg2 ]
            [ d(P*J)/dg1dg2  d(P*J)/dg2dg2 ]_{g=0}

        Derivatives are calculated using finite differencing
        """
        if numpy.isscalar(g1in):
            isscalar=True
        else:
            isscalar=False

        g1 = numpy.array(g1in, dtype='f8', ndmin=1, copy=False)
        g2 = numpy.array(g2in, dtype='f8', ndmin=1, copy=False)
        h2=1./(2.*h)
        hsq=1./h**2

        P=self.get_pj(g1, g2, 0.0, 0.0)

        Q1_1 = self.get_pj(g1, g2, +h, 0.0)
        Q1_2 = self.get_pj(g1, g2, -h, 0.0)
        Q1 = (Q1_1 - Q1_2)*h2

        Q2_1 = self.get_pj(g1, g2, 0.0, +h)
        Q2_2 = self.get_pj(g1, g2, 0.0, -h)
        Q2 = (Q2_1 - Q2_2)*h2

        R11_1 = self.get_pj(g1, g2, +h, +h)
        R11_2 = self.get_pj(g1, g2, -h, -h)

        R11 = (Q1_1 - 2*P + Q1_2)*hsq
        R22 = (Q2_1 - 2*P + Q2_2)*hsq
        R12 = (R11_1 - Q1_1 - Q2_1 + 2*P - Q1_2 - Q2_2 + R11_2)*hsq*0.5

        np=g1.size
        Q = numpy.zeros( (np,2) )
        R = numpy.zeros( (np,2,2) )

        Q[:,0] = Q1
        Q[:,1] = Q2
        R[:,0,0] = R11
        R[:,0,1] = R12
        R[:,1,0] = R12
        R[:,1,1] = R22

        if isscalar:
            P = P[0]
            Q = Q[0,:]
            R = R[0,:,:]

        return P, Q, R

    def get_pj(self, g1, g2, s1, s2):
        """
        PJ = p(g,-shear)*jacob

        where jacob is d(es)/d(eo) and
        es=eo(+)(-g)
        """
        import lensing

        # note sending negative shear to jacob
        s1m=-s1
        s2m=-s2
        J=lensing.shear.dgs_by_dgo_jacob(g1, g2, s1m, s2m)

        # evaluating at negative shear
        g1new,g2new=lensing.shear.gadd(g1, g2, s1m, s2m)
        P=self(g1new,g2new)

        return P*J

    def sample2d_pj(self, nrand, s1, s2):
        """
        Get random g1,g2 values from an approximate
        sheared distribution

        parameters
        ----------
        nrand: int
            Number to generate
        """
        from .util import srandu

        maxval_2d = self(0.0,0.0)
        g1,g2=zeros(nrand),zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:

            # generate on cube [-1,1,h]
            g1rand=srandu(nleft)
            g2rand=srandu(nleft)

            # a bit of padding since we are modifying the distribution
            fac=1.3
            h = fac*maxval_2d*random.random(nleft)

            pjvals = self.get_pj(g1rand,g2rand,s1,s2)
            
            #wbad,=where(pjvals > fac*maxval_2d)
            #if wbad.size > 0:
            #    raise ValueError("found %d > maxval" % wbad.size)

            w,=where(h < pjvals)
            if w.size > 0:
                g1[ngood:ngood+w.size] = g1rand[w]
                g2[ngood:ngood+w.size] = g2rand[w]
                ngood += w.size
                nleft -= w.size
   
        return g1,g2



    def sample1d(self, nrand):
        """
        Get random |g| from the 1d distribution

        Set self.gmax appropriately

        parameters
        ----------
        nrand: int
            Number to generate
        """

        if not hasattr(self,'maxval1d'):
            self.set_maxval1d()

        g = zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:

            # generate total g in [0,1)
            grand = self.gmax*random.random(nleft)

            # now the height from [0,maxval)
            h = self.maxval1d*random.random(nleft)

            pvals = self.prior1d(grand)

            w,=where(h < pvals)
            if w.size > 0:
                g[ngood:ngood+w.size] = grand[w]
                ngood += w.size
                nleft -= w.size
   
        return g


    def sample2d(self, nrand):
        """
        Get random g1,g2 values by first drawing
        from the 1-d distribution

        parameters
        ----------
        nrand: int
            Number to generate
        """

        grand=self.sample1d(nrand)
        rangle = random.random(nrand)*2*pi
        g1rand = grand*cos(rangle)
        g2rand = grand*sin(rangle)
        return g1rand, g2rand

    def sample2d_brute(self, nrand):
        """
        Get random g1,g2 values using 2-d brute
        force method

        parameters
        ----------
        nrand: int
            Number to generate
        """
        from .util import srandu

        maxval_2d = self(0.0,0.0)
        g1,g2=zeros(nrand),zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:

            # generate on cube [-1,1,h]
            g1rand=srandu(nleft)
            g2rand=srandu(nleft)

            # a bit of padding since we are modifying the distribution
            h = maxval_2d*random.random(nleft)

            vals = self(g1rand,g2rand)
            
            #wbad,=where(vals > maxval_2d)
            #if wbad.size > 0:
            #    raise ValueError("found %d > maxval" % wbad.size)

            w,=where(h < vals)
            if w.size > 0:
                g1[ngood:ngood+w.size] = g1rand[w]
                g2[ngood:ngood+w.size] = g2rand[w]
                ngood += w.size
                nleft -= w.size
   
        return g1,g2



    def set_maxval1d(self):
        """
        Use a simple minimizer to find the max value of the 1d 
        distribution
        """
        import scipy.optimize

        (minvalx, fval, iterations, fcalls, warnflag) \
                = scipy.optimize.fmin(self.prior1dneg,
                                      0.1,
                                      full_output=True, 
                                      disp=False)
        if warnflag != 0:
            raise ValueError("failed to find min: warnflag %d" % warnflag)
        self.maxval1d = -fval

    def prior1dneg(self, g, *args):
        """
        So we can use the minimizer
        """
        return -self.prior1d(g)


def test_pj_predict(type='exp', rng=[-0.2,0.2]):
    """
    Test how well the formalism predicts the sheared distribution
    """
    import lensing
    import esutil as eu
    import gmix_image

    s1=0.05
    s2=0.0

    if type=='exp':
        pars=[87.2156230877,
              1.30395318005,
              0.0641620331281,
              0.864555484617]

        gpe = gmix_image.priors.GPriorExp(pars)
    elif type=='BA':
        gsigma=0.3
        gpe = gmix_image.priors.GPriorBA(gsigma)

    nr=1000000
    binsize=0.01

    rg1,rg2 = gpe.sample2d(nr)
    #rg1_bf,rg2_bf = gpe.sample2d_brute(nr)
    #plt=eu.plotting.bhist(rg1,binsize=binsize,show=False)
    #eu.plotting.bhist(rg1_bf,binsize=binsize,color='red',plt=plt)
    #stop

    sheared_rg1,sheared_rg2 = lensing.shear.gadd(rg1,rg2,s1,s2)

    sheared_rg1_predict,sheared_rg2_predict = gpe.sample2d_pj(nr, s1, s2)


    plt=eu.plotting.bhist(sheared_rg1,
                          min=rng[0],max=rng[1],
                          binsize=binsize,
                          show=False)

    eu.plotting.bhist(sheared_rg1_predict,
                      min=rng[0],max=rng[1],
                      binsize=binsize,
                      color='red',
                      xrange=rng,
                      plt=plt,title='full')

def test_shear_recover_pqr(type='BA', nr=1000000, s1=0.05,s2=-0.03,h=1.e-6):
    """
    Shear a bunch of shapes drawn from the prior and try to
    recover using Bernstein & Armstrong
    """
    import lensing

    if type=='BA':
        gsigma=0.3
        gpe = GPriorBA(gsigma)
    else:
        raise ValueError("implement %s as differentiable prior" % type)

    rg1,rg2 = gpe.sample2d(nr)

    sheared_rg1,sheared_rg2 = lensing.shear.gadd(rg1,rg2,s1,s2)

    Pa,Qa,Ra=gpe.get_pqr(sheared_rg1,sheared_rg2,h=1.e-6)

    sh,C=lensing.shear.get_shear_pqr(Pa,Qa,Ra)
    print 'input shear:',s1,s2
    print 'meas shear: %g +/- %g  %g +/- %g' % (sh[0],sqrt(C[0,0]),sh[1],sqrt(C[1,1]))



class GPriorBA(GPrior):
    def __init__(self, pars):
        """
        pars are scalar gsigma from B&A 
        """
        super(GPriorBA,self).__init__(pars)

    def prior2d_gabs(self, gin):
        """
        Get the 2d prior for the input |g| value(s)
        """
        iss=numpy.isscalar(gin)

        g=numpy.array(gin,dtype='f8',ndmin=1,copy=False)

        prior=zeros(g.size)

        w,=where(g < 1.0)
        if w.size > 0:
            g2=g[w]**2
            prior[w] = (1-g2)**2*exp(-g2/2/self.pars**2)

        if iss:
            prior=prior[0]
        return prior



    def prior2d_gabs_scalar(self, g):
        """
        version for scalars
        """
        from math import exp

        if g < 1.0:
            g2=g**2
            prior = (1-g2)**2*exp(-g2/2/self.pars**2)
        else:
            prior = 0.0

        return prior




class GPriorExp(GPrior):
    def __init__(self, pars):
        """
        [A, a, g0, gmax]
        """
        super(GPriorExp,self).__init__(pars)
        self.gmax=pars[-1]

    def prior2d_gabs(self, g):
        """
        Get the 2d prior for the input |g| value(s)
        """
        if numpy.isscalar(g):
            return gprior2d_exp_scalar(self.pars, g)
        else:
            return gprior2d_exp_vec(self.pars, g)

    def prior2d_gabs_scalar(self, g):
        """
        Get the 2d prior for the input |g| scalar value
        """
        return gprior2d_exp_scalar(self.pars, g)


def gprior2d_exp_vec(pars, g):
    A=pars[0]
    a=pars[1]
    g0=pars[2]
    gmax=pars[3]

    prior=zeros(g.size)

    w,=where(g < gmax)
    if w.size > 0:
        numer = A*(1-exp( (g-gmax)/a ))
        denom = (1+g)*sqrt(g**2 + g0**2)

        prior[w]=numer[w]/denom[w]

    return prior

def gprior2d_exp_scalar(pars, g):
    from math import exp, sqrt
    A=pars[0]
    a=pars[1]
    g0=pars[2]
    gmax=pars[3]

    if g > gmax:
        return 0.0

    numer = A*(1-exp( (g-gmax)/a ))
    denom = (1+g)*sqrt(g**2 + g0**2)

    prior=numer/denom

    return prior


def gprior1d_exp_vec(pars, g):
    return 2*pi*g*gprior2d_exp_vec(pars, g)


class GPriorExpFitterFixedGMax:
    def __init__(self, xvals, yvals, gmax=0.87):
        """
        Input is the histogram data
        """
        self.xvals=xvals
        self.yvals=yvals
        self.gmax=gmax

    def __call__(self, pars):
        w,=where(pars < 0)
        if w.size > 0:
            return zeros(self.xvals.size) + numpy.inf

        send_pars=list(pars) + [self.gmax]

        model=gprior1d_exp_vec(send_pars, self.xvals)
        return model-self.yvals

class GPriorExpFitter:
    def __init__(self, xvals, yvals, ivar, Aprior=None, Awidth=None, gmax_min=None):
        """
        Fit with gmax free
        Input is the histogram data
        """
        self.xvals=xvals
        self.yvals=yvals
        self.ivar=ivar

        self.Aprior=Aprior
        self.Awidth=Awidth

        self.gmax_min=gmax_min


    def get_lnprob(self, pars):
        w,=where(pars < 0)
        if w.size > 0:
            return -9.999e20
        if pars[3] > 1:
            return -9.999e20

        if self.gmax_min is not None:
            if pars[-1] < self.gmax_min:
                return -9.999e20

        model=gprior1d_exp_vec(pars, self.xvals)

        chi2 = (model - self.yvals)**2
        chi2 *= self.ivar

        lnprob = -0.5*chi2.sum()

        if self.Aprior is not None and self.Awidth is not None:
            aprior=-0.5*( (self.Aprior-pars[0])/self.Awidth )**2
            lnprob += aprior

        return lnprob


def fit_gprior_exp_mcmc(xdata, ydata, ivar, a=0.25, g0=0.1, gmax=0.87, gmax_min=None, Awidth=1.0):
    """
    This works much better than the lm fitter
    Input is the histogram data.
    """
    import mcmc
    import emcee

    nwalkers=200
    burnin=100
    nstep=100

    print 'fitting exp'

    A=ydata.sum()*(xdata[1]-xdata[0])

    pcen=[A,a,g0,gmax]
    npars=4
    guess=zeros( (nwalkers,npars) )
    guess[:,0] = pcen[0]*(1.+0.1*srandu(nwalkers))
    guess[:,1] = pcen[1]*(1.+0.1*srandu(nwalkers))
    guess[:,2] = pcen[2]*(1.+0.1*srandu(nwalkers))
    guess[:,3] = pcen[3]*(1.+0.1*srandu(nwalkers))


    gfitter=GPriorExpFitter(xdata, ydata, ivar, Aprior=A, Awidth=Awidth, gmax_min=gmax_min)

    print 'pcen:',pcen

    sampler = emcee.EnsembleSampler(nwalkers, 
                                    npars,
                                    gfitter.get_lnprob,
                                    a=2)

    pos, prob, state = sampler.run_mcmc(guess, burnin)
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nstep)

    trials  = sampler.flatchain

    pars,pcov=mcmc.extract_stats(trials)

    d=diag(pcov)
    perr = sqrt(d)

    res={'A':pars[0],
         'A_err':perr[0],
         'a':pars[1],
         'a_err':perr[1],
         'g0':pars[2],
         'g0_err':perr[2],
         'gmax': pars[3],
         'gmax_err':perr[3],
         'pars':pars,
         'pcov':pcov,
         'perr':perr}


    fmt="""
A:    %(A).6g +/- %(A_err).6g
a:    %(a).6g +/- %(a_err).6g
g0:   %(g0).6g +/- %(g0_err).6g
gmax: %(gmax).6g +/- %(gmax_err).6g
    """.strip()

    print fmt % res

    return res


class GPriorDev(GPrior):
    def __init__(self, pars):
        """
        I'm not using this for anything right now.  In clusterstep
        the exp model works better in all cases pretty much.
        [A,b,c]
        """
        super(GPriorDev,self).__init__(pars)

    def prior2d_gabs(self, g):
        """
        Get the 2d prior for the input |g| value(s)
        """
        if numpy.isscalar(g):
            return gprior2d_vec_scalar(self.pars, g)
        else:
            return gprior2d_vec_vec(self.pars, g)



    def prior2d_gabs_scalar(self, g):
        """
        Get the 2d prior for the input |g| scalar value
        """
        return gprior2d_dev_scalar(self.pars, g)




def gprior2d_dev_vec(pars, g):
    A=pars[0]
    b=pars[1]
    c=pars[2]
    return A*exp( -b*g - c*g**2 )

def gprior1d_dev_vec(pars, g):
    return 2*pi*g*gprior2d_dev_vec(pars, g)

def gprior2d_dev_scalar(pars, g):
    from math import exp
    A=pars[0]
    b=pars[1]
    c=pars[2]
    return A*exp( -b*g - c*g**2 )



class GPriorDevFitter:
    def __init__(self, xvals, yvals):
        """
        Input is the histogram data
        """
        self.xvals=xvals
        self.yvals=yvals

    def __call__(self, pars):
        w,=where(pars < 0)
        if w.size > 0:
            return zeros(self.xvals.size) + numpy.inf

        model=gprior1d_dev_vec(pars, self.xvals)
        return model-self.yvals



def fit_gprior_dev(xdata, ydata):
    """
    Input is the histogram data, should be close to
    normalized
    """
    from scipy.optimize import leastsq


    A=ydata.sum()*(xdata[1]-xdata[0])
    b=2.3
    c=6.7

    pstart=[A,b,c]
    print 'fitting dev'
    print 'pstart:',pstart
    gfitter=GPriorDevFitter(xdata, ydata)
    res = leastsq(gfitter, pstart, full_output=1)

    pars, pcov0, infodict, errmsg, ier = res

    if ier == 0:
        raise ValueError("bad args")

    if ier > 4:
        raise ValueError("fitting failed with\n    %s" % errmsg)

    pcov=None
    perr=None
    if pcov0 is None:
        raise ValueError("pcov0 is None")

    dof=xdata.size-pars.size

    ydiff=gfitter(pars)
    s_sq = (ydiff**2).sum()/dof
    pcov = pcov0 * s_sq 

    d=diag(pcov)
    w,=where(d < 0)

    if w.size > 0:
        raise ValueError("negative diag: %s" % d[w])

    perr = sqrt(d)

    print """    A:  %.6g +/- %.6g
    b:  %.6g +/- %.6g
    c:  %.6g +/- %.6g
    """ % (pars[0],perr[0],
           pars[1],perr[1],
           pars[2],perr[2])

    return {'A':pars[0],
            'b':pars[1],
            'c':pars[2],
            'pars':pars,
            'pcov':pcov,
            'perr':perr}



def fit_gprior_exp_gmix(g1, g2):
    import esutil as eu
    from scikits.learn import mixture
    ngauss=5
    gmm = mixture.GMM(n_states=ngauss)

    vals = zeros(g1.size, 2)
    vals[:,0] = g1
    vals[:,1] = g2

    gmm.fit(vals, n_iter=400)#, min_covar=1.e-6)

    mg = MultiGauss(gmm)
    print mg

    return mg


class TPrior(object):
    """
    Prior on T.  
    
    The actual underlying distribution is a lognormal on 

        sigma=sqrt(T/2)

    And it is the mean sigma and the width that are passed to the constructor

    """
    def __init__(self, sigma_mean, sigma_width):
        self.sigma_mean=sigma_mean
        self.sigma_width=sigma_width
        
        self._set_prior()

    def lnprob(self, T):
        sigma=sqrt(T/2)
        return self.ln.lnprob(sigma)

    def _set_prior(self):
        from esutil.random import LogNormal

        self.ln=LogNormal(self.sigma_mean, self.sigma_width)

class MultiGauss:
    def __init__(self, gmm):
        """
        Takes a gmm object

        eval(x) returns normalized evaluation of gaussians
        """
        self.gmm = gmm

    def __repr__(self):
        mess=[]
        gmm = self.gmm
        for i in xrange(gmm.n_states):
            mean = gmm.means[i,0]
            var = gmm.covars[i][0][0]
            weight = gmm.weights[i]
            mess0='p: %.6g x0: %.6g s: %.6g' 
            mess0=mess0 % (weight, mean, sqrt(var))
            mess.append(mess0)

        return '\n'.join(mess)

    def eval(self, x):
        """
        Actually this can just be exp(gmm.eval(x))
        """
        model = numpy.zeros(x.size, dtype='f8')

        gmm = self.gmm
        for i in xrange(gmm.n_states):
            mean = gmm.means[i,0]
            var = gmm.covars[i][0][0]
            weight = gmm.weights[i]
            g = self.gauss(x, mean, var)
            model[:] += weight*g


        return model

    def evalone(self, x, i):
        """
        Just evaluate one of the gaussians
        """

        gmm = self.gmm
        mean = gmm.means[i,0]
        var = gmm.covars[i][0][0]
        weight = gmm.weights[i]
        return weight*self.gauss(x, mean, var)


    def gauss(self, x, mean, var):
        siginv2 = 1./var

        g = exp(-0.5*(x-mean)**2*siginv2)
        norm = sqrt(siginv2/2./numpy.pi)

        g *= norm
        return g





class GPriorOld:
    """
    This is in g1,g2 space

    2D
    Prob = A cos(|g| pi/2) exp( - [ 2 |g| / B / (1 + |g|^D) ]^C )
    d/dE(  A cos(sqrt(E^2+q^2) pi/2) exp( - ( 2 sqrt(E^2 + q^2) / B / (1 + sqrt(E^2 + q^2)^D) )^C ) )

    For 1D prob, you need to multiply by 2*pi*|g|
    """
    def __init__(self, A=12.25, B=0.03, C=0.45, D=13.):
        # A actually depends on norm when doing full thing
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.maxval = self(0., 0.)

    def __call__(self, g1, g2):
        """
        Get the prior for the input g1,g2 value(s)

        This is the 2 dimensional prior.  If you want to just
        generate the |g| values, use prior1d
        """
        g = sqrt(g1**2 + g2**2)
        return self.prior_gabs(g)

    def dbyg1(self, g1, g2, h=1.e-6):
        """
        Derivative with respect to g1 at the input g1,g2 location

        Uses central difference and a small enough step size
        to use just two points
        """
        ff = self(g1+h/2, g2)
        fb = self(g1-h/2, g2)

        return (ff - fb)/h

    def dbyg2(self, g1, g2, h=1.e-6):
        """
        Derivative with respect to g2 at the input g1,g2 location

        Uses central difference and a small enough step size
        to use just two points
        """
        ff = self(g1, g2+h/2)
        fb = self(g1, g2-h/2)
        return (ff - fb)/h


    def prior_gabs(self, g):
        """
        Get the 2d prior for the input |g| value(s)
        """
        g = array(g, ndmin=1, copy=False)
        prior = zeros(g.size)

        w,=where(g < 1)
        if w.size > 0:
            prior[w] = self.A * cos(g[w]*pi/2)*exp( - ( 2*g[w] / self.B / (1 + g[w]**self.D) )**self.C )
        return prior

    def prior2d_gabs_scalar(self, g):
        """
        Get the 2d prior for the input |g| value(s)
        """
        return self.A * math.cos(g*pi/2)*math.exp( - ( 2*g / self.B / (1 + g**self.D) )**self.C )

    def sample2d(self, nrand, as_shear=False):
        """
        Get random g1,g2 values

        parameters
        ----------
        nrand: int
            Number to generate
        as_shear: bool, optional
            If True, get a list of Shear objects
        """
        g1 = zeros(nrand)
        g2 = zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:

            # generate total g**2 in [0,1)
            grand2 = random.random(nleft)
            grand = sqrt(grand2)
            # now uniform angles
            rangle = random.random(nleft)*2*pi

            # now get cartesion locations in g1,g2 plane
            g1rand = grand*cos(rangle)
            g2rand = grand*sin(rangle)

            # now finally the height from [0,maxval)
            h = self.maxval*random.random(nleft)

            pvals = self(g1rand, g2rand)

            w,=where(h < pvals)
            if w.size > 0:
                g1[ngood:ngood+w.size] = g1rand[w]
                g2[ngood:ngood+w.size] = g2rand[w]
                ngood += w.size
                nleft -= w.size

        if as_shear:
            from lensing.shear import Shear
            shlist=[]
            for g1i,g2i in zip(g1,g2):
                shlist.append(Shear(g1=g1i,g2=g2i))
            return shlist
        else:
            return g1, g2


    def prior1d(self, g):
        """
        Get the 1d prior for an input |g| value(s).

        To generate 2-d g1,g2, use prior()
        """
        return 2*pi*g*self.prior_gabs(g)

    def sample1d(self, nrand):
        """
        Get random |g| from the 1d distribution

        parameters
        ----------
        nrand: int
            Number to generate
        """

        if not hasattr(self,'maxval1d'):
            self.set_maxval1d()

        g = zeros(nrand)

        ngood=0
        nleft=nrand
        while ngood < nrand:

            # generate total g**2 in [0,1)
            grand = random.random(nleft)

            # now finally the height from [0,maxval)
            h = self.maxval1d*random.random(nleft)

            pvals = self.prior1d(grand)

            w,=where(h < pvals)
            if w.size > 0:
                g[ngood:ngood+w.size] = grand[w]
                ngood += w.size
                nleft -= w.size
   
        return g

    def set_maxval1d(self):
        import scipy.optimize
        
        (minvalx, fval, iterations, fcalls, warnflag) \
                = scipy.optimize.fmin(self.prior1dneg, 0.1, full_output=True, 
                                      disp=False)
        if warnflag != 0:
            raise ValueError("failed to find min: warnflag %d" % warnflag)
        self.maxval1d = -fval

    def prior1dneg(self, g, *args):
        """
        So we can use the minimizer
        """
        return -self.prior1d(g)

class CenPrior:
    def __init__(self, cen, sigma):
        self.cen=cen
        self.sigma=sigma
        self.sigma2=[s**2 for s in sigma]

    def lnprob(self, pos):
        lnprob0 = -0.5*(self.cen[0]-pos[0])**2/self.sigma2[0]
        lnprob1 = -0.5*(self.cen[1]-pos[1])**2/self.sigma2[1]
        return lnprob0 + lnprob1




def test_shear_mean():
    gp=GPrior(A= 12.25,
              B= 0.2,
              C= 1.05,
              D= 13.)
    n=100000
    e1=zeros(n)
    g1true,g2true=gp.sample2d(n)
    shear=Shear(g1=0.04,g2=0.0)

    g1meas=zeros(n)
    for i in xrange(n):
        s=Shear(g1=g1true[i],g2=g2true[i])
        ssum = s + shear

        g1meas[i] = ssum.g1

    g1mean=g1meas.mean()
    g1err=g1meas.std()/sqrt(n)
    print 'shear: %.16g +/- %.16g' % (g1mean,g1err)

