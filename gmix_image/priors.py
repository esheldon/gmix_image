import numpy
from numpy import sqrt, cos, sin, exp, pi, zeros,  \
        random, where, array, log
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

    def lnprob(self, g1, g2):
        """
        log of prob
        """
        p=self(g1,g2)
        if p < 1.e-10:
            return -23.025850929940457
        return log(p)

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


 
    def get_pqr_num(self, g1in, g2in, h=1.e-6):
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

        Q1_p = self.get_pj(g1, g2, +h, 0.0)
        Q1_m = self.get_pj(g1, g2, -h, 0.0)
        Q2_p = self.get_pj(g1, g2, 0.0, +h)
        Q2_m = self.get_pj(g1, g2, 0.0, -h)
        R12_pp = self.get_pj(g1, g2, +h, +h)
        R12_mm = self.get_pj(g1, g2, -h, -h)

        Q1 = (Q1_p - Q1_m)*h2
        Q2 = (Q2_p - Q2_m)*h2

        R11 = (Q1_p - 2*P + Q1_m)*hsq
        R22 = (Q2_p - 2*P + Q2_m)*hsq
        R12 = (R12_pp - Q1_p - Q2_p + 2*P - Q1_m - Q2_m + R12_mm)*hsq*0.5

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

class GPriorTimesGauss(object):
    """
    Gprior*g1_gauss*g2_gauss

    g1,g2 are treated without covariance
    """
    def __init__(self, g_prior, g_gauss):
        self.g_prior = g_prior
        self.g_gauss=g_gauss

        self.ga = numpy.zeros(2)
    def lnprob(self, g1, g2):
        self.ga[0] = g1
        self.ga[1] = g2
        lnprob =   self.g_prior.lnprob(g1,g2) + self.g_gauss.lnprob(self.ga)
        return lnprob

    def prob(self, g1, g2):
        return exp(self.lnprob(g1,g2))

    def sample2d(self, nrand):
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


def plot_pqr(gsigma=0.3, other=0.0):
    import biggles
    biggles.configure('default','fontsize_min',1.0)
    gp = GPriorBA(gsigma)

    n=100
    z=numpy.zeros(n) + other
    gmin=-1
    gmax=1
    g1=numpy.linspace(gmin,gmax,n)
    g2=numpy.linspace(gmin,gmax,n)

    P_g1,Q_g1,R_g1=gp.get_pqr(g1,z)
    P_g2,Q_g2,R_g2=gp.get_pqr(z,g2)

    tab=biggles.Table(3,2)

    plt_P_g1 = biggles.FramedPlot()
    pts_P_g1 = biggles.Points(g1, P_g1, type='filled circle')
    plt_P_g1.add(pts_P_g1)
    plt_P_g1.xlabel='g1'
    plt_P_g1.ylabel='P'

    plt_P_g2 = biggles.FramedPlot()
    pts_P_g2 = biggles.Points(g2, P_g1, type='filled circle')
    plt_P_g2.add(pts_P_g2)
    plt_P_g2.xlabel='g2'
    plt_P_g2.ylabel='P'


    plt_Q_g1 = biggles.FramedPlot()
    pts_Q1_g1 = biggles.Points(g1, Q_g1[:,0], type='filled circle', color='red')
    pts_Q2_g1 = biggles.Points(g1, Q_g1[:,1], type='filled circle', color='blue')
    pts_Q1_g1.label='Q1'
    pts_Q2_g1.label='Q2'

    Qkey_g1 = biggles.PlotKey(0.9,0.2,[pts_Q1_g1,pts_Q2_g1],halign='right')
    plt_Q_g1.add(pts_Q1_g1,pts_Q2_g1,Qkey_g1)
    plt_Q_g1.xlabel='g1'
    plt_Q_g1.ylabel='Q'


    plt_Q_g2 = biggles.FramedPlot()
    pts_Q1_g2 = biggles.Points(g2, Q_g2[:,0], type='filled circle', color='red')
    pts_Q2_g2 = biggles.Points(g2, Q_g2[:,1], type='filled circle', color='blue')
    pts_Q1_g2.label='Q1'
    pts_Q2_g2.label='Q2'

    Qkey_g2 = biggles.PlotKey(0.9,0.2,[pts_Q1_g2,pts_Q2_g2],halign='right')
    plt_Q_g2.add(pts_Q1_g2,pts_Q2_g2,Qkey_g2)
    plt_Q_g2.xlabel='g2'
    plt_Q_g2.ylabel='Q'

    
    plt_R_g1 = biggles.FramedPlot()
    pts_R11_g1 = biggles.Points(g1, R_g1[:,0,0], type='filled circle', color='red')
    pts_R12_g1 = biggles.Points(g1, R_g1[:,0,1], type='filled circle', color='darkgreen')
    pts_R22_g1 = biggles.Points(g1, R_g1[:,1,1], type='filled circle', color='blue')
    pts_R11_g1.label = 'R11'
    pts_R12_g1.label = 'R12'
    pts_R22_g1.label = 'R22'
    Rkey_g1 = biggles.PlotKey(0.9,0.2,[pts_R11_g1,pts_R12_g1,pts_R22_g1],halign='right')

    plt_R_g1.add(pts_R11_g1,pts_R12_g1,pts_R22_g1,Rkey_g1)
    plt_R_g1.xlabel='g1'
    plt_R_g1.ylabel='R'

    plt_R_g2 = biggles.FramedPlot()
    pts_R11_g2 = biggles.Points(g2, R_g2[:,0,0], type='filled circle', color='red')
    pts_R12_g2 = biggles.Points(g2, R_g2[:,0,1], type='filled circle', color='darkgreen')
    pts_R22_g2 = biggles.Points(g2, R_g2[:,1,1], type='filled circle', color='blue')

    pts_R11_g2.label = 'R11'
    pts_R12_g2.label = 'R12'
    pts_R22_g2.label = 'R22'
    Rkey_g2 = biggles.PlotKey(0.9,0.2,[pts_R11_g2,pts_R12_g2,pts_R22_g2],halign='right')

    plt_R_g2.add(pts_R11_g2,pts_R12_g2,pts_R22_g2,Rkey_g2)
    plt_R_g2.xlabel='g2'
    plt_R_g2.ylabel='R'
 
    tab[0,0] = plt_P_g1
    tab[1,0] = plt_Q_g1
    tab[2,0] = plt_R_g1

    tab[0,1] = plt_P_g2
    tab[1,1] = plt_Q_g2
    tab[2,1] = plt_R_g2

    tab.show()

def test_pj_predict(type='BA', rng=[-0.2,0.2]):
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
    binsize=0.005

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

def test_shear_recover_pqr(gsigma=0.3, nchunk=10, nper=1000000, s1=0.04, s2=0.00):
    """
    Shear a bunch of shapes drawn from the prior and try to
    recover using Bernstein & Armstrong
    """
    import lensing

    gpe = GPriorBA(gsigma)

    g1sum=0.0
    g2sum=0.0
    for i in xrange(nchunk):
        print '-'*70
        print '%d/%d' % (i+1,nchunk)
        print 'getting sample'
        rg1,rg2 = gpe.sample2d(nper)

        print 'shearing'
        sheared_rg1,sheared_rg2 = lensing.shear.gadd(rg1,rg2,s1,s2)

        g1sum += sheared_rg1.sum()
        g2sum += sheared_rg2.sum()

        print 'getting P,Q,R'
        Pa,Qa,Ra=gpe.get_pqr(sheared_rg1,sheared_rg2)

        print 'getting measured shear\n'
        shi,Ci,Q_sumi, Cinv_sumi =lensing.shear.get_shear_pqr(Pa,Qa,Ra,get_sums=True)

        if i==0:
            Q_sum = Q_sumi.copy()
            Cinv_sum = Cinv_sumi.copy()
        else:
            Q_sum += Q_sumi
            Cinv_sum += Cinv_sumi

    print '-'*70
    print 'Q_sum:',Q_sum
    print 'Cinv_sum:',Cinv_sum

    C = numpy.linalg.inv(Cinv_sum)
    sh = numpy.dot(C,Q_sum)

    err1=sqrt(C[0,0])
    err2=sqrt(C[1,1])
    print 'input shear:',s1,s2
    print 'meas shear: %g +/- %g  %g +/- %g' % (sh[0],err1,sh[1],err2)
    if s1 != 0:
        print 's1meas/s1-1 %g +/- %g:' % (sh[0]/s1-1, err1/s1)
    if s2 != 0:
        print 's2meas/s1-1 %g +/- %g:' % (sh[1]/s2-1, err2/s2)

    ntot = nchunk*nper
    print '<g1>:',g1sum/ntot
    print '<g2>:',g2sum/ntot


class GPriorBA(GPrior):
    def __init__(self, pars):
        """
        pars are scalar gsigma from B&A 
        """
        super(GPriorBA,self).__init__(pars)

    def get_max(self):
        return 1.0

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


    def get_pqr(self, g1in, g2in):
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
        """

        if numpy.isscalar(g1in):
            isscalar=True
        else:
            isscalar=False

        g1 = numpy.array(g1in, dtype='f8', ndmin=1, copy=False)
        g2 = numpy.array(g2in, dtype='f8', ndmin=1, copy=False)
        #g1 = numpy.array(g1in, dtype='f16', ndmin=1, copy=False)
        #g2 = numpy.array(g2in, dtype='f16', ndmin=1, copy=False)

        # these are the same
        #P=self.get_pj(g1, g2, 0.0, 0.0)
        P=self(g1, g2)


        #g=sqrt(g1**2 + g2**2)
        #w,=numpy.where(g >= 1)
        #if w.size > 0:
        #    print 'g > 1:',w.size
        #    stop

        sigma = self.pars
        sig2 = sigma**2
        sig4 = sigma**4
        sig2inv = 1./sigma**2
        sig4inv = 1./sigma**4

        gsq = g1**2 + g2**2
        omgsq = 1. - gsq

        fac = exp(-0.5*gsq*sig2inv)*omgsq**2

        Qf = fac*(omgsq + 8*sig2)*sig2inv

        Q1 = g1*Qf
        Q2 = g2*Qf

        R11 = (fac * (g1**6 + g1**4*(-2 + 2*g2**2 - 19*sig2) + (1 + g2**2)*sig2*(-1 + g2**2 - 8*sig2) + g1**2*(1 + g2**4 + 20*sig2 + 72*sig4 - 2*g2**2*(1 + 9*sig2))))*sig4inv
        R22 = (fac * (g2**6 + g2**4*(-2 + 2*g1**2 - 19*sig2) + (1 + g1**2)*sig2*(-1 + g1**2 - 8*sig2) + g2**2*(1 + g1**4 + 20*sig2 + 72*sig4 - 2*g1**2*(1 + 9*sig2))))*sig4inv

        R12 = fac * g1*g2 * (80 + omgsq**2*sig4inv + 20*omgsq*sig2inv)

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

class GPriorFlat(GPrior):
    def __init__(self):
        """
        pars are scalar gsigma from B&A 
        """
        self.gmax=1.0

    def prior2d_gabs(self, gin):
        """
        Get the 2d prior for the input |g| value(s)
        """
        iss=numpy.isscalar(gin)
        if iss:
            return self.prior2d_gabs_scalar(gin)

        g=numpy.array(gin,dtype='f8',ndmin=1,copy=False)

        prior=zeros(g.size)

        w,=where(g < 1.0)
        if w.size > 0:
            prior[w] = 1.0

        if iss:
            prior=prior[0]
        return prior

    def prior2d_gabs_scalar(self, g):
        """
        version for scalars
        """
        from math import exp

        if g < 1.0:
            return 1.0
        else:
            return 0.0

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

class Gauss(object):
    def __init__(self, cen, sigma):
        self.cen=cen
        self.sigma=sigma
        self.ivar=1.0/sigma**2

    def lnprob(self, x):
        return -0.5*(x-self.cen)**2*self.ivar

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
        self.cen=numpy.array(cen)
        self.sigma=numpy.array(sigma)
        self.sigma2=numpy.array( [s**2 for s in sigma] )

    def get_max(self):
        return 1.0

    def lnprob(self, pos):
        if len(pos.shape) > 1:
            lnprob0 = -0.5*(self.cen[0]-pos[:,0])**2/self.sigma2[0]
            lnprob1 = -0.5*(self.cen[1]-pos[:,1])**2/self.sigma2[1]
        else:
            lnprob0 = -0.5*(self.cen[0]-pos[0])**2/self.sigma2[0]
            lnprob1 = -0.5*(self.cen[1]-pos[1])**2/self.sigma2[1]

        return lnprob0 + lnprob1

    def sample(self, n=None):
        """
        Get a single sample
        """
        if n is None:
            rand=self.cen + self.sigma*numpy.random.randn(2)
        else:
            rand = numpy.random.randn(n,2).reshape(n,2)
            rand[:,0] *= self.sigma[0]
            rand[:,0] += self.cen[0]

            rand[:,1] *= self.sigma[1]
            rand[:,1] += self.cen[1]

        return rand

class CombinedPriorSimple(object):
    """
    Combine all these prior into one.

    Note covariance between the points is not supported.  This must be
    put into the randoms, for example
    """
    def __init__(self,
                 cen_prior,
                 g_prior,
                 T_prior,
                 counts_prior):

        self.npars=6


        self.cen_prior=cen_prior
        self.g_prior=g_prior
        self.T_prior=T_prior
        self.counts_prior=counts_prior


    def __call__(self, pars_in):
        """
        Get the probability of the input parameters
        """
        lnp=self.lnprob(pars_in)
        return exp(lnp)

    def lnprob(self, pars_in):
        """
        Get the log probability of the input parameters

        parameters
        ----------
        pars: sequence,array
            Array of parameters
        """
        LOWVAL=-9999.0e47

        ndim_in=len(pars_in.shape)

        pars=numpy.atleast_2d(pars_in)
        pshape=pars.shape
        ndim=len(pars.shape)
        if ndim > 2:
            raise ValueError("ndim should be 2")

        lnprob = numpy.zeros( pars.shape[0] )

        try:
            lnprob += self.cen_prior.lnprob(pars[:,0:2])
            lnprob += self.g_prior.lnprob(pars[:,2],pars[:,3])
            lnprob += self.T_prior.lnprob(pars[:,4])
            lnprob += self.counts_prior.lnprob(pars[:,5])
        except ValueError:
            lnprob[:] = LOWVAL

        if ndim_in == 1:
            return lnprob[0]
        else:
            return lnprob

    def sample(self,
               guess,
               nrand, 
               nwalkers=20,
               burnin=25,
               get_sampler=False):
        """
        Get random values in 6-d

        parameters
        ----------
        nrand: int
            Number to generate
        method: string, optional
            'mcmc' or 'cut'
        full: bool, optional
            Return the mcmc sampler
        """
        import emcee

        nstep=nrand/nwalkers

        sampler = emcee.EnsembleSampler(nwalkers, 
                                        self.npars, 
                                        self.lnprob,
                                        a=2)
        pos, prob, state = sampler.run_mcmc(guess, burnin)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nstep)

        if get_sampler:
            return sampler
        else:
            prand = sampler.flatchain
            return prand

def test_combined(show=False,
                  nwalkers=20,
                  burnin=25,
                  nrand=500,
                  ntry=1,
                  gerr=0.1):
    import esutil as eu
    from esutil.random import LogNormal, Normal, NormalND, srandu
    import time
    npars=6

    cen_prior = CenPrior([15.0, 16.0], [0.1, 0.2])

    g_prior = GPriorBA(0.3)
    print 'pre-generating g values from prior (for seed to sampler)'
    ng_pre = 10000
    g1vals_pre,g2vals_pre = g_prior.sample2d(ng_pre)
    print 'done'

    g_dist_gauss = NormalND( [0.2,0.1], [gerr,gerr] )

    gg_prior = GPriorTimesGauss(g_prior, g_dist_gauss)

    T_prior = LogNormal(16.0, 3.0)
    counts_prior = LogNormal(100.0, 10.0)


    comb=CombinedPriorSimple(cen_prior,
                             gg_prior,
                             T_prior,
                             counts_prior)


    nwalkers=20
    print 'nstep per:',nrand/nwalkers
    start=numpy.zeros( (nwalkers,npars) )

    minerr2=0.01**2
    tm=time.time()
    for i in xrange(ntry):
        start[:,0:2] = cen_prior.sample(nwalkers)

        # g start
        g_err2 = g_dist_gauss.sigma[0]**2 + g_dist_gauss.sigma[1]**2

        if g_err2 < minerr2:
            print 'error small enough from max like, skipping resample'
            continue

        nsig_g = 3
        g1_range = numpy.array([g_dist_gauss.mean[0]-nsig_g*g_dist_gauss.sigma[0],
                                g_dist_gauss.mean[0]+nsig_g*g_dist_gauss.sigma[0]])
        g2_range = numpy.array([g_dist_gauss.mean[1]-nsig_g*g_dist_gauss.sigma[1],
                                g_dist_gauss.mean[1]+nsig_g*g_dist_gauss.sigma[1]])

        g1_range.clip(-1.0,1.0,g1_range)
        g2_range.clip(-1.0,1.0,g2_range)
        g1_width = g1_range.max() - g1_range.min()
        g2_width = g2_range.max() - g2_range.min()
        if g1_width==0 or g2_width==0:
            # crazy, probably near e==0.  Just draw from prior
            randi = eu.numpy_util.randind(ng_pre, nwalkers)
            g1rand=g1vals_pre[randi]
            g2rand=g2vals_pre[randi]
        else:

            w,=numpy.where(  (g1vals_pre > g1_range[0])
                           & (g1vals_pre < g1_range[1])
                           & (g2vals_pre > g1_range[0])
                           & (g2vals_pre < g1_range[1]) )
            if w.size < nwalkers:
                raise ValueError("too few")

            randi = eu.numpy_util.randind(w.size, nwalkers)
            randi=w[randi]
            g1rand=g1vals_pre[randi]
            g2rand=g2vals_pre[randi]


        """
        sig2=g_dist_gauss.sigma[0]**2 + g_dist_gauss.sigma[1]**2
        if sig2 < 0.3**2:
            g1rand = g_dist_gauss.mean[0]*(1.0 + 0.01*srandu(nwalkers))
            g2rand = g_dist_gauss.mean[1]*(1.0 + 0.01*srandu(nwalkers))
        else:
            g1rand,g2rand=g_prior_ba.sample2d(nwalkers)
        """

        start[:,2] = g1rand
        start[:,3] = g2rand

        start[:,4] = T_prior.sample(nwalkers)
        start[:,5] = counts_prior.sample(nwalkers)

        prand = comb.sample(start,
                            nrand, 
                            burnin=burnin,
                            nwalkers=nwalkers)

    print 'time per:',(time.time()-tm)/ntry

    print 'g1: %g +/- %g' % (prand[:,2].mean(), prand[:,2].std())
    print 'g2: %g +/- %g' % (prand[:,3].mean(), prand[:,3].std())
    if show:
        plot_many_hist(prand)

def plot_many_hist(arr):
    import biggles
    import esutil as eu
    biggles.configure('default','fontsize_min',1)
    ndim = arr.shape[1]

    nrow,ncol = eu.plotting.get_grid(ndim)

    plt=biggles.Table(nrow,ncol)

    names=['cen1','cen1','g1','g2','T','counts']
    for dim in xrange(ndim):
        row=dim/ncol
        col=dim % ncol

        bsize=0.2*arr[:,dim].std()

        p=eu.plotting.bhist(arr[:,dim],binsize=bsize,
                            xlabel=names[dim],
                            show=False)
        p.aspect_ratio=1
        plt[row,col] = p

    plt.show()


    
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

