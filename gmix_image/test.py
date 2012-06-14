import os
from sys import stderr
import numpy
from numpy import sqrt, array, ogrid, random, exp, zeros, cos, sin, diag, \
        tanh, pi
from numpy.random import random as randu
import gmix_image
import gmix_fit
from .gmix_fit import print_pars, ellip2eta, eta2ellip
from .gmix_em import gmix2image, gmix2image_psf
import copy

import esutil as eu

try:
    import images
    have_images=True
except:
    have_images=False

#
# EM tests
#

def test_all():
    test(add_noise=False)
    test(add_noise=True)
    test_psf()
    test_psf(add_noise=True)
    test_psf_colocate(add_noise=False, npsf=1)
    test_psf_colocate(add_noise=True, npsf=1)
    test_psf_colocate(add_noise=False, npsf=2)
    test_psf_colocate(add_noise=True, npsf=2)
def test(add_noise=False):
    print '\nnoise:',add_noise
    tol=1.e-6
    dims=[31,31]
    gd = [{'p':0.6,'row':15,'col':15,'irr':2.0,'irc':0.0,'icc':2.0},
          {'p':0.4,'row':8,'col':10,'irr':4.0,'irc':0.3,'icc':1.5}]

    counts=1000
    im = gmix2image(gd, dims, counts=counts)

    if add_noise:
        skysig=0.1*im.max()
        print 'image max:',im.max()
        print 'skysig   :',skysig
        im += skysig*randu(im.size).reshape(dims[0],dims[1])

    im_min = im.min()
    if im_min < 0:
        sky=-im_min
    else:
        sky=im_min
    sky += 1
    im += sky

    print 'sky:',sky

    maxiter=500
    verbose=False
    gm = gmix_image.GMixEM(im,gd,
                         sky=sky,
                         counts=counts,
                         maxiter=maxiter,
                         tol=tol,
                         verbose=verbose)
    gm.write()
    print 'truth'
    for i,d in enumerate(gd):
        print '%i'% i,
        for k in ['p','row','col','irr','irc','icc']:
            print '%s: %9.6lf' % (k,d[k]),
        print


def test_psf_colocate(add_noise=False, npsf=1):
    print '\nnoise:',add_noise
    tol=1.e-6
    maxiter=2000
    verbose=False
    counts=1000

    dims=[21,21]
    cen=[(dims[0]-1)/2., (dims[1]-1)/2.]
    gd = [{'p':0.4,'row':cen[0],'col':cen[1],'irr':2.5,'irc':0.1,'icc':3.1},
          {'p':0.6,'row':cen[0],'col':cen[1],'irr':1.7,'irc':0.3,'icc':1.5}]

    if npsf==3:
        gpsf = [{'p':0.8,'irr':1.2,'irc':0.2,'icc':1.0},
                {'p':0.4,'irr':2.0,'irc':0.1,'icc':1.5},
                {'p':0.1,'irr':4.0,'irc':0.4,'icc':3.2}]
    elif npsf==2:
        gpsf = [{'p':0.8,'irr':1.2,'irc':0.2,'icc':1.0},
                {'p':0.2,'irr':2.0,'irc':0.1,'icc':1.5}]
    else:
        gpsf = [{'p':1.0,'irr':1.0,'irc':0.2,'icc':1.0}]

    im_prepsf = gmix2image(gd,dims,counts=counts)
    im = gmix2image_psf(gd,gpsf,dims,counts=counts)

    if add_noise:
        skysig=0.05*im.max()
        print 'image max:',im.max()
        print 'skysig   :',skysig
        im += skysig*randu(im.size).reshape(dims[0],dims[1])
    else:
        skysig=None

    # must have non-zero sky
    im_nosky = im.copy()

    sky=1
    im_min=im.min()
    if im_min < 0:
        im += (sky-im_min)

    guess=copy.deepcopy(gd)
    for g in guess:
        g['row'] += 0.2*randu()
        g['col'] += 0.2*randu()
        g['irr'] += 0.2*randu()
        g['irc'] += 0.2*randu()
        g['icc'] += 0.2*randu()
    print guess
    post_counts=im.sum()
    gm = gmix_image.GMixEM(im,guess,
                         sky=sky,
                         counts=post_counts,
                         maxiter=maxiter,
                         tol=tol,
                         psf=gpsf,
                         verbose=verbose)
    gm.write()
    print 'truth'
    for i,d in enumerate(gd):
        print '%i'% i,
        for k in ['p','row','col','irr','irc','icc']:
            print '%s: %9.6lf' % (k,d[k]),
        print

    if gm.flags != 0:
        raise ValueError("halting")

    if have_images:
        pars = gm.pars

        im_fit = gmix2image(pars,dims,counts=counts)
        im_fit_conv = gmix2image_psf(pars,gpsf,dims,counts=counts)

        images.compare_images(im_prepsf, im_fit)

        # im_nosky includes noise
        images.compare_images(im_nosky, im_fit_conv, skysig=skysig)

        mean_moms = gmix_image.total_moms(gd)
        fit_mean_moms = gmix_image.total_moms(pars)
        psf_mean_moms = gmix_image.total_moms(gpsf)

        print 'psf total moms:', \
            psf_mean_moms['irr'],psf_mean_moms['irc'],psf_mean_moms['icc']
        print 'input total moms:    ', \
            mean_moms['irr'],mean_moms['irc'],mean_moms['icc']
        print 'estimated total moms:', \
            fit_mean_moms['irr'],fit_mean_moms['irc'],fit_mean_moms['icc']

def test_psf(add_noise=False, npsf=1):
    print '\nnoise:',add_noise
    tol=1.e-8
    maxiter=2000
    verbose=False
    counts=1000.

    dims=[31,31]
    gd = [{'p':0.4,'row':10,'col':10,'irr':2.5,'irc':0.1,'icc':3.1},
          {'p':0.6,'row':15,'col':17,'irr':1.7,'irc':0.3,'icc':1.5}]
    im_prepsf = gmix_image.gmix2image(gd,dims)

    #gd = [{'p':0.4,'row':15,'col':15,'irr':2.5,'irc':0.1,'icc':3.1}]

    if npsf==3:
        gpsf = [{'p':0.8,'irr':1.2,'irc':0.2,'icc':1.0},
                {'p':0.4,'irr':2.0,'irc':0.1,'icc':1.5},
                {'p':0.1,'irr':4.0,'irc':0.4,'icc':3.2}]
    elif npsf==2:
        gpsf = [{'p':0.8,'irr':1.2,'irc':0.2,'icc':1.0},
                {'p':0.2,'irr':2.0,'irc':0.1,'icc':1.5}]
    else:
        gpsf = [{'p':1.0,'irr':1.0,'irc':0.2,'icc':1.0}]

    im = gmix_image.gmix2image_psf(gd,gpsf,dims,counts=counts)

    if add_noise:
        skysig=0.05*im.max()
        print 'image max:',im.max()
        print 'skysig   :',skysig
        im += skysig*randu(im.size).reshape(dims[0],dims[1])


    # must have non-zero sky
    im -= im.min()
    sky = 0.01*im.max()
    im += sky

    guess=copy.deepcopy(gd)
    for g in guess:
        g['row'] += 0.5*randu()
        g['col'] += 0.5*randu()
        g['irr'] += 0.5*randu()
        g['icc'] += 0.5*randu()
    print guess
    counts=im.sum()
    gm = gmix_image.GMixEM(im,guess,
                         sky=sky,
                         counts=counts,
                         maxiter=maxiter,
                         tol=tol,
                         psf=gpsf,
                         verbose=verbose)
    gm.write()
    print 'truth'
    for i,d in enumerate(gd):
        print '%i'% i,
        for k in ['p','row','col','irr','irc','icc']:
            print '%s: %9.6lf' % (k,d[k]),
        print

    if gm.flags != 0:
        raise ValueError("halting")

    if have_images:
        pars = gm.pars
        im_fit = gmix_image.gmix2image(pars,dims)

        images.compare_images(im_prepsf, im_fit)

def test_fit_dev_by_ellip(sigma, method='lm'):
    """
    Fixed sigma, different ellip
    """
    import biggles
    from scipy.optimize import leastsq
    from fimage import model_image
    numpy.random.seed(35)

    ngauss=4
    nsig=15
    npars=2*ngauss+4
    nsigma_vals=20


    data=numpy.zeros(nsigma_vals,dtype=[('sigma','f8'),('pars','f8',npars)])
    sigvals = numpy.linspace(1.5,5.0,nsigma_vals)
    for isigma,sigma in enumerate(sigvals):
        print '-'*70
        print 'sigma:',sigma
        T = 2*sigma**2

        Irr = sigma**2
        Irc = 0.0
        Icc = sigma**2
        dim = int(2*nsig*sigma)
        if (dim % 2) == 0:
            dim += 1
        dims=array([dim,dim])
        cen=(dims-1)/2.
        cov=[Irr,Irc,Icc]
        im = model_image('dev',dims,cen,cov,nsub=16)


        flags=9999
        while flags != 0:
            stderr.write('.')
            if ngauss == 3:
                p0 = [cen[0],cen[1],Irr,Irc,Icc, 0.4,0.07,0.55, 0.2,3.8]
            elif ngauss==4:
                # at sigma==3 pixelization, expect f vals of 
                #   0.044, 0.22, 6.0
                # but always start on the *inside* of the expected
                p0 = array([cen[0],cen[1],Irr,Irc,Icc, 
                            .22, .35, .25, .15, 
                            .1, .25, 4.])
                p0[5] += 0.1*(randu()-0.5)
                p0[6] += 0.1*(randu()-0.5)
                p0[7] += 0.1*(randu()-0.5)
                p0[8] += 0.1*(randu()-0.5)

                p0[9] += 0.1*(randu()-0.5)
                p0[10] += 0.1*(randu()-0.5)
                p0[11] += 2*(randu()-0.5)

            else:
                raise ValueError("implement guess ngauss > 4")

            verbose=False
            gf=gmix_fit.GMixFitCoellip(im, p0, method=method,verbose=verbose)
            flags = gf.flags

        stderr.write('\n')
        print 'numiter:',gf.numiter
        for i in xrange(len(gf.popt)):
            print '%.6g' % gf.popt[i]

        if gf.flags != 0:
            raise RuntimeError("failed")

        data['sigma'][isigma] = sigma
        data['pars'][isigma,:] = gf.popt

    # plot the last one
    gmix = gmix_fit.pars2gmix_coellip(gf.popt)
    model = gmix2image(gmix,im.shape)
    images.compare_images(im,model)

    biggles.configure('fontsize_min', 1.0)
    biggles.configure('linewidth',1.0) # frame only
    nrows=3
    ncols=4
    tab=biggles.Table(nrows,ncols)
    for par in xrange(npars):
        plt=biggles.FramedPlot()
        plt.add(biggles.Curve(data['sigma'],data['pars'][:,par]))
        plt.xlabel = r'$\sigma$'
        plt.ylabel = 'p%d' % par
        tab[par//ncols,par%ncols] = plt

    tab.show()


def test_fit_dev_bysigma(method='lm'):
    """
    Round object as a function of sigma
    """
    import biggles
    from scipy.optimize import leastsq
    from fimage import model_image
    numpy.random.seed(35)

    ngauss=4
    nsig=15
    npars=2*ngauss+4
    nsigma_vals=20

    f='test-opt-dev-bysigma.rec'
    pngf=f.replace('.rec','.png')
    if not os.path.exists(f):
        data=numpy.zeros(nsigma_vals,dtype=[('sigma','f8'),('pars','f8',npars)])
        sigvals = numpy.linspace(1.5,5.0,nsigma_vals)
        for isigma,sigma in enumerate(sigvals):
            print '-'*70
            print 'sigma:',sigma
            T = 2*sigma**2

            Irr = sigma**2
            Irc = 0.0
            Icc = sigma**2
            dim = int(2*nsig*sigma)
            if (dim % 2) == 0:
                dim += 1
            dims=array([dim,dim])
            cen=(dims-1)/2.
            cov=[Irr,Irc,Icc]
            im = model_image('dev',dims,cen,cov,nsub=16)

            flags=9999
            while flags != 0:
                stderr.write('.')
                if ngauss == 3:
                    p0 = [cen[0],cen[1],Irr,Irc,Icc, 0.4,0.07,0.55, 0.2,3.8]
                elif ngauss==4:
                    # at sigma==3 pixelization, expect f vals of 
                    #   0.044, 0.22, 6.0
                    # but always start on the *inside* of the expected
                    p0 = array([cen[0],cen[1],Irr,Irc,Icc, 
                                .22, .35, .25, .15, 
                                .1, .25, 4.])
                    p0[5] += 0.1*(randu()-0.5)
                    p0[6] += 0.1*(randu()-0.5)
                    p0[7] += 0.1*(randu()-0.5)
                    p0[8] += 0.1*(randu()-0.5)

                    p0[9] += 0.1*(randu()-0.5)
                    p0[10] += 0.1*(randu()-0.5)
                    p0[11] += 2*(randu()-0.5)

                else:
                    raise ValueError("implement guess ngauss > 4")

                verbose=False
                gf=gmix_fit.GMixFitCoellip(im, p0, method=method,verbose=verbose)
                flags = gf.flags

            print 'numiter:',gf.numiter
            for i in xrange(len(gf.popt)):
                print '%.6g' % gf.popt[i]

            if gf.flags != 0:
                raise RuntimeError("failed")

            data['sigma'][isigma] = sigma
            data['pars'][isigma,:] = gf.popt

        # plot the last one
        gmix = gmix_fit.pars2gmix_coellip(gf.popt)
        model = gmix2image(gmix,im.shape)
        images.compare_images(im,model)
    else:
        data=eu.io.read(f)

    biggles.configure('fontsize_min', 1.0)
    biggles.configure('linewidth',1.0) # frame only
    nrows=3
    ncols=4
    tab=biggles.Table(nrows,ncols)
    for par in xrange(npars):
        plt=biggles.FramedPlot()
        plt.add(biggles.Curve(data['sigma'],data['pars'][:,par]))
        plt.add(biggles.Points(data['sigma'],data['pars'][:,par],
                               type='filled circle'))
        plt.xlabel = r'$\sigma$'
        plt.ylabel = 'p%d' % par
        tab[par//ncols,par%ncols] = plt

    tab.show()
    tab.write_img(1024,1024,pngf)

def test_fit_1gauss_fix(imove, use_jacob=True):

    import images
    numpy.random.seed(45)

    T1=3.0
    nsig=5
    dim = int(nsig*T1)
    if (dim % 2) == 0:
        dim += 1
    dims=array([dim,dim])
    cen=(dims-1.)/2.

    theta=23.7*numpy.pi/180.
    eta=-0.7
    ellip=(1+tanh(eta))/2
    print >>stderr,'ellip:',ellip
    print >>stderr,'e1:',ellip*cos(2*theta)
    print >>stderr,'e2:',ellip*sin(2*theta)

    pars=array([cen[0],cen[1],eta,theta,1.,T1])
    print >>stderr,'pars'
    gmix = gmix_fit.pars2gmix_coellip(pars,ptype='eta')

    im=gmix2image(gmix,dims)
    #images.multiview(im)
    
    p0=pars.copy()
    if imove == 0:
        p0[0] += 1*(randu()-0.5)  # cen0
    elif imove == 1:
        p0[1] += 1*(randu()-0.5)  # cen1
    elif imove == 2:
        p0[2] += 1*(randu()-0.5)  # eta
    elif imove == 3:
        p0[3] += 1*(randu()-0.5)   # theta radians
    elif imove == 4:
        p0[4] += 0.2*(randu()-0.5)  # p
    elif imove == 5:
        p0[5] += 1*(randu()-0.5)   # T
    print_pars(pars,front='pars:  ')
    print_pars(p0,  front='guess: ')

    gf=gmix_fit.GMixFitCoellipFix(im, p0, imove, ptype='eta',
                                  use_jacob=use_jacob,
                                  verbose=True)

    print 'numiter:',gf.numiter
    print gf.popt
    print gf.pcov

def test_fit_1gauss_psf_fix(imove, use_jacob=True, seed=45):
    from fimage import ellip2mom
    import images
    numpy.random.seed(seed)

    Tpsf = 2.0
    epsf = 0.2
    theta_psf = 80.0
    cov_psf = ellip2mom(Tpsf, e=epsf, theta=theta_psf)
    psf=[{'p':1.0, 
          'irr':cov_psf[0], 
          'irc':cov_psf[1], 
          'icc':cov_psf[2]}]

    T=3.0

    nsig=5
    dim = int(nsig*T)
    if (dim % 2) == 0:
        dim += 1
    dims=array([dim,dim])
    cen=(dims-1.)/2.

    theta=23.7*numpy.pi/180.
    eta=-0.7
    ellip=(1+tanh(eta))/2
    print >>stderr,'ellip:',ellip
    print >>stderr,'e1:',ellip*cos(2*theta)
    print >>stderr,'e2:',ellip*sin(2*theta)

    pars=array([cen[0],cen[1],eta,theta,1.,T])
    print >>stderr,'pars'
    gmix = gmix_fit.pars2gmix_coellip(pars,ptype='eta')

    im=gmix2image(gmix,dims,psf=psf)
    #images.multiview(im)
    
    p0=pars.copy()
    if imove == 0:
        p0[0] += 1*(randu()-0.5)  # cen0
    elif imove == 1:
        p0[1] += 1*(randu()-0.5)  # cen1
    elif imove == 2:
        p0[2] += 1*(randu()-0.5)  # eta
    elif imove == 3:
        p0[3] += 1*(randu()-0.5)   # theta radians
    elif imove == 4:
        p0[4] += 0.2*(randu()-0.5)  # p
    elif imove == 5:
        p0[5] += 1*(randu()-0.5)   # T
    print_pars(pars,front='pars:  ')
    print_pars(p0,  front='guess: ')

    gf=gmix_fit.GMixFitCoellipFix(im, p0, imove, 
                                  psf=psf,
                                  ptype='eta',
                                  use_jacob=use_jacob,
                                  verbose=True)

    print 'numiter:',gf.numiter
    print gf.popt
    print gf.pcov



def test_fit_1gauss():
    import images
    numpy.random.seed(35)

    T1=3.0
    nsig=5
    dim = int(nsig*T1)
    if (dim % 2) == 0:
        dim += 1
    dims=array([dim,dim])
    cen=(dims-1.)/2.

    theta=23.7*numpy.pi/180.
    eta=-0.7
    ellip=(1+tanh(eta))/2
    print >>stderr,'ellip:',ellip
    pars=array([cen[0],cen[1],eta,theta,1.,T1])
    print >>stderr,'pars'
    gmix = gmix_fit.pars2gmix_coellip(pars,ptype='eta')

    im=gmix2image(gmix,dims)
    #images.multiview(im)
    
    p0=pars.copy()
    p0[0] += 1*(randu()-0.5)  # cen0
    p0[1] += 1*(randu()-0.5)  # cen1
    p0[2] += 0.2*(randu()-0.5)  # eta
    p0[3] += 0.5*(randu()-0.5)   # theta radians
    p0[4] += 0.1*(randu()-0.5)  # p
    p0[5] += 1*(randu()-0.5)   # T
    print_pars(pars,front='pars:  ')
    print_pars(p0,  front='guess: ')

    gf=gmix_fit.GMixFitCoellip(im, p0, ptype='eta',verbose=True)

    print 'numiter:',gf.numiter
    print gf.popt


def test_fit_2gauss():
    import images

    T1=3.0
    T2=6.0
    nsig=5
    dim = int(nsig*T2)
    if (dim % 2) == 0:
        dim += 1
    dims=array([dim,dim])
    cen=(dims-1)/2.

    theta=23.7*numpy.pi/180.
    eta=-0.7
    ellip=(1+tanh(eta))/2
    p1=0.4
    p2=0.6
    print >>stderr,'ellip:',ellip
    pars=array([cen[0],cen[1],eta,theta,p1,p2,T1,T2])
    print >>stderr,'pars'

    gmix = gmix_fit.pars2gmix_coellip(pars,ptype='eta')
    im=gmix2image(gmix,dims)
    
    p0=pars.copy()
    p0[0] += 0.02*(randu()-0.5)  # cen0
    p0[1] += 0.02*(randu()-0.5)  # cen1
    p0[2] = -0.5 + 0.02*(randu()-0.5)  # eta
    p0[3] = 15.0*numpy.pi/180. + 0.2*(randu()-0.5)   # theta radians
    p0[4] = 0.5  # p
    p0[5] = 0.5  # p
    p0[6] = 2.0 + 0.5*(randu()-0.5)   # T
    p0[7] = 3.0 + 0.5*(randu()-0.5)   # T
    print_pars(pars,front='pars:  ')
    print_pars(p0,  front='guess: ')
    gf=gmix_fit.GMixFitCoellip(im, p0, ptype='eta',verbose=True)

    print 'numiter:',gf.numiter
    print gf.popt
    for i in xrange(len(pars)):
        print '%10.6f %10.6f' % (pars[i],gf.popt[i])

def test_fit_exp_eta():
    import biggles
    from scipy.optimize import leastsq
    from fimage import model_image
    numpy.random.seed(35)

    nsig=7
    ngauss=3
    npars=2*ngauss+4
    nsigma=20
    data=numpy.zeros(nsigma,dtype=[('sigma','f8'),('pars','f8',npars)])
    sigvals = numpy.linspace(1.5,5.0,nsigma)
    for isigma,sigma in enumerate(sigvals):
        print '-'*70
        T = 2*sigma**2
        e = 0.3
        eta = ellip2eta(e)

        #theta = randu()*360.*pi/180.
        theta = 23.7*pi/180.
        e1 = e*cos(2*theta)
        e2 = e*sin(2*theta)

        Irc = e2*T/2.
        Icc = (1+e1)*T/2.
        Irr = (1-e1)*T/2.
        sigma = sqrt( (Irr+Icc)/2. ) 
        dim = int(2*nsig*sigma)
        if (dim % 2) == 0:
            dim += 1
        dims=array([dim,dim])
        cen=(dims-1)/2.
        cov=[Irr,Irc,Icc]
        im = model_image('exp',dims,cen,cov,nsub=16)

        ngauss=3
        p0 = [cen[0],# + 0.1*(randu()-0.5),
              cen[1],# + 0.1*(randu()-0.5),
              eta,# + 0.2*(randu()-0.5), 
              theta,# + 10.*pi/180.*(randu()-0.5),
              0.2,
              0.5,
              0.3,
              T,
              0.05*T,
              3.8*T]

        gf=gmix_fit.GMixFitCoellip(im, p0, ptype='eta', verbose=True)
        if gf.flags != 0:
            raise RuntimeError("failed")
        print 'numiter:',gf.numiter
        if gf.flags != 0:
            stop
        #pcov = gf.pcov
        #err = sqrt(diag(pcov))
        for i in xrange(len(gf.popt)):
            #print '%.6g %.6g' % (gf.popt[i],err[i])
            print '%.6g %.6g' % (gf.popt[i],gf.perr[i])
        data['sigma'][isigma] = sigma
        data['pars'][isigma,:] = gf.popt

        tvals = gf.popt[4+ngauss:]
        tmax=tvals.max()
        print 't ratios:',tvals/tmax
        print 'p values:',gf.popt[4:4+ngauss]

    # plot the last one
    gmix = gmix_fit.pars2gmix_coellip_eta(gf.popt)
    model = gmix2image(gmix,im.shape)
    images.compare_images(im,model)

    biggles.configure('fontsize_min', 1.0)
    biggles.configure('linewidth',1.0) # frame only
    nrows=3
    ncols=4
    tab=biggles.Table(nrows,ncols)
    for par in xrange(npars):
        plt=biggles.FramedPlot()
        plt.add(biggles.Curve(data['sigma'],data['pars'][:,par]))
        plt.xlabel = r'$\sigma$'
        plt.ylabel = 'p%d' % par
        tab[par//ncols,par%ncols] = plt

    tab.show()



def test_fit_exp_cov(method='lm'):
    import biggles
    from scipy.optimize import curve_fit, leastsq
    from fimage import model_image
    numpy.random.seed(35)

    nsig=7
    ngauss=3
    npars=2*ngauss+4
    nsigma=20
    data=numpy.zeros(nsigma,dtype=[('sigma','f8'),('pars','f8',npars)])
    sigvals = numpy.linspace(1.5,5.0,nsigma)
    for isigma,sigma in enumerate(sigvals):
        print '-'*70
        T = 2*sigma**2
        #T = 2*8.
        e = 0.3
        theta = randu()*360.
        e1 = e*cos(2*theta*numpy.pi/180.0)
        e2 = e*sin(2*theta*numpy.pi/180.0)

        Irc = e2*T/2.0
        Icc = (1+e1)*T/2.0
        Irr = (1-e1)*T/2.0
        sigma = sqrt( (Irr+Icc)/2. ) 
        dim = int(2*nsig*sigma)
        if (dim % 2) == 0:
            dim += 1
        dims=array([dim,dim])
        cen=(dims-1)/2.
        cov=[Irr,Irc,Icc]
        im = model_image('exp',dims,cen,cov,nsub=16)

        ngauss=3
        p0 = [cen[0],cen[1],Irr,Irc,Icc, 0.4,0.07,0.55,0.2,3.8]
        #p0 = [cen[0],cen[1],Irr,Irc,Icc,
        #      .9,.8,.6,0.25,7.]

        gf=gmix_fit.GMixFitCoellip(im, p0, ptype='cov',
                                   method=method,verbose=True)
        if gf.flags != 0:
            raise RuntimeError("failed")
        print 'numiter:',gf.numiter
        if gf.flags != 0:
            stop
        #pcov = gf.pcov
        #err = sqrt(diag(pcov))
        for i in xrange(len(gf.popt)):
            #print '%.6g %.6g' % (gf.popt[i],err[i])
            print '%.6g' % gf.popt[i]
        data['sigma'][isigma] = sigma
        data['pars'][isigma,:] = gf.popt
    # plot the last one
    gmix = gmix_fit.pars2gmix_coellip(gf.popt)
    model = gmix2image(gmix,im.shape)
    images.compare_images(im,model)

    biggles.configure('fontsize_min', 1.0)
    biggles.configure('linewidth',1.0) # frame only
    nrows=3
    ncols=4
    tab=biggles.Table(nrows,ncols)
    for par in xrange(npars):
        plt=biggles.FramedPlot()
        plt.add(biggles.Curve(data['sigma'],data['pars'][:,par]))
        plt.xlabel = r'$\sigma$'
        plt.ylabel = 'p%d' % par
        tab[par//ncols,par%ncols] = plt

    tab.show()

def test_gmix_exp():
    import admom
    from fimage import model_image
    
    numpy.random.seed(35)

    ngauss=3
    nsig=7.
    #dims=[41,41]
    #cen=[(dims[0]-1)/2., (dims[1]-1)/2.]
    #cov=[10.5,0.0,10.5]

    T = 2*3
    e = 0.3
    theta = randu()*360.
    e1 = e*cos(2*theta*numpy.pi/180.0)
    e2 = e*sin(2*theta*numpy.pi/180.0)

    Irc = e2*T/2.0
    #Icc = (1+e1)*T/2.0
    #Irr = (1-e1)*T/2.0
    Icc = (1+e1)*T/2.0
    Irr = (1-e1)*T/2.0

    #T = 2.*3.0
    sigma = sqrt(T/2.)
    dim = int(2*nsig*sigma)
    if (dim % 2) == 0:
        dim += 1
    dims=array([dim,dim])
    print 'dims:',dims
    cen=(dims-1)/2.
    #cov = [T/2.,0.0,T/2.]
    cov = [Irr,Irc,Icc]

    # need order='c' since we will use in in C code!
    im = model_image('exp',dims,cen,cov,order='c')

    sky = 0.01*im.max()
    im_fakesky = im + sky

    ntry=0
    max_try = 10
    flags=9999
    while flags != 0 and ntry < max_try:
        stderr.write('.')
        guess = get_exp_guess(cen,cov,ngauss)

        gm=gmix_image.GMixEM(im_fakesky, guess, sky=sky, maxiter=5000)
        flags = gm.flags

        ntry += 1
    if ntry == max_try:
        raise ValueError("too many tries")

    gmix_image.gmix_print(gm.pars)
    stderr.write('\n')
    model = gmix2image(gm.pars,im.shape)
    images.compare_images(im, model)

def get_exp_guess(cen,cov,ngauss):
    guess=[]
    for i in xrange(ngauss):
        g= {'p':1./ngauss,
            'row':cen[0] + 0.1*(randu()-0.5),
            'col':cen[1] + 0.1*(randu()-0.5),
            'irr':cov[0] + 0.5*(randu()-0.5),
            'irc':cov[1] + 0.5*(randu()-0.5),
            'icc':cov[2] + 0.5*(randu()-0.5)}
        guess.append(g)
    return guess



if __name__ == "__main__":
    test(add_noise=False)
    test(add_noise=True)
    test_psf()
