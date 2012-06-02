import numpy
from numpy import sqrt, array, ogrid, random, exp, zeros, cos, sin, diag
from numpy.random import random
import gmix_image
import gmix_fit
from gmix_image import ogrid_image, gmix2image, gmix2image_psf
import copy

from sys import stderr

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
        im += skysig*random(im.size).reshape(dims[0],dims[1])

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
        im += skysig*random(im.size).reshape(dims[0],dims[1])
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
        g['row'] += 0.2*random()
        g['col'] += 0.2*random()
        g['irr'] += 0.2*random()
        g['irc'] += 0.2*random()
        g['icc'] += 0.2*random()
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
        im += skysig*random(im.size).reshape(dims[0],dims[1])


    # must have non-zero sky
    im -= im.min()
    sky = 0.01*im.max()
    im += sky

    guess=copy.deepcopy(gd)
    for g in guess:
        g['row'] += 0.5*random()
        g['col'] += 0.5*random()
        g['irr'] += 0.5*random()
        g['icc'] += 0.5*random()
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


def test_lm_exp():
    from scipy.optimize import curve_fit, leastsq
    from fimage import model_image
    numpy.random.seed(35)

    nsig=7

    T = 2*3.
    e = 0.3
    theta = random()*360.
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
    im = model_image('exp',dims,cen,cov,nsub=1)

    ngauss=3
    p0 = [cen[0],cen[1],Irr,Irc,Icc,
          .9,.8,.6,0.25,7.]

    gf=gmix_fit.GMixFitCoellip(im, ngauss)
    res=leastsq(gf.ydiff,p0,full_output=1,Dfun=gf.jacob,col_deriv=1)
    popt, pcov0, infodict, errmsg, ier = res
    if ier not in [1,2,3,4]:
        print "failure: %s" % errmsg
    elif pcov0 is None:
        print 'Singular matrix!'
    print 'numiter:',infodict['nfev']
    pcov = gf.scale_cov(popt, pcov0)
    err = sqrt(diag(pcov))
    fmt='%.5g '*len(p0)
    for i in xrange(len(popt)):
        print '%.6g %.6g' % (popt[i],err[i])
    gmix = gmix_fit.pars2gmix_coellip(popt)
    model = gmix2image(gmix,im.shape)
    images.compare_images(im,model)

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
    theta = random()*360.
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
            'row':cen[0] + 0.1*(random()-0.5),
            'col':cen[1] + 0.1*(random()-0.5),
            'irr':cov[0] + 0.5*(random()-0.5),
            'irc':cov[1] + 0.5*(random()-0.5),
            'icc':cov[2] + 0.5*(random()-0.5)}
        guess.append(g)
    return guess



if __name__ == "__main__":
    test(add_noise=False)
    test(add_noise=True)
    test_psf()
