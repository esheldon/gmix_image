import numpy
from numpy import sqrt, array, ogrid, random, exp, zeros
import gmix_image
from gmix_image import ogrid_image, gmix2image, gmix2image_psf
import copy

try:
    import images
    have_images=True
except:
    have_images=False

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
        im += skysig*random.random(im.size).reshape(dims[0],dims[1])

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
    gm = gmix_image.GMix(im,gd,
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
        im += skysig*random.random(im.size).reshape(dims[0],dims[1])
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
        g['row'] += 0.2*random.random()
        g['col'] += 0.2*random.random()
        g['irr'] += 0.2*random.random()
        g['irc'] += 0.2*random.random()
        g['icc'] += 0.2*random.random()
    print guess
    post_counts=im.sum()
    gm = gmix_image.GMix(im,guess,
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
        im += skysig*random.random(im.size).reshape(dims[0],dims[1])


    # must have non-zero sky
    im -= im.min()
    sky = 0.01*im.max()
    im += sky

    guess=copy.deepcopy(gd)
    for g in guess:
        g['row'] += 0.5*random.random()
        g['col'] += 0.5*random.random()
        g['irr'] += 0.5*random.random()
        g['icc'] += 0.5*random.random()
    print guess
    counts=im.sum()
    gm = gmix_image.GMix(im,guess,
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



if __name__ == "__main__":
    test(add_noise=False)
    test(add_noise=True)
    test_psf()
