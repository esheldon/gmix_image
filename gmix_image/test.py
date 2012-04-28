import numpy
from numpy import sqrt, array, ogrid, random, exp, zeros
import gmix_image
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
    #gd = [{'p':0.4,'row':10,'col':10,'irr':2.5,'irc':0.1,'icc':3.1},
    #      {'p':0.6,'row':15,'col':17,'irr':1.7,'irc':0.3,'icc':1.5}]
    gd = [{'p':0.6,'row':15,'col':15,'irr':2.0,'irc':0.0,'icc':2.0},
          {'p':0.4,'row':8,'col':10,'irr':4.0,'irc':0.3,'icc':1.5}]

    im1=ogrid_image('gauss',dims,
                    [gd[0]['row'],gd[0]['col']],
                    [gd[0]['irr'],gd[0]['irc'],gd[0]['icc']],
                    counts=1000*gd[0]['p'])
    im2=ogrid_image('gauss',dims,
                    [gd[1]['row'],gd[1]['col']],
                    [gd[1]['irr'],gd[1]['irc'],gd[1]['icc']],
                    counts=1000*gd[1]['p'])

    im = im1 + im2

    if add_noise:
        skysig=0.05
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

    # must have non-zero sky
    #im -= im.min()
    #sky = 0.01*im.max()
    #im += sky

    print 'sky:',sky

    counts=im.sum()
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

    dims=[31,31]
    #gd = [{'p':0.4,'row':15.,'col':15.,'irr':2.5,'irc':0.1,'icc':3.1},
    #      {'p':0.6,'row':15.,'col':15.,'irr':1.7,'irc':0.3,'icc':1.5}]
    gd = [{'p':0.4,'row':15.,'col':15.,'irr':2.0,'irc':0.0,'icc':2.0},
          {'p':0.6,'row':15.,'col':15.,'irr':1.0,'irc':0.0,'icc':1.0}]
    im_prepsf = gmix_image.gmix2image(gd,dims)

    if npsf==2:
        gpsf = [{'p':0.8,'irr':1.0,'irc':0.2,'icc':1.0},
                {'p':0.2,'irr':2.0,'irc':0.1,'icc':2.0}]
    else:
        gpsf = [{'p':1.0,'irr':1.0,'irc':0.0,'icc':1.0}]

    im1conv = ogrid_image_convolved('gauss',dims,
                                    [gd[0]['row'], gd[0]['col']],
                                    [gd[0]['irr'], 
                                     gd[0]['irc'], 
                                     gd[0]['icc']],
                                    gpsf,
                                    counts=1000*gd[0]['p'])
    if len(gd) == 2:
        im2conv = ogrid_image_convolved('gauss',dims,
                                        [gd[1]['row'], gd[1]['col']],
                                        [gd[1]['irr'], 
                                         gd[1]['irc'], 
                                         gd[1]['icc']],
                                        gpsf,
                                        counts=1000*gd[1]['p'])

    if len(gd) == 2:
        im = im1conv + im2conv
    else:
        im = im1conv


    if add_noise:
        skysig=0.05
        print 'image max:',im.max()
        print 'skysig   :',skysig
        im += skysig*random.random(im.size).reshape(dims[0],dims[1])


    # must have non-zero sky
    im -= im.min()
    sky = 0.01*im.max()
    im += sky

    guess=copy.deepcopy(gd)
    for g in guess:
        g['row'] += 0.2*random.random()
        g['col'] += 0.2*random.random()
        g['irr'] += 0.2*random.random()
        g['irc'] += 0.2*random.random()
        g['icc'] += 0.2*random.random()
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

        #images.multiview(im_prepsf, title='true')
        #images.multiview(im_fit, title='fit')
        images.multiview(im-sky)
        images.compare_images(im_prepsf, im_fit)

        Irr=0.0
        Irc=0.0
        Icc=0.0
        eIrr=0.0
        eIrc=0.0
        eIcc=0.0
        Psum=0.0
        ePsum=0.0
        for i in xrange(len(gd)):
            Psum += gd[i]['p']
            ePsum += pars[i]['p']
            Irr += gd[i]['p']*gd[i]['irr']
            Irc += gd[i]['p']*gd[i]['irc']
            Icc += gd[i]['p']*gd[i]['icc']

            eIrr += pars[i]['p']*pars[i]['irr']
            eIrc += pars[i]['p']*pars[i]['irc']
            eIcc += pars[i]['p']*pars[i]['icc']

        Irr /= Psum
        Irc /= Psum
        Icc /= Psum
        eIrr /= ePsum
        eIrc /= ePsum
        eIcc /= ePsum

        print 'input total moms:    ',Irr,Irc,Icc
        print 'estimated total moms:',eIrr,eIrc,eIcc

def test_psf(add_noise=False, npsf=1):
    print '\nnoise:',add_noise
    tol=1.e-8
    maxiter=2000
    verbose=False

    dims=[31,31]
    gd = [{'p':0.4,'row':10,'col':10,'irr':2.5,'irc':0.1,'icc':3.1},
          {'p':0.6,'row':15,'col':17,'irr':1.7,'irc':0.3,'icc':1.5}]
    im_prepsf = gmix_image.gmix2image(gd,dims)

    #gd = [{'p':0.4,'row':15,'col':15,'irr':2.5,'irc':0.1,'icc':3.1}]

    if npsf==2:
        gpsf = [{'p':0.8,'irr':1.0,'irc':0.2,'icc':1.0},
                {'p':0.2,'irr':2.0,'irc':0.1,'icc':2.0}]
    else:
        gpsf = [{'p':1.0,'irr':1.0,'irc':0.0,'icc':1.0}]

    im1conv = ogrid_image_convolved('gauss',dims,
                                    [gd[0]['row'], gd[0]['col']],
                                    [gd[0]['irr'], 
                                     gd[0]['irc'], 
                                     gd[0]['icc']],
                                    gpsf,
                                    counts=1000*gd[0]['p'])
    if len(gd) == 2:
        im2conv = ogrid_image_convolved('gauss',dims,
                                        [gd[1]['row'], gd[1]['col']],
                                        [gd[1]['irr'], 
                                         gd[1]['irc'], 
                                         gd[1]['icc']],
                                        gpsf,
                                        counts=1000*gd[1]['p'])

    if len(gd) == 2:
        im = im1conv + im2conv
    else:
        im = im1conv


    if add_noise:
        skysig=0.05
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

        #images.multiview(im_prepsf, title='true')
        #images.multiview(im_fit, title='fit')
        images.compare_images(im_prepsf, im_fit)

def ogrid_image_convolved(model, dims, cen, cov, gpsf, counts=1.0):
    imconv = zeros(tuple(dims))
    psum=0.0
    for i in xrange(len(gpsf)):
        p=gpsf[i]['p']
        psum += p

        cov= [cov[0]+gpsf[i]['irr'],
              cov[1]+gpsf[i]['irc'],
              cov[2]+gpsf[i]['icc']]

        imconv += p*ogrid_image('gauss',dims,cen,cov,counts=counts)


    imconv /= psum
    return imconv

def ogrid_image(model, dims, cen, cov, counts=1.0):

    Irr,Irc,Icc = cov
    det = Irr*Icc - Irc**2
    if det == 0.0:
        raise RuntimeError("Determinant is zero")

    Wrr = Irr/det
    Wrc = Irc/det
    Wcc = Icc/det

    # ogrid is so useful
    row,col=ogrid[0:dims[0], 0:dims[1]]

    rm = array(row - cen[0], dtype='f8')
    cm = array(col - cen[1], dtype='f8')

    rr = rm**2*Wcc -2*rm*cm*Wrc + cm**2*Wrr

    model = model.lower()
    if model == 'gauss':
        rr = 0.5*rr
    elif model == 'exp':
        rr = sqrt(rr*3.)
    elif model == 'dev':
        rr = 7.67*( (rr)**(.125) -1 )
    else: 
        raise ValueError("model must be one of gauss, exp, or dev")

    image = exp(-rr)

    image *= counts/image.sum()

    return image

if __name__ == "__main__":
    test(add_noise=False)
    test(add_noise=True)
    test_psf()
