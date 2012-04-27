import numpy
from numpy import sqrt, array, ogrid, random, exp
import gmix_image
import pprint

def test(add_noise=False):
    print '\nnoise:',add_noise
    tol=1.e-6
    dims=[31,31]
    gd = [{'p':0.4,'row':10,'col':10,'irr':2.5,'irc':0.1,'icc':3.1},
          {'p':0.6,'row':15,'col':17,'irr':1.7,'irc':0.3,'icc':1.5}]

    im1=ogrid_image('gauss',dims,
                    [gd[0]['row'],gd[0]['col']],
                    [gd[0]['irr'],gd[0]['irc'],gd[0]['icc']],
                    counts=1000)
    im2=ogrid_image('gauss',dims,
                    [gd[1]['row'],gd[1]['col']],
                    [gd[1]['irr'],gd[1]['irc'],gd[1]['icc']],
                    counts=1000)

    im = gd[0]['p']*im1 + gd[1]['p']*im2

    # must have non-zero sky
    sky=1.0

    if add_noise:
        skysig=0.05
        print 'image max:',im.max()
        print 'skysig   :',skysig
        im += skysig*random.random(im.size).reshape(dims[0],dims[1])
    im += sky

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

    return image

if __name__ == "__main__":
    test(add_noise=False)
    test(add_noise=True)
