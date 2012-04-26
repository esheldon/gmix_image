import numpy
import _gmix_image


def dotest():
    dims=[31,31]
    gd = [{'p':0.4,'row':10,'col':10,'irr':2.5,'irc':0.1,'icc':3.1},
          {'p':0.6,'row':15,'col':17,'irr':1.7,'irc':0.3,'icc':1.5}]
    gv = _gmix_image.GVec(gd)
    gv.print_n()
    print gv

    im1=ogrid_image('gauss',dims,
                    [gd[0]['row'],gd[0]['col']],
                    [gd[0]['irr'],gd[0]['irc'],gd[0]['icc']],
                    counts=1000)
    im2=ogrid_image('gauss',dims,
                    [gd[1]['row'],gd[1]['col']],
                    [gd[1]['irr'],gd[1]['irc'],gd[1]['icc']],
                    counts=1000)

    # must have non-zero sky
    sky=1.0
    im = sky + gd[0]['p']*im1 + gd[1]['p']*im2

    counts=im.sum()
    maxiter=500
    verbose=1
    gm = _gmix_image.GMix(gv,im,sky,counts,maxiter,verbose)
    gm.print_n()
    print gm


def ogrid_image(model, dims, cen, cov, counts=1.0):

    Irr,Irc,Icc = cov
    det = Irr*Icc - Irc**2
    if det == 0.0:
        raise RuntimeError("Determinant is zero")

    Wrr = Irr/det
    Wrc = Irc/det
    Wcc = Icc/det

    # ogrid is so useful
    row,col=numpy.ogrid[0:dims[0], 0:dims[1]]

    rm = numpy.array(row - cen[0], dtype='f8')
    cm = numpy.array(col - cen[1], dtype='f8')

    rr = rm**2*Wcc -2*rm*cm*Wrc + cm**2*Wrr

    model = model.lower()
    if model == 'gauss':
        rr = 0.5*rr
    elif model == 'exp':
        rr = numpy.sqrt(rr*3.)
    elif model == 'dev':
        rr = 7.67*( (rr)**(.125) -1 )
    else: 
        raise ValueError("model must be one of gauss, exp, or dev")

    image = numpy.exp(-rr)

    return image

if __name__ == "__main__":
    dotest()
