"""
functions
---------
gmix2pars:
    Convert a list-of-dictionaries representation of a gaussian mixture
    to an array

total_moms:
    total moments of a gaussian mixture.

gmix_print:
    print out a gaussian mixture
"""

import numpy
from numpy import zeros, array, where, ogrid, diag, sqrt, isfinite, \
        tanh, arctanh, cos, sin, exp

def gmix2pars(gmix):
    """
    convert a list of dictionaries to an array.

    The packing is [p1,row1,col1,irr1,irc1,icc1,
                    p2,row2,....]
    """
    ngauss=len(gmix)
    pars=zeros(ngauss*6,dtype='f8')
    for i,g in enumerate(gmix):
        beg=i*6
        pars[beg+0] = g['p']
        pars[beg+1] = g['row']
        pars[beg+2] = g['col']
        pars[beg+3] = g['irr']
        pars[beg+4] = g['irc']
        pars[beg+5] = g['icc']

    return pars


def total_moms(gauss_list, psf=None):
    """
    Only makes sense if the centers are the same

    parameters
    ----------
    gauss_list: 
        A gaussian mixture model as a list of dicts.
    psf: optional
        A PSF as a gaussian mixture model.  The result
        will be convolved with the PSF.
    """
    if psf is not None:
        return _total_moms_psf(gauss_list, psf)

    d={'irr':0.0, 'irc':0.0, 'icc':0.0}
    psum=0.0
    for g in gauss_list:
        p=g['p']
        psum += p
        d['irr'] += p*g['irr']
        d['irc'] += p*g['irc']
        d['icc'] += p*g['icc']

    d['irr'] /= psum
    d['irc'] /= psum
    d['icc'] /= psum
    return d

def _total_moms_psf(gauss_list, psf_list):
    """
    Only makes sense if the centers are the same
    """
    d={'irr':0.0, 'irc':0.0, 'icc':0.0}
    psf_totmom = total_moms(psf_list)

    psum=0.0
    for g in gauss_list:
        p=g['p']
        psum += p
        d['irr'] += p*(g['irr'] + psf_totmom['irr'])
        d['irc'] += p*(g['irc'] + psf_totmom['irc'])
        d['icc'] += p*(g['icc'] + psf_totmom['icc'])

    d['irr'] /= psum
    d['irc'] /= psum
    d['icc'] /= psum
    return d

def gmix_print(gmix):
    """
    Print a gaussian mixture.

    The gmix should be in list-of-dicts represenation
    """
    hfmt = ['%10s']*6
    hfmt = ' '.join(hfmt)
    h = hfmt % ('p','row','col','irr','irc','icc')
    print h

    fmt = ['%10.6g']*6
    fmt = ' '.join(fmt)
    for g in gmix:
        print fmt % tuple([g[k] for k in ['p','row','col','irr','irc','icc']])

def pars2gmix(pars, coellip=False):
    """
    Convert a parameter array.  

    if coellip, the packing is
        [cen1,cen2,e1,e2,Tmax,Tfrac2,Tfrac3..,p1,p2,p3...]
    otherwise
        [p1,row1,col1,irr1,irc1,icc1,
         p2,row2,col2,irr2,irc2,icc2,
         ...]
    """

    if coellip:
        return _pars2gmix_coellip(pars)

    ngauss = len(pars)/6
    gmix=[]

    for i in xrange(ngauss):
        beg=i*6
        d={}

        d['p']   = pars[beg+0]
        d['row'] = pars[beg+1]
        d['col'] = pars[beg+2]
        d['irr'] = pars[beg+3]
        d['irc'] = pars[beg+4]
        d['icc'] = pars[beg+5]
        gmix.append(d)

    return gmix




def _pars2gmix_coellip(pars):
    """
    Convert a parameter array.  

    [cen1,cen2,e1,e2,Tmax,Tfrac2,Tfrac3..,p1,p2,p3...]
    """
    ngauss = (len(pars)-4)/2
    gmix=[]

    row=pars[0]
    col=pars[1]
    e1 = pars[2]
    e2 = pars[3]
    Tmax = pars[4]

    for i in xrange(ngauss):
        d={}

        if i == 0:
            T = Tmax
        else:
            Tfrac = pars[4+i]
            T = Tmax*Tfrac

        p = pars[4+ngauss+i]
        
        d['p'] = p
        d['row'] = row
        d['col'] = col
        d['irr'] = (T/2.)*(1-e1)
        d['irc'] = (T/2.)*e2
        d['icc'] = (T/2.)*(1+e1)
        gmix.append(d)

    return gmix



