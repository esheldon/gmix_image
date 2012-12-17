"""
functions
---------
gmix2image:
    Create an image from the gaussian input mixture model.  Optionally
    send a PSF, which results in a call to gmix2image_psf.
"""

import numpy
from numpy import zeros

from .util import gmix2pars
from . import _render
from . import gmix
from .gmix import GMix, togmix

def gmix2image(gmix, dims, psf=None, 
               coellip=False, 
               Tfrac=False, 
               nsub=1, 
               getflags=False):
    """
    Create an image from the gaussian input mixture model.

    TODO: allow non-coelliptical PSF when using coelliptical
    object

    parameters
    ----------
    gmix:
        The gaussian mixture model.
    dims:
        The dimensions of the result.  This matters since
        the gaussian centers are in this coordinate syste.
    psf: optional
        An optional gaussian mixture PSf model. Must be a generic mixture
    coellip:
        If True, and the input are parameter arrays, then the model
        represents coelliptical gaussians.
    nsub:
        Sub-pixel integration for simulations.  Default 1
    """

    if len(dims)  != 2:
        raise ValueError("dims must be 2 element sequence/array")

    gmix=togmix(gmix, coellip=coellip)
    if psf:
        psf=togmix(psf)
        gmix=gmix.convolve(psf)

    im=zeros(dims,dtype='f8')
    flags=_render.fill_model(im, gmix, nsub)

    if getflags:
        return im, flags
    else:
        return im



def _gmix2image_lod(gmix, dims, psf=None):
    pars = gmix2pars(gmix)
    psf_pars = None
    if psf is not None:
        if not isinstance(psf[0],dict):
            raise ValueError("psf must be list of dicts")
        psf_pars = gmix2pars(psf)

    im = zeros(dims,dtype='f8')
    flags=_render.fill_model_old(im, pars, psf_pars, None)
    return im, flags

def _gmix2image_pars(pars, dims, psf=None, coellip=False):

    obj_pars = numpy.array(pars, dtype='f8')

    psf_pars=None
    if psf is not None:
        if not isinstance(psf[0],dict):
            raise ValueError("psf must be list of dicts")
        psf_pars = gmix2pars(psf)

    im = zeros(dims,dtype='f8')
    if coellip:
        if ( (len(obj_pars)-4) % 2 ) != 0:
            raise ValueError("object pars must have size 2*ngauss+4 "
                             "for coellip")
        flags=_render.fill_model_coellip_old(im, obj_pars, psf_pars, None)
    else:
        if ( len(obj_pars) % 6 ) != 0:
            raise ValueError("object pars must have size 6*ngauss "
                             "for not coellip")
        if ( len(psf_pars) % 6 ) != 0:
            raise ValueError("psf pars must have size 6*ngauss "
                             "for not coellip")

       
        flags=_render.fill_model_old(im, obj_pars, psf_pars, None)
    return im, flags

def gmix2image_old(gmix, dims, psf=None, coellip=False, getflags=False):
    """
    Create an image from the gaussian input mixture model.

    TODO: allow non-coelliptical PSF when using coelliptical
    object

    parameters
    ----------
    gmix:
        The gaussian mixture model as a list of dictionaries.
    dims:
        The dimensions of the result.  This matters since
        the gaussian centers are in this coordinate syste.
    psf: optional
        An optional gaussian mixture PSf model. Must be a generic list of
        dicts.
    coellip:
        If True, and the input are parameter arrays, then the model
        represents coelliptical gaussians.
    """

    if len(dims)  != 2:
        raise ValueError("dims must be 2 element sequence/array")

    if isinstance(gmix[0], dict):
        im, flags = _gmix2image_lod(gmix, dims, psf=psf)
    else:
        im, flags = _gmix2image_pars(gmix, dims, psf=psf, coellip=coellip)


    if getflags:
        return im, flags
    else:
        return im


