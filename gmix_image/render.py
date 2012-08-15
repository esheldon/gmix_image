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

def gmix2image(gmix, dims, psf=None, coellip=False, getflags=False):
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
        An optional gaussian mixture PSf model.  The models will be convolved
        with this PSF.
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

def _gmix2image_lod(gmix, dims, psf=None):
    pars = gmix2pars(gmix)
    psf_pars = None
    if psf is not None:
        if not isinstance(psf[0],dict):
            raise ValueError("if gmix is a list of dicts, psf must be "
                             "also")
        psf_pars = gmix2pars(psf)

    im = zeros(dims,dtype='f8')
    flags=_render.fill_model(im, pars, psf_pars, None)
    return im, flags

def _gmix2image_pars(pars, dims, psf=None, coellip=False):

    obj_pars = numpy.array(pars, dtype='f8')

    psf_pars=None
    if psf is not None:
        psf_pars = numpy.array(psf, dtype='f8')

    im = zeros(dims,dtype='f8')
    if coellip:
        if ( (len(obj_pars)-4) % 2 ) != 0:
            raise ValueError("object pars must have size 2*ngauss+4 "
                             "for coellip")
        if psf_pars:
            if ( (len(psf_pars)-4) % 2 ) != 0:
                raise ValueError("psf pars must have size 2*ngauss+4 "
                                 "for coellip")

        flags=_render.fill_model_coellip(im, obj_pars, psf_pars, None)
    else:
        if ( len(obj_pars) % 6 ) != 0:
            raise ValueError("object pars must have size 6*ngauss "
                             "for coellip")
        if ( len(psf_pars) % 6 ) != 0:
            raise ValueError("psf pars must have size 6*ngauss "
                             "for coellip")

       
        flags=_render.fill_model(im, obj_pars, psf_pars, None)
    return im, flags


