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

def gmix2image(gmix, dims, psf=None, coellip=False):
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

    if isinstance(gmix[0], dict):
        return _gmix2imag_lod(gmix, dims, psf=psf)
    else:
        return _gmix2image_pars(gmix, dims, psf=psf, coellip=coellip)


def _gmix2image_lod(gmix, dims, psf=None):
    pars = gmix2pars(gauss_list)
    psf_pars = None
    if psf is not None:
        if not isinstance(psf[0],dict):
            raise ValueError("if gmix is a list of dicts, psf must be "
                             "also")
        psf_pars = gmix2pars(psf)

    im = zeros(dims)
    _render.fill_model(im, pars, psf_pars, None)
    return im

def _gmix2image_pars(pars, dims, psf=None, coellip=False):

    obj_pars = numpy.array(pars, dtype='f8')

    if psf is not None:
        psf_pars = numpy.array(pars, dtype='f8')

    im = zeros(dims)
    if coellip:
        if ( (len(obj_pars)-4) % 2 ) != 0:
            raise ValueError("object pars must have size 2*ngauss+4 "
                             "for coellip")
        if ( (len(psf_pars)-4) % 2 ) != 0:
            raise ValueError("psf pars must have size 2*ngauss+4 "
                             "for coellip")

        _render.fill_model_coellip(im, obj_pars, psf_pars, None)
    else:
        if ( len(obj_pars) % 6 ) != 0:
            raise ValueError("object pars must have size 6*ngauss "
                             "for coellip")
        if ( len(psf_pars) % 6 ) != 0:
            raise ValueError("psf pars must have size 6*ngauss "
                             "for coellip")

       
        _render.fill_model(im, obj_pars, psf_pars, None)
    return im


