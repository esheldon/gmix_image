"""
functions
---------
gmix2image:
    Create an image from the gaussian input mixture model.  Optionally
    send a PSF, which results in a call to gmix2image_psf.
"""

import numpy
from numpy import zeros

from .util import gmix2pars, check_jacobian
from . import _render
from . import gmix
from .gmix import GMix, togmix

def gmix2image(gmix, dims, psf=None, 
               coellip=False, 
               Tfrac=False, 
               nsub=1, 
               jacobian=None,
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
        Convolve with this gaussian mixture PSf model.
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
    if jacobian is None:
        flags=_render.fill_model(im, gmix, nsub)
    else:
        check_jacobian(jacobian)
        flags=_render.fill_model_jacob(im,
                                       gmix,
                                       jacobian['dudrow'],
                                       jacobian['dudcol'],
                                       jacobian['dvdrow'],
                                       jacobian['dvdcol'],
                                       jacobian['row0'], # coord system center
                                       jacobian['col0'],
                                       nsub)

    if getflags:
        return im, flags
    else:
        return im



