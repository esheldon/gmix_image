"""
See docs for
    gmix_image.gmix_em
        Gaussian mixtures using Expectation Maximization
    gmix_image.gmix_fit
        Gaussian mixtures using a levenberg marquardt algorithm.
"""
from sys import stderr

from . import gmix_em
from . import gmix_fit
from . import util

from . import render
from .render import gmix2image

from . import priors

from . import gmix
from .gmix import GMix
from .gmix import GMixCoellip
from .gmix import GMixExp
from .gmix import GMixDev
from .gmix import GMixTurb

from .gmix import GMIX_FULL
from .gmix import GMIX_COELLIP
from .gmix import GMIX_EXP
from .gmix import GMIX_DEV
from .gmix import GMIX_TURB

from .gmix import gmix2pars


from .util import pars2gmix, total_moms,  gmix_print

from .gmix_mcmc import MixMC

from .gmix_em import GMixEM, GMixEMBoot, GMixEMPSF
from .gmix_em import GMIXEM_ERROR_NEGATIVE_DET
from .gmix_em import GMIXEM_ERROR_MAXIT
from .gmix_em import GMIXEM_ERROR_NEGATIVE_DET_COCENTER
from .gmix_em import GMIXEM_ERROR_ADMOM_FAILED
from .gmix_em import gmix2image_em
from .gmix_em import ogrid_image

from .gmix_fit import GMixFitCoellip

from .gmix_fit import get_ngauss_coellip

from .gmix_fit import ellip2eta
from .gmix_fit import eta2ellip
from .gmix_fit import print_pars
from .gmix_fit import GMIXFIT_MAXITER
from .gmix_fit import GMIXFIT_SINGULAR_MATRIX
from .gmix_fit import GMIXFIT_NEG_COV_EIG
from .gmix_fit import GMIXFIT_NEG_COV_DIAG
from .gmix_fit import GMIXFIT_NEG_MCOV_DIAG
from .gmix_fit import GMIXFIT_MCOV_NOTPOSDEF
from .gmix_fit import GMIXFIT_CALLS_NOT_CHANGING
from .gmix_fit import GMIXFIT_LOW_S2N
from . import test

_flagmap_em=\
    {GMIXEM_ERROR_NEGATIVE_DET:            'GMIXEM_ERROR_NEGATIVE_DET',
     'GMIXEM_ERROR_NEGATIVE_DET':           GMIXEM_ERROR_NEGATIVE_DET,
     GMIXEM_ERROR_MAXIT:                   'GMIXEM_ERROR_MAXIT',
     'GMIXEM_ERROR_MAXIT':                  GMIXEM_ERROR_MAXIT,
     GMIXEM_ERROR_NEGATIVE_DET_COCENTER:   'GMIXEM_ERROR_NEGATIVE_DET_COCENTER',
     'GMIXEM_ERROR_NEGATIVE_DET_COCENTER':  GMIXEM_ERROR_NEGATIVE_DET_COCENTER}

_flagmap_fit=\
    {GMIXFIT_MAXITER:              'GMIXFIT_MAXITER',
     'GMIXFIT_MAXITER':             GMIXFIT_MAXITER,
     GMIXFIT_SINGULAR_MATRIX:      'GMIXFIT_SINGULAR_MATRIX',
     'GMIXFIT_SINGULAR_MATRIX':     GMIXFIT_SINGULAR_MATRIX,
     GMIXFIT_NEG_COV_EIG:          'GMIXFIT_NEG_COV_EIG',
     'GMIXFIT_NEG_COV_EIG':         GMIXFIT_NEG_COV_EIG,
     GMIXFIT_NEG_COV_DIAG:         'GMIXFIT_NEG_COV_DIAG',
     'GMIXFIT_NEG_COV_DIAG':        GMIXFIT_NEG_COV_DIAG,
     GMIXFIT_NEG_MCOV_DIAG:        'GMIXFIT_NEG_MCOV_DIAG',
     'GMIXFIT_NEG_MCOV_DIAG':       GMIXFIT_NEG_MCOV_DIAG,
     GMIXFIT_MCOV_NOTPOSDEF:       'GMIXFIT_MCOV_NOTPOSDEF',
     'GMIXFIT_MCOV_NOTPOSDEF':      GMIXFIT_MCOV_NOTPOSDEF,
     GMIXFIT_CALLS_NOT_CHANGING:   'GMIXFIT_CALLS_NOT_CHANGING',
     'GMIXFIT_CALLS_NOT_CHANGING':  GMIXFIT_CALLS_NOT_CHANGING,
     GMIXFIT_LOW_S2N:              'GMIXFIT_LOW_S2N',
     'GMIXFIT_LOW_S2N':             GMIXFIT_LOW_S2N}

def printflags(type, flags):
    if type == 'em':
        fmap =_flagmap_em
    else:
        fmap =_flagmap_fit
    for i in xrange(31):
        flag=2**i
        if flag in fmap:
            if (flags & flag) != 0:
                stderr.write("%d %s\n" % (i,flagname(type,flag)))

def flagname(type, flag):
    if type == 'em':
        name =_flagmap_em.get(flag,None)
    else:
        name =_flagmap_fit.get(flag,None)
    if name is None:
        raise ValueError("unknown %s flag: %s" % (type,flag))
    return name

def flagval(type, name):
    if type == 'em':
        val =_flagmap_em.get(name,None)
    else:
        val =_flagmap_fit.get(name,None)
    if val is None:
        raise ValueError("unknown %s flag name: '%s'" % (type,name))
    return val


