"""
See docs for
    gmix_image.gmix_em
        Gaussian mixtures using Expectation Maximization
    gmix_image.gmix_fit
        Gaussian mixtures using a levenberg marquardt algorithm.
"""
import gmix_em
import gmix_fit

from gmix_em import GMixEM
from gmix_em import GMIXEM_ERROR_NEGATIVE_DET
from gmix_em import GMIXEM_ERROR_MAXIT
from gmix_em import GMIXEM_ERROR_NEGATIVE_DET_COCENTER
from gmix_em import gmix2image, gmix2image_psf
from gmix_em import ogrid_image, total_moms, total_moms_psf
from gmix_em import gmix_print

from gmix_fit import GMixFitCoellip
from gmix_fit import pars2gmix_coellip
from gmix_fit import ellip2eta
from gmix_fit import eta2ellip
from gmix_fit import print_pars
from gmix_fit import GMIXFIT_MAXITER
from gmix_fit import GMIXFIT_SINGULAR_MATRIX
from gmix_fit import GMIXFIT_NEG_COV_EIG
from gmix_fit import GMIXFIT_NEG_COV_DIAG
from gmix_fit import GMIXFIT_NEG_MCOV_DIAG
from gmix_fit import GMIXFIT_MCOV_NOTPOSDEF
from gmix_fit import GMIXFIT_CALLS_NOT_CHANGING
from gmix_fit import GMIXFIT_LOW_S2N
import test

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
                print i,flagname(type,flag)

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


