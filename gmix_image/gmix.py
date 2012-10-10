import numpy
from numpy import array
import copy
from . import _render
from .util import gmix2pars

GMIX_FULL=0
GMIX_COELLIP=1
GMIX_EXP=2
GMIX_DEV=3
GMIX_TURB=4

def GMixCoellip(pars):
    """
    Generate a co-elliptical gaussian mixture.

    parameters
    ----------
    pars: sequence
        [row,col,e1,e2,Tmax,f2,f3,...,p1,p2,p3...]
    """
    return GMix(pars, type=GMIX_COELLIP)


def GMixExp(pars):
    """
    Generate a gaussian mixture representing an approximate exponential disk.

    parameters
    ----------
    pars: sequence
        [row,col,e1,e2,T,p]
    """
    return GMix(pars, type=GMIX_EXP)

def GMixDev(pars):
    """
    Generate a gaussian mixture representing an approximate devauc profile.

    parameters
    ----------
    pars: sequence
        [row,col,e1,e2,T,p]
    """
    return GMix(pars, type=GMIX_EXP)

def GMixTurb(pars):
    """
    Generate a gaussian mixture representing an approximate turbulent 
    atmospheric PSF.  Can be ellipticial.

    parameters
    ----------
    pars: sequence
        [row,col,e1,e2,T,p]
    """
    return GMix(pars, type=GMIX_EXP)



class GMix(_render.GVec):
    """
    Generate a gaussian mixture from the input parameters.

    parameters
    ----------
    pars: sequence or list of dicts
        A sequence describing the gaussians, as determined
        by the type parameter.  For GMIX_FULL can also be
        a list of dictionaries

    type: int, optional
        GMIX_FULL 
            input is a full gaussian mixture, either represented as
                [p1,row1,row2,irr1,irc1,icc2,...]
            or as a list of dicts with each dict
                p,row,col,irr,irc,icc
            This is the default

        GMIX_COELLIP 
            input is for coelliptical gaussians
                [row,col,e1,e2,Tmax,f2,f3...,p1,p2,p3...]

        GMIX_EXP 
            input specifies an approximate exponential disk with
            parameters
                [row,col,e1,e2,T,p]

        GMIX_DEV 
            input specifies an approximate devauc profile with
            parameters
                [row,col,e1,e2,T,p]

        GMIX_TURB 
            input specifies an approximate turbulent psf with
            parameters.  Can be elliptical.
                [row,col,e1,e2,T,p]


    methods
    -------
    convolve(psf):
        Get a new GMix that is the convolution of the GMix with the input psf
    get_dlist():
        return a list of dicts representing the gaussian mixture
    get_T():
        Return T=sum(p*T_i)/sum(p)
    get_cen():
        Return cen=sum(p*cen_i)/sum(p)
    set_cen(row,col):
        Set all centers to the input.
    get_pars():
        Get a copy of the parameters used to initialize the GMix
    get_type():
        Get a copy of the type of the input parameters
    """
    def __init__(self, pars, type=GMIX_FULL):
        type=int(type)

        if type==GMIX_FULL and isinstance(pars[0], dict):
            pars_array=gmix2pars(pars)
        else:
            pars_array=array(pars,dtype='f8')

        super(GMix,self).__init__(type, pars_array)
        self._pars=pars_array
        self._type=type


    def convolve(self, psf):
        """
        Get a new GMix that is the convolution of the GMix with the input psf

        parameters
        ----------
        psf: GMix object
        """
        if not isinstance(psf, GMix):
            raise ValueError("Can only convolve with another GMix object")

        gmix = GMix(self._pars, type=self._type)
        gmix._convolve_inplace(psf)
        return gmix

    def get_pars(self):
        """
        Get a copy of the parameter array used to initialize the GMix
        """
        return self._pars.copy()

    def get_type(self):
        """
        Get a copy of the type of the input parameters
        """
        return copy.copy(self._type)


    def __repr__(self):
        import pprint
        dlist=self.get_dlist()
        return pprint.pformat(dlist)
