import numpy
from numpy import array
import copy
from . import _gvec
from ._gvec import version

GMIX_FULL=0
GMIX_COELLIP=1
GMIX_EXP=2
GMIX_DEV=3
GMIX_TURB=4

class GMix(_gvec.GVec):
    """
    Generate a gaussian mixture vector from the input parameters.

    parameters
    ----------
    type: int
        0 pars are full gaussian mixture 
                [p1,row1,row2,irr1,irc1,icc2,...]
        1 pars are coellip pars
                [row,col,e1,e2,Tmax,f2,f3...,p1,p2,p3...]
        2 pars specify an approximate exponential disk with
            parameters
                [row,col,e1,e2,T,p]
        3 pars specify an approximate devauc profile with
            parameters
                [row,col,e1,e2,T,p]
        4 pars specify an approximate turbulent psf with
            parameters.  Always round.
                [row,col,T,p]

    pars: sequence
        A sequence describing the gaussians, as determined
        by the type parameter

    methods
    -------
    convolve(psf):
        Get a new GMix that is the convolution of the GMix with the input psf
    get_dlist():
        return a list of dicts representing the gaussian mixture
    get_pars():
        Get a copy of the parameter array used to initialize the GMix
    get_type():
        Get a copy of the type of the input parameters
    """
    def __init__(self, type, pars):
        pars=array(pars,dtype='f8')
        type=int(type)
        super(GMix,self).__init__(type, pars)
        self._pars=pars
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

        gmix = GMix(self._type,self._pars)
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
