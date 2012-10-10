import numpy
from numpy import array
import copy
from . import _gvec
from ._gvec import version

class GVec(_gvec.GVec):
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
            parameters
                [row,col,e1,e2,T,p]

    pars: sequence
        A sequence describing the gaussians, as determined
        by the type parameter

    methods
    -------
    convolve(psf):
        Get a new GVec that is the convolution of the GVec with the input psf
    get_dlist():
        return a list of dicts representing the gaussian mixture
    get_pars():
        Get a copy of the parameter array used to initialize the GVec
    get_type():
        Get a copy of the type of the input parameters
    """
    def __init__(self, type, pars):
        pars=array(pars,dtype='f8')
        type=int(type)
        super(GVec,self).__init__(type, pars)
        self._pars=pars
        self._type=type


    def convolve(self, psf):
        """
        Get a new GVec that is the convolution of the GVec with the input psf

        parameters
        ----------
        psf: GVec object
        """
        if not isinstance(psf, GVec):
            raise ValueError("Can only convolve with another GVec object")

        gvec = GVec(self._type,self._pars)
        gvec.convolve_inplace(psf)
        return gvec

    def get_pars(self):
        """
        Get a copy of the parameter array used to initialize the GVec
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
