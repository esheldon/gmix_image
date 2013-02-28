import numpy
from numpy import array, zeros
import copy
from . import _render

GMIX_FULL=0
GMIX_COELLIP=1
GMIX_TURB=2
GMIX_EXP=3
GMIX_DEV=4
GMIX_COELLIP_TFRAC=5

_gmix_type_dict={'full':    GMIX_FULL,
                 'coellip': GMIX_COELLIP,
                 'gturb':   GMIX_TURB,
                 'turb':    GMIX_TURB,
                 'gexp':    GMIX_EXP,
                 'exp':     GMIX_EXP,
                 'gdev':    GMIX_DEV,
                 'dev':     GMIX_DEV,
                 'coellip-Tfrac': GMIX_COELLIP_TFRAC}

def as_gmix_type(type_in):
    if isinstance(type_in,basestring):
        type_in=type_in.lower()
        if type_in not in _gmix_type_dict:
            raise ValueError("unknown gmixtype: '%s'" % type_in)
        type_out = _gmix_type_dict[type_in]
    else:
        type_out = int(type_in)

    return type_out


def GMixCoellip(pars):
    """
    Generate a co-elliptical gaussian mixture.

    parameters
    ----------
    pars: sequence
        [row,col,e1,e2,T1,T2...,p1,p2...]
    """
    return GMix(pars, type=GMIX_COELLIP)


def GMixExp(pars):
    """
    Generate a gaussian mixture representing an approximate exponential disk.
    Only works well if the object is not too large compared to the PSF.

    parameters
    ----------
    pars: sequence
        [row,col,e1,e2,T,p]
    """
    gmix=GMix(pars, type=GMIX_EXP)
    return gmix

def GMixDev(pars):
    """
    Generate a gaussian mixture representing an approximate devauc profile.
    Only works well if the object is not too large compared to the PSF.

    parameters
    ----------
    pars: sequence
        [row,col,e1,e2,T,p]
    """
    gmix=GMix(pars, type=GMIX_DEV)
    return gmix

def GMixTurb(pars):
    """
    Generate a gaussian mixture representing an approximate turbulent 
    atmospheric PSF.  Can be ellipticial.

    parameters
    ----------
    pars: sequence
        [row,col,e1,e2,T,p]
    """
    return GMix(pars, type=GMIX_TURB)



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
    get_size():
        get number of gaussians in mixture
    get_dlist():
        get a list of dicts representing the gaussian mixture
    get_T():
        get T=sum(p*T_i)/sum(p)
    get_e1e2T():
        get stats based on averaged moments val=sum(p*val_i)/sum(p)
    get_cen():
        get cen=sum(p*cen_i)/sum(p)
    set_cen(row,col):
        set all centers to the input.
    get_pars():
        get a copy of the parameters used to initialize the GMix
    get_type():
        get a copy of the type of the input parameters
    """
    def __init__(self, pars, type=GMIX_FULL):

        if isinstance(pars, GMix):
            self.__init__(pars.get_pars())
        else:
            type=as_gmix_type(type)

            if type==GMIX_FULL: 
                # we also want a pars array if list of dicts was sent
                if isinstance(pars,list) and isinstance(pars[0],dict):
                    pars_array=gmix2pars(pars)
                else:
                    pars_array=array(pars, dtype='f8')
            else:
                pars_array=array(pars,dtype='f8')

            super(GMix,self).__init__(type, pars_array)

    #def _print_type(self,t):
    #    print 'type(type):',type(t)

    def convolve(self, psf):
        """
        Get a new GMix that is the convolution of the GMix with the input psf

        parameters
        ----------
        psf: GMix object
        """
        if not isinstance(psf, GMix):
            raise ValueError("Can only convolve with another GMix object")

        gmix = GMix(self.get_pars())
        gmix._convolve_replace(psf)
        return gmix

    '''
    def get_type(self):
        """
        Get a copy of the type of the input parameters
        """
        return copy.copy(self._type)
    '''

    def __repr__(self):
        import pprint
        dlist=self.get_dlist()
        return pprint.pformat(dlist)


def togmix(gmix, coellip=False, Tfrac=False):
    """
    Conver the input to a GMix object

    parameters
    ----------
    gmix:
        either a pars arrary or a list of dicts or a GMix object
    coellip: bool
        If True, interpret the pars array as coelliptical
    """
    if isinstance(gmix, GMix):
        # noop
        return gmix

    if isinstance(gmix[0], dict):
        # full gaussian mixture as list of dicts
        return GMix(gmix)

    if coellip:
        # special coelliptical form
        return GMix(gmix, type=gmix.GMIX_COELLIP)
    elif Tfrac:
        return GMix(gmix, type=gmix.GMIX_COELLIP_TFRAC)
    else:
        # we assume this is a full gaussian mixture in array form
        return GMix(gmix)


def gmix2pars(gmix_in):
    """
    convert a list of dictionaries to an array.

    The packing is [p1,row1,col1,irr1,irc1,icc1,
                    p2,row2,....]
    """
    if isinstance(gmix_in, GMix):
        gm=gmix_in.get_dlist()
    elif isinstance(gmix_in, list) and isinstance(gmix_in[0],dict):
        gm=gmix_in
    else:
        raise ValueError("input should be a GMix object or list of dicts")

    ngauss=len(gm)
    pars=zeros(ngauss*6,dtype='f8')
    for i,g in enumerate(gm):
        beg=i*6
        pars[beg+0] = g['p']
        pars[beg+1] = g['row']
        pars[beg+2] = g['col']
        pars[beg+3] = g['irr']
        pars[beg+4] = g['irc']
        pars[beg+5] = g['icc']

    return pars


