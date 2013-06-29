from sys import stderr
import copy

PROB_BA13=1
from .gmix import as_gmix_type, GMix
from . import _prob

_prob_type_dict={'ba13':PROB_BA13}

def as_prob_type(type_in):
    if isinstance(type_in,basestring):
        type_in=type_in.lower()
        if type_in not in _prob_type_dict:
            raise TypeError("unknown prob type: '%s'" % type_in)
        type_out = _prob_type_dict[type_in]
    else:
        type_out = int(type_in)

    return type_out



class Prob(_prob.Prob):
    def __init__(self, im_list, wt_list, jacob_list, psf_list, config):
        self.config=copy.deepcopy(config)

        self._integerify_types()

        self._check_psf(psf_list)
        super(Prob,self).__init__(im_list,
                                  wt_list,
                                  jacob_list,
                                  psf_list,
                                  self.config)

    def _integerify_types(self):
        # convert to integers
        self.config['model']     = as_gmix_type(self.config['model'])
        self.config['prob_type'] = as_prob_type(self.config['prob_type'])

    def _check_psf(self, psf_list):
        """
        We check types internally except for psf as gmix
        """
        for psf in psf_list:
            if not isinstance(psf,GMix):
                raise TypeError("psfs must be GMix objects")

