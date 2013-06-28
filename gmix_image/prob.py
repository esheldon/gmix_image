from sys import stderr
PROB_BA13=1
from .gmix import as_gmix_type
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
    def __init__(self, im_list, wt_list, jacob_list, psf_list,
                 prob_type, model,
                 prior_dict):
        prob_type_int, model_int = self._check_args(im_list,
                                                    wt_list,
                                                    jacob_list,
                                                    psf_list, 
                                                    prob_type,
                                                    model,
                                                    prior_dict)

        super(Prob,self).__init__(im_list,
                                  wt_list,
                                  jacob_list,
                                  psf_list,
                                  prob_type_int,
                                  model_int,
                                  prior_dict)

    def _check_args(self,
                    im_list,
                    wt_list,
                    jacob_list,
                    psf_list,
                    prob_type,
                    model,
                    prior_dict):

        model_int = as_gmix_type(model)
        prob_type_int = as_prob_type(prob_type)

        return prob_type_int, model_int
