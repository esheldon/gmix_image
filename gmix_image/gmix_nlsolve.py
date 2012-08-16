# This file was automatically generated by SWIG (http://www.swig.org).
# Version 1.3.40
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.
# This file is compatible with both classic and new-style classes.

from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_gmix_nlsolve', [dirname(__file__)])
        except ImportError:
            import _gmix_nlsolve
            return _gmix_nlsolve
        if fp is not None:
            try:
                _mod = imp.load_module('_gmix_nlsolve', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _gmix_nlsolve = swig_import_helper()
    del swig_import_helper
else:
    import _gmix_nlsolve
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


class GMixCoellipSolver(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, GMixCoellipSolver, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, GMixCoellipSolver, name)
    __repr__ = _swig_repr
    def __init__(self, *args, **kwargs): 
        this = _gmix_nlsolve.new_GMixCoellipSolver(*args, **kwargs)
        try: self.this.append(this)
        except: self.this = this
    __swig_destroy__ = _gmix_nlsolve.delete_GMixCoellipSolver
    __del__ = lambda self : None;
    def get_success(self): return _gmix_nlsolve.GMixCoellipSolver_get_success(self)
    def get_chi2per(self): return _gmix_nlsolve.GMixCoellipSolver_get_chi2per(self)
    def get_pars(self): return _gmix_nlsolve.GMixCoellipSolver_get_pars(self)
    def get_cov(self): return _gmix_nlsolve.GMixCoellipSolver_get_cov(self)
    def get_nrows(self): return _gmix_nlsolve.GMixCoellipSolver_get_nrows(self)
    def get_ncols(self): return _gmix_nlsolve.GMixCoellipSolver_get_ncols(self)
    def get_val(self, *args, **kwargs): return _gmix_nlsolve.GMixCoellipSolver_get_val(self, *args, **kwargs)
GMixCoellipSolver_swigregister = _gmix_nlsolve.GMixCoellipSolver_swigregister
GMixCoellipSolver_swigregister(GMixCoellipSolver)



