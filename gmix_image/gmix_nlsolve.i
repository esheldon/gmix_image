%module gmix_nlsolve
%{
#include "gmix_nlsolve.h"
%}
%feature("kwargs");


%typemap(throws) const char * %{
    PyErr_SetString(PyExc_RuntimeError, $1);
    SWIG_fail;
%}


class GMixCoellipSolver : public NLSolver { 

    public:
        GMixCoellipSolver(PyObject* image_obj, 
                          PyObject* guess_obj, 
                          double skysig,
                          int maxiter, 
                          PyObject* psf_obj, 
                          int verbose) throw (const char*);
        ~GMixCoellipSolver();
        bool get_success() const;
        int get_flags() const;
        double get_chi2per() const;

        PyObject *get_pars() const;
        PyObject *get_pcov() const;
        PyObject *get_perr() const;

        PyObject *get_model() const;
        double get_s2n() const;

        // for testing
        size_t get_nrows();
        size_t get_ncols();
        double get_val(size_t row, size_t col) throw (const char*);
};
