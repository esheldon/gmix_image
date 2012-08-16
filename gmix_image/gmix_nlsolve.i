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
        GMixCoellipSolver(PyObject* image_obj, PyObject* guess_obj, int maxiter) throw (const char*);
        ~GMixCoellipSolver();
        bool get_success() const;
        PyObject *get_pars() const;
        PyObject *get_cov() const;

        // for testing
        size_t get_nrows();
        size_t get_ncols();
        double get_val(size_t row, size_t col) throw (const char*);
};
