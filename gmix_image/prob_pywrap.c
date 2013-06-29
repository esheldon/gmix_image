#include <Python.h>
#include <numpy/arrayobject.h> 

#include "prob.h"
#include "gmix.h"
#include "jacobian.h"
#include "render.h"
#include "image.h"
#include "defs.h"
#include "obs.h"

#include "gmix_pywrap.h"

#include "py_helpers.h"

struct PyProbObject {
    PyObject_HEAD
    enum prob_type type;
    enum gmix_model model;
    long npars;
    struct obs_list *obs_list;
    void *data;
};


static int check_numpy_image(PyObject *obj, const char *name)
{

    if (!PyArray_Check(obj) 
            || NPY_DOUBLE != PyArray_TYPE(obj)
            || 2 != PyArray_NDIM(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be numpy double array",name);
        return 0;
    }
    return 1;
}
static int check_numpy_array(PyObject *obj, const char *name)
{
    if (!PyArray_Check(obj) || NPY_DOUBLE != PyArray_TYPE(obj)) {
        PyErr_Format(PyExc_TypeError, "%s must be numpy double array",name);
        return 0;
    }
    return 1;
}

static long check_lists(PyObject *im_list,
                        PyObject *wt_list,
                        PyObject *jacob_list,
                        PyObject *psf_gmix_list)
{
    size_t s=0;
    if (!PyList_Check(im_list)) {
        PyErr_Format(PyExc_TypeError, "im_list is not a list");
        return 0;
    }
    if (!PyList_Check(wt_list)) {
        PyErr_Format(PyExc_TypeError, "wt_list is not a list");
        return 0;
    }
    if (!PyList_Check(jacob_list)) {
        PyErr_Format(PyExc_TypeError, "jacob_list is not a list");
        return 0;
    }
    if (!PyList_Check(psf_gmix_list)) {
        PyErr_Format(PyExc_TypeError, "psf_gmix_list is not a list");
        return 0;
    }
    s=PyList_Size(im_list);
    if (   (PyList_Size(wt_list) != s)
        || (PyList_Size(jacob_list) != s)
        || (PyList_Size(psf_gmix_list) != s) ) {

        PyErr_Format(PyExc_ValueError, "all lists must be same size");
        return 0;
    }
    return 1;
}

// no error checking
static
struct obs_list *load_obs_list(PyObject *im_list,
                               PyObject *wt_list,
                               PyObject *jacob_list,
                               PyObject *psf_gmix_list)
{
    ssize_t nobs=0;
    struct obs_list *obs_list=NULL;
    ssize_t i=0;

    PyObject *im_obj=NULL;
    PyObject *wt_obj=NULL;
    PyObject *jacob_dict_obj=NULL;
    struct PyGMixObject *gmix_obj=NULL;

    struct image *im_tmp=NULL;
    struct image *wt_tmp=NULL;
    struct jacobian jacob_tmp;

    long ok=1;

    if (! (ok=check_lists(im_list,wt_list,jacob_list,psf_gmix_list))) {
        goto _load_obs_list_bail;
    }
    nobs = PyList_Size(im_list);

    obs_list=obs_list_new(nobs);
    for (i=0; i<nobs; i++) {

        // borrowed reference, no need to decref
        im_obj=PyList_GetItem(im_list, i);
        wt_obj=PyList_GetItem(wt_list, i);
        jacob_dict_obj=PyList_GetItem(jacob_list, i);

        // we have no type checking for this... how to?
        gmix_obj=(struct PyGMixObject *) PyList_GetItem(psf_gmix_list, i);

        Py_XINCREF(im_obj);
        if ( !(ok=check_numpy_image(im_obj,"image")) ) {
            goto _load_obs_list_bail;
        }
        Py_XDECREF(im_obj);

        if (! (ok=check_numpy_image(wt_obj,"weight") ) ) {
            goto _load_obs_list_bail;
        }

        im_tmp = pyhelp_associate_image(im_obj);
        wt_tmp = pyhelp_associate_image(wt_obj);

        if (! (ok=pyhelp_dict_to_jacob(jacob_dict_obj, &jacob_tmp) ) ) {
            goto _load_obs_list_bail;
        }

        obs_fill(&obs_list->data[i],
                 im_tmp,
                 wt_tmp,
                 &jacob_tmp,
                 gmix_obj->gmix);

        // underlying data not freed
        im_tmp=image_free(im_tmp);
        wt_tmp=image_free(wt_tmp);
    }

_load_obs_list_bail:
    if (!ok) {
        obs_list=obs_list_free(obs_list);
        // it is OK if they are NULL
        im_tmp=image_free(im_tmp);
        wt_tmp=image_free(wt_tmp);
    }
    return obs_list;
}


static
struct prob_data_simple_ba *load_ba_data(const struct obs_list *obs_list,
                                         enum gmix_model model,
                                         PyObject *config)
{
    PyObject *tmp=NULL;

    struct prob_data_simple_ba *data=NULL;

    struct dist_gauss cen1_prior, cen2_prior;
    struct dist_g_ba g_prior;
    struct dist_lognorm T_prior, counts_prior;
    double mean=0, width=0;
    long status=0;

    if (!PyDict_Check(config)) {
        PyErr_Format(PyExc_TypeError, "prior is not a dict");
        return NULL;
    }

    // borrowed ref
    // cen1
    tmp=PyDict_GetItemString(config, "cen1_mean");
    mean = PyFloat_AsDouble(tmp);

    mean = pyhelp_dict_get_double(config,"cen1_mean",&status);
    if (status) {
        return NULL;
    }
    width = pyhelp_dict_get_double(config,"cen1_width",&status);
    if (status) {
        return NULL;
    }

    dist_gauss_fill(&cen1_prior, mean, width);
    DBG dist_gauss_print(&cen1_prior,stderr);

    mean = pyhelp_dict_get_double(config,"cen2_mean",&status);
    if (status) {
        return NULL;
    }
    width = pyhelp_dict_get_double(config,"cen2_width",&status);
    if (status) {
        return NULL;
    }

    dist_gauss_fill(&cen2_prior, mean, width);
    DBG dist_gauss_print(&cen2_prior,stderr);


    // g
    width = pyhelp_dict_get_double(config,"g_width",&status);
    if (status) {
        return NULL;
    }

    dist_g_ba_fill(&g_prior, width);
    DBG dist_g_ba_print(&g_prior,stderr);

    // T
    mean = pyhelp_dict_get_double(config,"T_mean",&status);
    if (status) {
        return NULL;
    }
    width = pyhelp_dict_get_double(config,"T_width",&status);
    if (status) {
        return NULL;
    }

    dist_lognorm_fill(&T_prior, mean, width);
    DBG dist_lognorm_print(&T_prior,stderr);

    // counts
    mean = pyhelp_dict_get_double(config,"counts_mean",&status);
    if (status) {
        return NULL;
    }
    width = pyhelp_dict_get_double(config,"counts_width",&status);
    if (status) {
        return NULL;
    }

    dist_lognorm_fill(&counts_prior, mean, width);
    DBG dist_lognorm_print(&counts_prior,stderr);


    data=prob_data_simple_ba_new(model,
                                 obs_list,

                                 &cen1_prior,
                                 &cen2_prior,

                                 &g_prior,

                                 &T_prior,
                                 &counts_prior);

    return data;
}


long load_data(struct PyProbObject* self, PyObject *config)
{
    long status=1;
    switch (self->type) {
        case PROB_BA13:
            self->data = (void *) load_ba_data(self->obs_list,
                                               self->model,
                                               config);
            break;
        default:
            PyErr_Format(PyExc_ValueError, "Invalid PROB_TYPE: %d", self->type);
    }

    if (!self->data) {
        status=0;
    }
    return status;
}

static void cleanup(struct PyProbObject* self)
{
    if (self) {
        if (self->obs_list) {
            self->obs_list = obs_list_free(self->obs_list);
        }
        if (self->data) {
            switch (self->type) {
                case PROB_BA13:
                    self->data = prob_data_simple_ba_free(self->data);
                    break;

                default:
                    PyErr_Format(PyExc_ValueError,
                            "Invalid PROB_TYPE in dealloc: %d; memory leak is likely",
                            self->type);
            }
        }
    }
}

static long set_prob_and_model(struct PyProbObject* self, PyObject *config)
{
    long status=0;

    if (!PyDict_Check(config)) {
        PyErr_Format(PyExc_TypeError, "config must be a dict");
        return 0;
    }
    self->type=pyhelp_dict_get_long(config, "prob_type", &status);
    if (status) {
        return 0;
    }
    self->model=pyhelp_dict_get_long(config, "model", &status);
    if (status) {
        return 0;
    }

    return 1;
}

static int
PyProbObject_init(struct PyProbObject* self, PyObject *args)
{

    long ok=1;

    PyObject *im_list=NULL;
    PyObject *wt_list=NULL;
    PyObject *jacob_list=NULL;
    PyObject *psf_gmix_list=NULL;

    PyObject *config=NULL;

    if (!PyArg_ParseTuple(args,
                          (char*)"OOOOO",
                          &im_list,
                          &wt_list,
                          &jacob_list,
                          &psf_gmix_list,
                          &config) ) {
        return -1;
    }


    if ( !(ok=set_prob_and_model(self, config)) ) {
        goto _prob_obj_init_bail;
    }
    self->obs_list = load_obs_list(im_list,
                                   wt_list,
                                   jacob_list,
                                   psf_gmix_list);
    if (!self->obs_list) {
        ok=0;
        goto _prob_obj_init_bail;
    }

    if ( !(ok=load_data(self, config)) ) {
        goto _prob_obj_init_bail;
    }

_prob_obj_init_bail:
    if (!ok) {
        cleanup(self);
        return -1;
    }
    return 0;
}


static void
PyProbObject_dealloc(struct PyProbObject* self)
{
    cleanup(self);

#if PY_MAJOR_VERSION >= 3
    // introduced in python 2.6
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    // old way, removed in python 3
    self->ob_type->tp_free((PyObject*)self);
#endif

}

static PyObject *
PyProbObject_repr(struct PyProbObject* self) {
    return PyString_FromString("Prob");
}

static PyObject *PyProbObject_get_prob_type(struct PyProbObject* self)
{
    return PyInt_FromLong( (long) self->type );
}


static void do_calc(struct PyProbObject* self,
                    double *pars, long npars,
                    double *s2n_numer, double *s2n_denom,
                    double *lnprob, long *flags)
{
    switch (self->type) {
        case PROB_BA13:
            prob_simple_ba_calc(self->data,
                                pars, npars,
                                s2n_numer, s2n_denom,
                                lnprob, flags);

            if (*flags != 0) {
                PyErr_Format(PyExc_ValueError, "prob internal error, flags: %ld", *flags);
            }
            break;
        default:
            *flags=GMIX_WRONG_PROB_TYPE;
            PyErr_Format(PyExc_ValueError, "Invalid PROB_TYPE: %d", self->type);
    }
}

static PyObject *
PyProbObject_get_lnprob(struct PyProbObject* self, PyObject *args)
{
    PyObject* pars_obj=NULL;
    double *pars=NULL;
    double lnprob=0, s2n_numer=0, s2n_denom=0;
    long npars=0;
    long flags=0;
    PyObject *tup=NULL;

    if (!PyArg_ParseTuple(args, (char*)"O", &pars_obj)) {
        return NULL;
    }

    if (!check_numpy_array(pars_obj,"pars")) {
        return NULL;
    }

    npars=PyArray_SIZE(pars_obj);
    pars=PyArray_DATA(pars_obj);

    do_calc(self, pars, npars, &s2n_numer, &s2n_denom,
            &lnprob, &flags);

    // we only set flags if something catastrophic happened
    if (flags != 0) {
        return NULL;
    }

    tup = PyTuple_New(4);
    PyTuple_SetItem(tup, 0, PyFloat_FromDouble(lnprob));
    PyTuple_SetItem(tup, 1, PyFloat_FromDouble(s2n_numer));
    PyTuple_SetItem(tup, 2, PyFloat_FromDouble(s2n_denom));
    PyTuple_SetItem(tup, 3, PyInt_FromLong(flags));

    return tup;
}

static void eval_g_prior(struct PyProbObject* self,
                         double *g1,
                         double *g2,
                         double *prob,
                         long n,
                         long *status)
{
    long i=0;
    (*status)=0;
    switch (self->type) {
        case PROB_BA13:
            {
                struct prob_data_simple_ba *data=self->data;
                for (i=0; i<n; i++) {
                    prob[i] = dist_g_ba_prob(&data->g_prior, g1[i], g2[i]);
                } 
            }
            break;
        default:
            (*status)=1;
            PyErr_Format(PyExc_ValueError, "Invalid PROB_TYPE: %d", self->type);
    }
}

/*
   Evalutate just the gprior at the indicated g1,g2 values
*/
static PyObject *
PyProbObject_get_g_prior(struct PyProbObject* self, PyObject *args)
{
    PyObject* g1_obj=NULL;
    PyObject* g2_obj=NULL;
    PyObject* prob_obj=NULL;

    double *g1data=NULL, *g2data=NULL, *prob_data=NULL;
    npy_intp ng=0, dims[1];
    long status=0;

    if (!PyArg_ParseTuple(args, (char*)"OO", &g1_obj, &g2_obj)) {
        return NULL;
    }

    if (!check_numpy_array(g1_obj,"g1")) {
        return NULL;
    }
    if (!check_numpy_array(g2_obj,"g2")) {
        return NULL;
    }

    ng=PyArray_SIZE(g1_obj);
    if (PyArray_SIZE(g2_obj) != ng) {
        PyErr_Format(PyExc_ValueError, "g1 and g2 must be same size");
        return NULL;
    }

    g1data=PyArray_DATA(g1_obj);
    g2data=PyArray_DATA(g2_obj);

    dims[0] = ng;
    prob_obj =PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);
    prob_data=PyArray_DATA(prob_obj);

    eval_g_prior(self, g1data, g2data, prob_data, ng, &status);
    if (status) {
        Py_XDECREF(prob_obj);
        return NULL;
    }

    return prob_obj;
}



static PyMethodDef PyProbObject_methods[] = {
    {"get_lnprob", (PyCFunction)PyProbObject_get_lnprob,  METH_VARARGS,  "get the loglike for the input pars"},
    {"get_prob_type", (PyCFunction)PyProbObject_get_prob_type, METH_NOARGS, "Get the prob type"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyProbObjectType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_prob.ProbObject",             /*tp_name*/
    sizeof(struct PyProbObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyProbObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    //0,                         /*tp_repr*/
    (reprfunc)PyProbObject_repr,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "GVecIO Class",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyProbObject_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyProbObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    //PyProbObject_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};


static PyMethodDef prob_module_methods[] = {
    {NULL}
};


#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_render",      /* m_name */
        "Defines the some prob methods",  /* m_doc */
        -1,                  /* m_size */
        prob_module_methods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_prob(void) 
{
    PyObject* m;

    PyProbObjectType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyProbObjectType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    if (PyType_Ready(&PyProbObjectType) < 0) {
        return;
    }
    m = Py_InitModule3("_prob", prob_module_methods, 
            "This module gmix fit related routines.\n");
    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyProbObjectType);
    PyModule_AddObject(m, "Prob", (PyObject *)&PyProbObjectType);
    import_array();
}
