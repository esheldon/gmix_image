#include <Python.h>
#include <numpy/arrayobject.h> 

#include "prob.h"
#include "gmix.h"
#include "jacobian.h"
#include "render.h"
#include "image.h"
#include "defs.h"

#include "py_helpers.h"

struct PyProbObject {
    PyObject_HEAD
    enum prob_type type;
    void *data;
};


static
struct observations *load_observations(PyObject *im_list,
                                      PyObject *wt_list,
                                      PyObject *jacob_list,
                                      PyObject *psf_gmix_list)
{

}


static
struct prob_data_simple_ba *load_ba_data(PyObject *im_list,
                                         PyObject *wt_list,
                                         PyObject *jacob_list,
                                         PyObject *psf_gmix_list,
                                         enum prob_type prob_type,
                                         enum gmix_model model,
                                         PyObject *prior_dict)
{

}

static int
PyProbObject_init(struct PyProbObject* self, PyObject *args)
{

    int prob_type=0, model=0;

    PyObject *im_list=NULL;
    PyObject *wt_list=NULL;
    PyObject *jacob_list=NULL;
    PyObject *psf_gmix_list=NULL;

    PyObject *prior_dict=NULL;


    if (!PyArg_ParseTuple(args,
                          (char*)"OOOOii",
                          &im_list,
                          &wt_list,
                          &jacob_list,
                          &psf_gmix_list,
                          &prob_type,
                          &model,
                          &prior_dict) ) {
        return -1;
    }

    if (prob_type != PROB_BA13) {
        return -1;
    }

    self->type=prob_type;
    switch (self->type) {
        case PROB_BA13:
            self->data=NULL;
            /*
            self->data = prob_data_simple_ba_new(model,
                                                 obs_list,
                                                 cen1_prior,
                                                 cen2_prior,
                                                 g_prior,
                                                 T_prior,
                                                 counts_prior);
                                                 */
            break;
        default:
            PyErr_Format(PyExc_ValueError, "Invalid PROB_TYPE: %d", prob_type);
            return -1;
    }
    return 0;
}

static void
PyProbObject_dealloc(struct PyProbObject* self)
{
    switch (self->type) {
        case PROB_BA13:
            self->data = prob_data_simple_ba_free(self->data);
            break;

        default:
            PyErr_Format(PyExc_ValueError,
                "Invalid PROB_TYPE in dealloc: %d; memory leak is likely",
                self->type);
    }

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


static PyMethodDef PyProbObject_methods[] = {
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
    PyModule_AddObject(m, "ProbObject", (PyObject *)&PyProbObjectType);
    import_array();
}
