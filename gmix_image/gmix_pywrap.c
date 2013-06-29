#include <Python.h>
#include <numpy/arrayobject.h> 

#include "gmix_pywrap.h"

static int check_numpy_array(PyObject *obj)
{
    if (!PyArray_Check(obj) || NPY_DOUBLE != PyArray_TYPE(obj)) {
        return 0;
    }
    return 1;
}


static int
PyGMixObject_init(struct PyGMixObject* self, PyObject *args)
{
    int type=0;
    long flags=0;
    PyObject *pars_obj=NULL;

    npy_intp size=0;
    double *pars=NULL;

    self->gmix=NULL;

    if (!PyArg_ParseTuple(args, (char*)"iO", &type, &pars_obj)) {
        return -1;
    }

    if (!check_numpy_array(pars_obj)) {
        PyErr_SetString(PyExc_ValueError, "pars must be an array");
        return -1;
    }
    pars = PyArray_DATA(pars_obj);
    size = PyArray_SIZE(pars_obj);

    enum gmix_model model=type;

    self->gmix = gmix_new_model(model, pars, size, &flags);
    if (!self->gmix) {
        PyErr_Format(PyExc_ValueError, 
                "error constructing gmix: %ld", flags);
        return -1;
    }
    /*
    switch (type) {
        case 0:
            self->gmix = gmix_from_pars(pars, size);
            if (self->gmix == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing gmix from full array");
                return -1;
            }
            break;
        case 1:
            self->gmix = gmix_new_coellip(pars, size);
            if (self->gmix == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing gmix from coellip");
                return -1;
            }
            break;
        case 5:
            fprintf(stderr,"Using old Tfrac\n");
            self->gmix = gmix_new_coellip_Tfrac(pars, size);
            if (self->gmix == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing gmix from coellip Tfrac");
                return -1;
            }
            break;

        case 2:
            self->gmix = gmix_from_pars_turb(pars, size);
            if (self->gmix == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing gmix from turb");
                return -1;
            }
            break;

        case 3:
            self->gmix = gmix_from_pars_exp6(pars, size);
            if (self->gmix == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing exp6 gmix");
                return -1;
            }
            break;
        case 4:
            self->gmix = gmix_from_pars_dev10(pars, size);
            if (self->gmix == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing dev10 gmix");
                return -1;
            }
            break;

        case 6:
            self->gmix = gmix_from_pars_bd(pars, size);
            if (self->gmix == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing bd gmix");
                return -1;
            }
            break;



        default:
            PyErr_Format(PyExc_ValueError, "bad pars type value: %d", type);
            return -1;
    }
    */

    return 0;

}
static void
PyGMixObject_dealloc(struct PyGMixObject* self)
{
    self->gmix = gmix_free(self->gmix);
#if PY_MAJOR_VERSION >= 3
    // introduced in python 2.6
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    // old way, removed in python 3
    self->ob_type->tp_free((PyObject*)self);
#endif
}

static PyObject *
PyGMixObject_repr(struct PyGMixObject* self) {
    return PyString_FromString("GMix");
}

static void add_double_to_dict(PyObject* dict, const char* key, double value) 
{
    PyObject* tobj=NULL;
    tobj=PyFloat_FromDouble(value);
    PyDict_SetItemString(dict, key, tobj);
    Py_XDECREF(tobj);
}

static PyObject *PyGMixObject_get_size(struct PyGMixObject* self)
{
    return PyLong_FromLong((long)self->gmix->size);
}
static PyObject *PyGMixObject_get_pars(struct PyGMixObject* self)
{
    PyObject *pars_array=NULL;
    npy_intp dims[1];
    int npy_dtype=NPY_FLOAT64;
    double *pars=NULL;
    struct gauss *gauss=NULL;
    int i=0, ii=0, ngauss=0;

    ngauss=self->gmix->size;
    dims[0] = 6*ngauss;

    pars_array=PyArray_ZEROS(1, dims, npy_dtype, 0);
    pars=PyArray_DATA(pars_array);

    gauss=self->gmix->data;
    for (i=0; i<self->gmix->size; i++) {
        ii=i*6;

        pars[ii+0] = gauss->p;
        pars[ii+1] = gauss->row;
        pars[ii+2] = gauss->col;
        pars[ii+3] = gauss->irr;
        pars[ii+4] = gauss->irc;
        pars[ii+5] = gauss->icc;

        gauss++;
    }

    return pars_array;
}

static PyObject *PyGMixObject_get_dlist(struct PyGMixObject* self)
{
    PyObject *list=NULL;
    PyObject *dict=NULL;
    struct gauss *gauss=NULL;
    int i=0;

    list=PyList_New(self->gmix->size);
    gauss=self->gmix->data;
    for (i=0; i<self->gmix->size; i++) {
        dict = PyDict_New();

        add_double_to_dict(dict, "p",   gauss->p);
        add_double_to_dict(dict, "row", gauss->row);
        add_double_to_dict(dict, "col", gauss->col);
        add_double_to_dict(dict, "irr", gauss->irr);
        add_double_to_dict(dict, "irc", gauss->irc);
        add_double_to_dict(dict, "icc", gauss->icc);

        PyList_SetItem(list, i, dict);
        gauss++;
    }

    return list;
}


static PyObject *PyGMixObject_get_T(struct PyGMixObject* self)
{
    double T=0;
    T = gmix_get_T(self->gmix);
    return PyFloat_FromDouble(T);
}
static PyObject *PyGMixObject_get_psum(struct PyGMixObject* self)
{
    double psum=0;
    psum = gmix_get_psum(self->gmix);
    return PyFloat_FromDouble(psum);
}
static PyObject *PyGMixObject_set_psum(struct PyGMixObject* self, PyObject *args)
{
    double psum=0;

    if (!PyArg_ParseTuple(args, (char*)"d", &psum)) {
        return NULL;
    }

    gmix_set_psum(self->gmix, psum);

    Py_XINCREF(Py_None);
    return Py_None;
}


static PyObject *PyGMixObject_get_cen(struct PyGMixObject* self)
{
    npy_intp dims[1] = {2};
    int npy_dtype=NPY_FLOAT64;
    PyObject *cen=NULL;
    double *cendata=NULL;

    cen=PyArray_ZEROS(1, dims, npy_dtype, 0);
    cendata=PyArray_DATA(cen);

    gmix_get_cen(self->gmix, &cendata[0], &cendata[1]);

    return cen;
}
static PyObject *PyGMixObject_set_cen(struct PyGMixObject* self, PyObject *args)
{
    double row=0, col=0;

    if (!PyArg_ParseTuple(args, (char*)"dd", &row, &col)) {
        return NULL;
    }

    gmix_set_cen(self->gmix, row, col);

    Py_XINCREF(Py_None);
    return Py_None;
}


static PyObject *PyGMixObject_get_e1e2T(struct PyGMixObject* self)
{
    double psum=0;
    size_t i=0;
    struct gauss *gauss=NULL;
    npy_intp dims[1] = {3};
    int npy_dtype=NPY_FLOAT64;
    double irr=0, irc=0, icc=0, T=0;
    double e1=0, e2=0;
    PyObject *arr=NULL;
    double *data=NULL;

    arr=PyArray_ZEROS(1, dims, npy_dtype, 0);
    data=PyArray_DATA(arr);

    gauss=self->gmix->data;
    for (i=0; i<self->gmix->size; i++) {
        irr += gauss->irr*gauss->p;
        irc += gauss->irc*gauss->p;
        icc += gauss->icc*gauss->p;
        psum += gauss->p;
        gauss++;
    }

    irr /= psum;
    irc /= psum;
    icc /= psum;

    T = icc+irr;
    e1 = (icc-irr)/T;
    e2 = 2.*irc/T;

    data[0] = e1;
    data[1] = e2;
    data[2] = T;

    return arr;
}



/* error checking should happen in python */
static PyObject *PyGMixObject_convolve_replace(struct PyGMixObject* self, PyObject *args)
{
    PyObject *psf_obj=NULL;
    struct PyGMixObject *psf=NULL;
    struct gmix *new_gmix=NULL;
    long flags=0;

    if (!PyArg_ParseTuple(args, (char*)"O", &psf_obj)) {
        return NULL;
    }

    psf=(struct PyGMixObject *) psf_obj;

    new_gmix = gmix_convolve(self->gmix, psf->gmix, &flags);
    if (flags) {
        PyErr_Format(PyExc_ValueError, 
                "error convolving gmix: %ld", flags);
        return NULL;
    }
    self->gmix = gmix_free(self->gmix);

    self->gmix = new_gmix;

    Py_XINCREF(Py_None);
    return Py_None;
}


static PyMethodDef PyGMixObject_methods[] = {
    {"get_size", (PyCFunction)PyGMixObject_get_size, METH_NOARGS, "get_size\n\nreturn number of gaussians."},
    {"get_dlist", (PyCFunction)PyGMixObject_get_dlist, METH_NOARGS, "get_dlist\n\nreturn list of dicts."},
    {"get_e1e2T", (PyCFunction)PyGMixObject_get_e1e2T, METH_NOARGS, "get_e1e2T\n\nreturn stats based on average moments val=sum(val_i*p)/sum(p)."},
    {"get_T", (PyCFunction)PyGMixObject_get_T, METH_NOARGS, "get_T\n\nreturn T=sum(T_i*p)/sum(p)."},
    {"get_psum", (PyCFunction)PyGMixObject_get_psum, METH_NOARGS, "get_psum\n\nreturn sum(p)."},
    {"set_psum", (PyCFunction)PyGMixObject_set_psum, METH_VARARGS, "set_psum\n\nset new sum(p)."},
    {"get_cen", (PyCFunction)PyGMixObject_get_cen, METH_NOARGS, "get_cen\n\nreturn cen=sum(cen_i*p)/sum(p)."},
    {"set_cen", (PyCFunction)PyGMixObject_set_cen, METH_VARARGS, "set_cen\n\nSet all centers to the input row,col"},
    {"get_pars", (PyCFunction)PyGMixObject_get_pars, METH_NOARGS, "get_pars\n\nreturn full pars."},
    {"_convolve_replace", (PyCFunction)PyGMixObject_convolve_replace, METH_VARARGS, "convolve_inplace\n\nConvolve with the psf in place."},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyGMixType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_gmix.GMix",             /*tp_name*/
    sizeof(struct PyGMixObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyGMixObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    //0,                         /*tp_repr*/
    (reprfunc)PyGMixObject_repr,                         /*tp_repr*/
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
    "GMixIO Class",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyGMixObject_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyGMixObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    //PyGMixObject_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};




static PyMethodDef gmix_module_methods[] = {
    {NULL}
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_gmix",      /* m_name */
        "Defines gmix module",  /* m_doc */
        -1,                  /* m_size */
        gmix_module_methods,    /* m_methods */
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
init_gmix(void) 
{
    PyObject* m;

    PyGMixType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyGMixType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    if (PyType_Ready(&PyGMixType) < 0) {
        return;
    }
    m = Py_InitModule3("_gmix", gmix_module_methods, 
            "This module gmix fit related routines.\n");
    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyGMixType);
    PyModule_AddObject(m, "GMix", (PyObject *)&PyGMixType);
    import_array();
}
