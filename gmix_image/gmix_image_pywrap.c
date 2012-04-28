#include <Python.h>
#include <numpy/arrayobject.h> 

#include "gvec.h"
#include "image.h"
#include "gmix_image.h"


struct PyGMixObject {
  PyObject_HEAD

  // holds the result
  struct gvec *gvec;

  int flags;
  size_t numiter;

  double fdiff; // the fractional diff in weighted moments of 
                // the last iteration
};


/*
 * methods working on python objects
 */

static int
get_dict_double(PyObject* dict, const char* key, double *val)
{
    int status=1;
    PyObject *obj=NULL;

    obj = PyDict_GetItemString(dict, key);
    if (obj == NULL) {
        PyErr_Format(PyExc_ValueError,
                    "Key '%s' not present in dict", key);
        status=0;
        goto _get_dict_double_bail;
    }

    *val = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError,
                    "Error converting '%s' to a double", key);
        status=0;
    }

_get_dict_double_bail:
    return status;
}

// With dicts we must decref the object we insert
void add_double_to_dict(PyObject* dict, const char* key, double value) {
    PyObject* tobj=NULL;
    tobj=PyFloat_FromDouble(value);
    PyDict_SetItemString(dict, key, tobj);
    Py_XDECREF(tobj);
}


static double* 
check_double_image(PyObject* image, size_t *nrows, size_t *ncols)
{
    double* ptr=NULL;
    npy_intp *dims=NULL;
    if (!PyArray_Check(image)) {
        PyErr_SetString(PyExc_ValueError,
                        "image must be a 2D numpy array of type 64-bit float");
        return NULL;
    }
    if (2 != PyArray_NDIM((PyArrayObject*)image)) {
        PyErr_Format(PyExc_ValueError,
                     "image must be a 2D numpy array of type 64-bit float");
        return NULL;
    }
    if (NPY_DOUBLE != PyArray_TYPE((PyArrayObject*)image)) {
        PyErr_Format(PyExc_ValueError,
                     "image must be a 2D numpy array of type 64-bit float");
        return NULL;
    }

    ptr = PyArray_DATA((PyArrayObject*)image);
    dims = PyArray_DIMS((PyArrayObject*)image);

    *nrows = dims[0];
    *ncols = dims[1];
    return ptr;
}




/*
 * methods working on gvec or gaussians
 */


static int
gauss_from_dict(struct gauss *self, PyObject *dict)
{
    int status=1;

    if (!get_dict_double(dict,"p", &self->p)) {
        status=0;
        goto _gauss_copy_from_dict_bail;
    }
    if (!get_dict_double(dict,"row", &self->row)) {
        status=0;
        goto _gauss_copy_from_dict_bail;
    }
    if (!get_dict_double(dict,"col", &self->col)) {
        status=0;
        goto _gauss_copy_from_dict_bail;
    }
    if (!get_dict_double(dict,"irr", &self->irr)) {
        status=0;
        goto _gauss_copy_from_dict_bail;
    }
    if (!get_dict_double(dict,"irc", &self->irc)) {
        status=0;
        goto _gauss_copy_from_dict_bail;
    }
    if (!get_dict_double(dict,"icc", &self->icc)) {
        status=0;
        goto _gauss_copy_from_dict_bail;
    }

    self->det = self->irr*self->icc - self->irc*self->irc;

_gauss_copy_from_dict_bail:
    return status;
}

static struct gvec
*gvec_from_list_of_dicts(PyObject* lod)
{
    int status=1;
    Py_ssize_t num=0, i=0;
    PyObject *dict;
    struct gvec *self=NULL;


    if (!PyList_Check(lod)) {
        PyErr_SetString(PyExc_ValueError,
                        "You must init GVec with a list of dictionaries");
        status=0;
        goto _gvec_copy_list_of_dicts_bail;
    }

    num = PyList_Size(lod);
    if (num <= 0) {
        PyErr_SetString(PyExc_ValueError, 
                        "You must init GVec with a lis of dictionaries "
                        "of size > 0");
        status=0;
        goto _gvec_copy_list_of_dicts_bail;
    }

    self = gvec_new(num);

    if (self==NULL) {
        PyErr_Format(PyExc_MemoryError, 
                     "GVec failed to allocate %ld gaussians",num);
        status=0;
        goto _gvec_copy_list_of_dicts_bail;
    }

    for (i=0; i<num; i++) {

        dict = PyList_GET_ITEM(lod, i);

        if (!PyDict_Check(dict)) {
            PyErr_Format(PyExc_ValueError, 
                    "Element %ld is not a dict", i);
            status=0;
            goto _gvec_copy_list_of_dicts_bail;
        }
        if (!gauss_from_dict(&self->data[i], dict)) {
            status=0;
            goto _gvec_copy_list_of_dicts_bail;
        }
    }
    
_gvec_copy_list_of_dicts_bail:
    if (status != 1) {
        if (self) {
            free(self);
            self=NULL;
        }
    }
    // may be NULL
    return self;
}







static
PyObject *gauss_to_dict(struct gauss *self)
{
    PyObject *dict=NULL;

    dict = PyDict_New();
    add_double_to_dict(dict, "p", self->p);
    add_double_to_dict(dict, "row", self->row);
    add_double_to_dict(dict, "col", self->col);
    add_double_to_dict(dict, "irr", self->irr);
    add_double_to_dict(dict, "irc", self->irc);
    add_double_to_dict(dict, "icc", self->icc);
    add_double_to_dict(dict, "det", self->det);

    return dict;
}





/*
 * PyGMixObject methods
 */



/*
 * no copy is made.
 */
static int associate_image(struct image *self, PyObject* image_obj)
{
    int status=1;

    self->data = 
        check_double_image(image_obj, &self->nrows, &self->ncols);
    if (!self->data) {
        status=0;
    } else {
        self->size = self->nrows*self->ncols;
    }

    return status;
}

static int
PyGMixObject_init(struct PyGMixObject* self, PyObject *args, PyObject *kwds)
{
    struct gmix gmix = {0};
    struct image image = {0};

    PyObject* gauss_lod=NULL;
    PyObject* image_obj=NULL;
    unsigned int maxiter=0;

    if (!PyArg_ParseTuple(args, (char*)"OddOIdi", 
                &image_obj, 
                &image.sky, 
                &image.counts, 
                &gauss_lod,  // this has the guesses
                &maxiter, 
                &gmix.tol,
                &gmix.verbose)) {
        return -1;
    }

    // no copying
    if (!associate_image(&image, image_obj)) {
        return -1;
    }
    image.has_sky=1;
    image.has_counts=1;

    // copy all data from dict into the gvec as a starting point
    // need to free gvec in the destructor
    self->gvec = gvec_from_list_of_dicts(gauss_lod);
    if (!self->gvec) {
        return -1;
    }

    gmix.maxiter = maxiter;
    self->flags = gmix_image(&gmix, 
                             &image, 
                             self->gvec, 
                             &self->numiter,
                             &self->fdiff);
    return 0;
}

static void
PyGMixObject_dealloc(struct PyGMixObject* self)
{

    self->gvec = gvec_free(self->gvec);

#if ((PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 6) || (PY_MAJOR_VERSION == 3))
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    // old way, removed in python 3
    self->ob_type->tp_free((PyObject*)self);
#endif

}

static PyObject*
PyGMixObject_write(struct PyGMixObject* self)
{
    printf("GMix\n"
           "\tngauss: %lu\n"
            "\tflags: %d\n"
            "\tnumiter: %lu\n"
            "\tfdiff: %g\n"
            ,self->gvec->size,
            self->flags,
            self->numiter,
            self->fdiff);

    gvec_print(self->gvec,stdout);
    Py_RETURN_NONE;
}

static PyObject *
PyGMixObject_repr(struct PyGMixObject* self) {
    char buff[1024];

    sprintf(buff,
            "GMix\n"
            "\tngauss: %lu\n"
            "\tflags: %d\n"
            "\tnumiter: %lu\n"
            "\tfdiff: %g"
            ,self->gvec->size,
            self->flags,
            self->numiter,
            self->fdiff);
    return PyString_FromString(buff);
    Py_RETURN_NONE;
}


static PyObject*
PyGMixObject_get_pars(struct PyGMixObject* self)
{
    PyObject* lod=NULL;
    PyObject* tdict=NULL;
    size_t i=0;
    struct gvec *gvec=NULL;
    struct gauss *gauss=NULL;

    gvec = self->gvec;
    lod=PyList_New(0);

    gauss=gvec->data;
    for (i=0; i<gvec->size; i++) {
        tdict = gauss_to_dict(gauss);
        PyList_Append(lod, tdict);
        Py_XDECREF(tdict);
        gauss++;
    }
    return lod;
}





static PyObject*
PyGMixObject_get_flags(struct PyGMixObject* self)
{
    return PyInt_FromLong((long) self->flags);
}
static PyObject*
PyGMixObject_get_numiter(struct PyGMixObject* self)
{
    return PyInt_FromLong((long) self->numiter);
}
static PyObject*
PyGMixObject_get_fdiff(struct PyGMixObject* self)
{
    return PyFloat_FromDouble(self->fdiff);
}








static PyMethodDef PyGMixObject_methods[] = {
    {"write", (PyCFunction)PyGMixObject_write, METH_VARARGS, 
        "print a representation\n"},
    {"get_pars", (PyCFunction)PyGMixObject_get_pars, METH_VARARGS, 
        "get the gaussian parameters as a list of dicts\n"},
    {"get_flags", (PyCFunction)PyGMixObject_get_flags, METH_VARARGS, 
        "get the flags from the processing\n"},
    {"get_numiter", (PyCFunction)PyGMixObject_get_numiter, METH_VARARGS, 
        "get the number of iterations during processing\n"},
    {"get_fdiff", (PyCFunction)PyGMixObject_get_fdiff, METH_VARARGS, 
        "get the number of iterations during processing\n"},
    {NULL}
};


static PyTypeObject PyGMixObjectType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_gmix_image.GMix",             /*tp_name*/
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
    "A class to fit a Gaussian Mixture to an image.\n",
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
    PyType_GenericNew,                 /* tp_new */
};


static PyMethodDef gmix_module_methods[] = {
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_gmix_image",      /* m_name */
        "Defines the GMix and GVec classes",  /* m_doc */
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
init_gmix_image(void) 
{
    PyObject* m;


    PyGMixObjectType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyGMixObjectType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    if (PyType_Ready(&PyGMixObjectType) < 0) {
        return;
    }
    m = Py_InitModule3("_gmix_image", gmix_module_methods, 
            "This module defines a class to fit a Gaussian Mixture to an image.\n");
    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyGMixObjectType);
    PyModule_AddObject(m, "GMix", (PyObject *)&PyGMixObjectType);

    import_array();
}
