#include <Python.h>
#include <numpy/arrayobject.h> 

#include "gvec.h"
#include "image.h"
#include "gmix_image.h"

struct PyGVecObject {
  PyObject_HEAD
  struct gvec *gvec;
};


struct PyGMixObject {
  PyObject_HEAD
  struct PyGVecObject* gvec_obj;

  // we will increment the ref count and keep until we don't need it
  PyObject* image_obj;
  // Don't call image_free on this; it does not own its data, only
  // pointing to the data of the image_obj
  struct image image;

  int flags;
  size_t numiter;
};


/*
 * PyGVecObject methods
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

static int
gvec_copy_list_of_dicts(struct PyGVecObject* self, PyObject* lod)
{
    int status=1;
    Py_ssize_t num=0, i=0;
    PyObject *dict;

    self->gvec = NULL;

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

    self->gvec = gvec_new(num);

    if (self->gvec==NULL) {
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
        if (!gauss_from_dict(&self->gvec->data[i], dict)) {
            status=0;
            goto _gvec_copy_list_of_dicts_bail;
        }
    }
    
_gvec_copy_list_of_dicts_bail:
    if (status != 1) {
        if (self->gvec) {
            free(self->gvec);
            self->gvec=NULL;
        }
    }
    return status;
}

static int
PyGVecObject_init(struct PyGVecObject* self, PyObject *args, PyObject *kwds)
{
    // a list of dicts
    PyObject* list_of_pars=NULL;
    if (!PyArg_ParseTuple(args, (char*)"O", &list_of_pars)) {
        return -1;
    }

    if (!gvec_copy_list_of_dicts(self, list_of_pars)) {
        return -1;
    }

    return 0;
}



static void
PyGVecObject_dealloc(struct PyGVecObject* self)
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
PyGVecObject_write(struct PyGVecObject* self)
{
    printf("GVec ngauss: %lu\n", self->gvec->size);
    gvec_print(self->gvec,stdout);
    Py_RETURN_NONE;
}

static PyObject *
PyGVecObject_repr(struct PyGVecObject* self) {
    char buff[1024];

    sprintf(buff,"GVec ngauss: %lu", self->gvec->size);
    return PyString_FromString(buff);
}


// With dicts we must decref the object we insert
void add_double_to_dict(PyObject* dict, const char* key, double value) {
    PyObject* tobj=NULL;
    tobj=PyFloat_FromDouble(value);
    PyDict_SetItemString(dict, key, tobj);
    Py_XDECREF(tobj);
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
static PyObject*
PyGVecObject_asdicts(struct PyGVecObject* self)
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







/*
 * PyGMixObject methods
 */


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
 * no copy is made
 */
static int associate_image(struct PyGMixObject* self, PyObject* image_obj)
{
    int status=1;

    self->image_obj=NULL;

    self->image.data = 
        check_double_image(image_obj, &self->image.nrows, &self->image.ncols);
    if (!self->image.data) {
        status=0;
    }

    self->image_obj=image_obj;
    self->image.size = self->image.nrows*self->image.ncols;

    // successful association, incref the object
    Py_XINCREF(image_obj);
    return status;
}

/*
 * We rely on the python wrapper to make sure the Object passed in is
 * a GVec object!
 */
static int
PyGMixObject_init(struct PyGMixObject* self, PyObject *args, PyObject *kwds)
{
    struct gmix gmix = {0};
    PyObject* gvec_obj=NULL;
    PyObject* image_obj=NULL;
    self->image.has_sky=1;
    self->image.has_counts=1;
    unsigned int maxiter=0;

    if (!PyArg_ParseTuple(args, (char*)"OOddIi", 
                &gvec_obj, &image_obj, 
                &self->image.sky, &self->image.counts, &maxiter, &gmix.verbose)) {
        return -1;
    }

    if (!associate_image(self, image_obj)) {
        return -1;
    }
    // rely on python code to make sure this is the right type
    self->gvec_obj = (struct PyGVecObject *) gvec_obj;

    gmix.maxiter = maxiter;
    self->flags = gmix_image(&gmix, 
                             &self->image, 
                             self->gvec_obj->gvec, 
                             &self->numiter);
    return 0;
}

static void
PyGMixObject_dealloc(struct PyGMixObject* self)
{
    Py_XDECREF( (PyObject*) self->gvec_obj);
    Py_XDECREF( self->image_obj);

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
    Py_RETURN_NONE;
    printf("GMix\n"
           "\tngauss: %lu\n"
           "\timage[%lu,%lu]"
           ,self->gvec_obj->gvec->size,
           self->image.nrows, self->image.ncols);

    gvec_print(self->gvec_obj->gvec,stdout);
}

static PyObject *
PyGMixObject_repr(struct PyGMixObject* self) {
    char buff[1024];

    sprintf(buff,
            "GMix\n"
            "\tngauss: %lu\n"
            "\timage[%lu,%lu]"
            ,self->gvec_obj->gvec->size,
            self->image.nrows, self->image.ncols);
    return PyString_FromString(buff);
}








static PyMethodDef PyGMixObject_methods[] = {
    {"write", (PyCFunction)PyGMixObject_write, METH_VARARGS, 
        "print a representation\n"},
    {NULL}
};

static PyMethodDef PyGVecObject_methods[] = {
    {"write", (PyCFunction)PyGVecObject_write, METH_VARARGS,
        "print a representation\n"},
    {"asdicts", (PyCFunction)PyGVecObject_asdicts, METH_VARARGS, 
        "Get the gaussian parameters as a list of dictionaries\n"},
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


static PyTypeObject PyGVecObjectType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_gmix_image.GVec",             /*tp_name*/
    sizeof(struct PyGVecObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyGVecObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    //0,                         /*tp_repr*/
    (reprfunc)PyGVecObject_repr,                         /*tp_repr*/
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
    "A class to hold gaussian mixture variables.\n",
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyGVecObject_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyGVecObject_init,      /* tp_init */
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
    PyGVecObjectType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyGMixObjectType) < 0) {
        return NULL;
    }
    if (PyType_Ready(&PyGVecObjectType) < 0) {
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
    if (PyType_Ready(&PyGVecObjectType) < 0) {
        return;
    }
    m = Py_InitModule3("_gmix_image", gmix_module_methods, 
            "This module defines a class to fit a Gaussian Mixture to an image.\n");
    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyGMixObjectType);
    Py_INCREF(&PyGVecObjectType);
    PyModule_AddObject(m, "GMix", (PyObject *)&PyGMixObjectType);
    PyModule_AddObject(m, "GVec", (PyObject *)&PyGVecObjectType);

    import_array();
}
