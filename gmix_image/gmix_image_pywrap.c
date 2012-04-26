#include <Python.h>
#include <numpy/arrayobject.h> 

struct PyGVecObject {
  PyObject_HEAD
  int n;
  double *test;
};


struct PyGMixObject {
  PyObject_HEAD
  int n;
  double *test;
};


static int
PyGVecObject_init(struct PyGVecObject* self, PyObject *args, PyObject *kwds)
{
    int num=0;
    if (!PyArg_ParseTuple(args, (char*)"i", &num)) {
        return -1;
    }

    if (num <= 0) {
        PyErr_Format(PyExc_IOError, 
                     "GVec test data must have size >= 0, got %d",num);
        return -1;
    }

    self->n=num;
    self->test=calloc(num, sizeof(double));

    if (self->test==NULL) {
        PyErr_Format(PyExc_IOError, 
                     "GVec failed to allocate %d doubles",num);
        return -1;
    }
    return 0;
}



static void
PyGVecObject_dealloc(struct PyGVecObject* self)
{
    free(self->test);

#if ((PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 6) || (PY_MAJOR_VERSION == 3))
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    // old way, removed in python 3
    self->ob_type->tp_free((PyObject*)self);
#endif

}

static PyObject*
PyGVecObject_print_n(struct PyGVecObject* self)
{
    fprintf(stderr,"GVec n: %d\n", self->n);
    Py_RETURN_NONE;
}

static PyObject *
PyGVecObject_repr(struct PyGVecObject* self) {
    char buff[1024];

    sprintf(buff,"GVec n: %d", self->n);
    return PyString_FromString(buff);
}






static int
PyGMixObject_init(struct PyGMixObject* self, PyObject *args, PyObject *kwds)
{
    int num=0;
    if (!PyArg_ParseTuple(args, (char*)"i", &num)) {
        return -1;
    }

    if (num <= 0) {
        PyErr_Format(PyExc_IOError, 
                     "GMix test data must have size >= 0, got %d",num);
        return -1;
    }

    self->n=num;
    self->test=calloc(num, sizeof(double));

    if (self->test==NULL) {
        PyErr_Format(PyExc_IOError, 
                     "GMix failed to allocate %d doubles",num);
        return -1;
    }
    return 0;
}

static void
PyGMixObject_dealloc(struct PyGMixObject* self)
{
    free(self->test);

#if ((PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 6) || (PY_MAJOR_VERSION == 3))
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    // old way, removed in python 3
    self->ob_type->tp_free((PyObject*)self);
#endif

}

static PyObject*
PyGMixObject_print_n(struct PyGMixObject* self)
{
    fprintf(stderr,"GMix n: %d\n", self->n);
    Py_RETURN_NONE;
}

static PyObject *
PyGMixObject_repr(struct PyGMixObject* self) {
    char buff[1024];

    sprintf(buff,"GMix n: %d", self->n);
    return PyString_FromString(buff);
}








static PyMethodDef PyGMixObject_methods[] = {
    {"print_n", (PyCFunction)PyGMixObject_print_n, METH_VARARGS, "print n\n"},
    {NULL}
};

static PyMethodDef PyGVecObject_methods[] = {
    {"print_n", (PyCFunction)PyGVecObject_print_n, METH_VARARGS, "print n\n"},
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
