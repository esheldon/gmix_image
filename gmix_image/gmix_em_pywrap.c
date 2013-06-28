#include <Python.h>
#include <numpy/arrayobject.h> 

#include "gmix.h"
#include "image.h"
#include "gmix_em.h"
#include "jacobian.h"


struct PyGMixEMObject {
  PyObject_HEAD

  struct image *image;
  // holds the result
  struct gmix *gmix;

  int flags;
  size_t numiter;

  double fdiff; // the fractional diff in weighted moments of 
                // the last iteration
};


/*
 * methods working on python objects
 */

/*
static int
get_dict_ssize_t(PyObject* dict, const char *key, ssize_t *val)
{
    int status=1;
    PyObject *obj=NULL;

    obj = PyDict_GetItemString(dict, key);
    if (obj == NULL) {
        PyErr_Format(PyExc_ValueError,
                    "Key '%s' not present in dict", key);
        status=0;
        goto _get_dict_ssize_t_bail;
    }

    *val = (ssize_t) PyInt_AsSsize_t(obj);
    if (PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError,
                    "Error converting '%s' to a ssize_t", key);
        status=0;
    }

_get_dict_ssize_t_bail:
    return status;
}
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
 * methods working on gmix or gaussians
 */


static int
gauss_from_dict(struct gauss *self, PyObject *dict)
{
    int status=1;
    double p=0,row=0,col=0,irr=0,irc=0,icc=0;

    if (!get_dict_double(dict,"p", &p)) {
        status=0;
        goto _gauss_copy_from_dict_bail;
    }
    if (!get_dict_double(dict,"row", &row)) {
        status=0;
        goto _gauss_copy_from_dict_bail;
    }
    if (!get_dict_double(dict,"col", &col)) {
        status=0;
        goto _gauss_copy_from_dict_bail;
    }
    if (!get_dict_double(dict,"irr", &irr)) {
        status=0;
        goto _gauss_copy_from_dict_bail;
    }
    if (!get_dict_double(dict,"irc", &irc)) {
        status=0;
        goto _gauss_copy_from_dict_bail;
    }
    if (!get_dict_double(dict,"icc", &icc)) {
        status=0;
        goto _gauss_copy_from_dict_bail;
    }

    gauss_set(self,p,row,col,irr,irc,icc);

_gauss_copy_from_dict_bail:
    return status;
}

static struct gmix
*gmix_from_list_of_dicts(PyObject* lod, const char* name)
{
    int status=1;
    Py_ssize_t num=0, i=0;
    PyObject *dict;
    struct gmix *self=NULL;


    if (!PyList_Check(lod)) {
        PyErr_SetString(PyExc_ValueError,
                        "You must init GMix with a list of dictionaries");
        status=0;
        goto _gmix_copy_list_of_dicts_bail;
    }

    num = PyList_Size(lod);
    if (num <= 0) {
        PyErr_SetString(PyExc_ValueError, 
                        "You must init GMix with a list of dictionaries "
                        "of size > 0");
        status=0;
        goto _gmix_copy_list_of_dicts_bail;
    }

    self = gmix_new(num);

    if (self==NULL) {
        PyErr_Format(PyExc_MemoryError, 
                     "GMix failed to allocate %ld gaussians",num);
        status=0;
        goto _gmix_copy_list_of_dicts_bail;
    }

    for (i=0; i<num; i++) {

        dict = PyList_GET_ITEM(lod, i);

        if (!PyDict_Check(dict)) {
            PyErr_Format(PyExc_ValueError, 
                    "Element %ld of '%s' is not a dict", i, name);
            status=0;
            goto _gmix_copy_list_of_dicts_bail;
        }
        if (!gauss_from_dict(&self->data[i], dict)) {
            status=0;
            goto _gmix_copy_list_of_dicts_bail;
        }
    }

_gmix_copy_list_of_dicts_bail:
    if (status != 1) {
        if (self) {
            free(self);
            self=NULL;
        }
    }
    // may be NULL
    return self;
}



int jacobian_from_dict(struct gmix_em *self, PyObject *jacobian_dict)
{
    int status=1;
    double row0=0,col0=0,dudrow=0,dudcol=0,dvdrow=0,dvdcol=0;

    if (!get_dict_double(jacobian_dict,"row0", &row0)) {
        status=0;
        goto _jacobian_from_dict_bail;
    }
    if (!get_dict_double(jacobian_dict,"col0", &col0)) {
        status=0;
        goto _jacobian_from_dict_bail;
    }
    if (!get_dict_double(jacobian_dict,"dudrow", &dudrow)) {
        status=0;
        goto _jacobian_from_dict_bail;
    }
    if (!get_dict_double(jacobian_dict,"dudcol", &dudcol)) {
        status=0;
        goto _jacobian_from_dict_bail;
    }
    if (!get_dict_double(jacobian_dict,"dvdrow", &dvdrow)) {
        status=0;
        goto _jacobian_from_dict_bail;
    }
    if (!get_dict_double(jacobian_dict,"dvdcol", &dvdcol)) {
        status=0;
        goto _jacobian_from_dict_bail;
    }

    gmix_em_add_jacobian(self, row0, col0, dudrow, dudcol, dvdrow, dvdcol);

_jacobian_from_dict_bail:
    return status;
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
 * PyGMixEMObject methods
 */



/*
 * no copy is made.
 */
struct image *associate_image(PyObject* image_obj, double counts)
{
    struct image *image=NULL;
    size_t nrows=0, ncols=0;
    double *data=NULL;
    int dont_alloc_data=0;
    size_t i=0;

    data = check_double_image(image_obj, &nrows, &ncols);
    if (data) {
        // do this ourselves instead of using image_from_array to avoid
        // re-calculating counts

        image = _image_new(nrows, ncols, dont_alloc_data);

        image->rows[0] = data;
        for(i = 1; i < nrows; i++) {
            image->rows[i] = image->rows[i-1] + ncols;
        }
        IM_SET_COUNTS(image, counts);
    }

    return image;
}

/*
int add_mask_to_image(struct PyGMixEMObject* self, PyObject* bound_obj)
{
    int status=1;
    struct bound bound = {0};

    if (!PyDict_Check(bound_obj)) {
        PyErr_SetString(PyExc_ValueError, "Bound is not a dict");
        status=0;
        goto _bound_copy_from_dict_bail;
    }


    if (!get_dict_ssize_t(bound_obj,"rowmin", &bound.rowmin)) {
        status=0;
        goto _bound_copy_from_dict_bail;
    }
    if (!get_dict_ssize_t(bound_obj,"rowmax", &bound.rowmax)) {
        status=0;
        goto _bound_copy_from_dict_bail;
    }
    if (!get_dict_ssize_t(bound_obj,"colmin", &bound.colmin)) {
        status=0;
        goto _bound_copy_from_dict_bail;
    }
    if (!get_dict_ssize_t(bound_obj,"colmax", &bound.colmax)) {
        status=0;
        goto _bound_copy_from_dict_bail;
    }

    // 1 means update counts, needed for EM
    image_add_mask(self->image, &bound, 1);

_bound_copy_from_dict_bail:
    return status;
}
*/
void gmix_cleanup(struct PyGMixEMObject* self)
{
    self->image    = image_free(self->image);
    self->gmix     = gmix_free(self->gmix);
}

static int
PyGMixEMObject_init(struct PyGMixEMObject* self, PyObject *args, PyObject *kwds)
{
    int status=1;

    // should use gmix_em_new generally, but 
    struct gmix_em *gmix_em = NULL;
    double tol=0;
    int cocenter=0, verbose=0;
    unsigned int maxiter=0;

    PyObject* guess_lod=NULL;
    PyObject* image_obj=NULL;
    PyObject* jacobian_dict=NULL;

    double sky=0, counts=0;
    self->image=NULL; self->gmix=NULL;

    static char* argnames[] = {"image", "sky", "counts", "guess",
                               "maxiter", "tol",
                               "cocenter",
                               "verbose",
                               "jacobian",
                               NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds,
                                     (char*)"OddOId|iiO",
                                     argnames,
                                     &image_obj, 
                                     &sky, 
                                     &counts, 
                                     &guess_lod,  // this has the guesses
                                     &maxiter, 
                                     &tol,
                                     &cocenter,
                                     &verbose,
                                     &jacobian_dict)) {
        return -1;
    }

    gmix_em = gmix_em_new(maxiter, tol, cocenter, verbose);

    // no copying
    self->image = associate_image(image_obj, counts);
    if (!self->image) {
        status=0;
        goto _gmix_init_bail;
    }
    IM_SET_SKY(self->image, sky);

    if (jacobian_dict != Py_None) {
       if (!jacobian_from_dict(gmix_em, jacobian_dict)) {
           status=0;
           goto _gmix_init_bail;
       }
    }

    // copy all data from dict into the gmix as a starting point
    // need to free gmix in the destructor
    self->gmix = gmix_from_list_of_dicts(guess_lod,"guess");
    if (!self->gmix) {
        status=0;
        goto _gmix_init_bail;
    }


    if (gmix_em->cocenter) {
        self->flags = gmix_em_cocenter_run(gmix_em, 
                self->image, 
                self->gmix, 
                &self->numiter,
                &self->fdiff);
    } else {
        self->flags = gmix_em_run(gmix_em, 
                self->image, 
                self->gmix, 
                &self->numiter,
                &self->fdiff);
    }
_gmix_init_bail:

    free(gmix_em);

    if (!status) {
        gmix_cleanup(self);
        return -1;
    }
    return 0;
}

static void
PyGMixEMObject_dealloc(struct PyGMixEMObject* self)
{
    gmix_cleanup(self);

#if ((PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 6) || (PY_MAJOR_VERSION == 3))
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    // old way, removed in python 3
    self->ob_type->tp_free((PyObject*)self);
#endif

}

static PyObject*
PyGMixEMObject_write(struct PyGMixEMObject* self)
{
    printf("GMix\n"
           "\tngauss:  %lu\n"
           "\tnumiter: %lu\n"
           "\tfdiff:   %g\n"
           "\tflags:   %d\n"
            ,self->gmix->size,
            self->numiter,
            self->fdiff,
            self->flags);

    printf("gaussians\n");
    gmix_print(self->gmix,stdout);
    Py_RETURN_NONE;
}

static PyObject *
PyGMixEMObject_repr(struct PyGMixEMObject* self) {
    char buff[1024];

    sprintf(buff,
            "GMix\n"
            "\tngauss:  %lu\n"
            "\tnumiter: %lu\n"
            "\tfdiff:   %g\n"
            "\tflags:   %d\n"
            ,self->gmix->size,
            self->numiter,
            self->fdiff,
            self->flags);
    return PyString_FromString(buff);
    Py_RETURN_NONE;
}


static PyObject*
PyGMixEMObject_get_dlist(struct PyGMixEMObject* self)
{
    PyObject* lod=NULL;
    PyObject* tdict=NULL;
    size_t i=0;
    struct gmix *gmix=NULL;
    struct gauss *gauss=NULL;

    gmix = self->gmix;
    lod=PyList_New(0);

    gauss=gmix->data;
    for (i=0; i<gmix->size; i++) {
        tdict = gauss_to_dict(gauss);
        PyList_Append(lod, tdict);
        Py_XDECREF(tdict);
        gauss++;
    }
    return lod;
}





static PyObject*
PyGMixEMObject_get_flags(struct PyGMixEMObject* self)
{
    return PyInt_FromLong((long) self->flags);
}
static PyObject*
PyGMixEMObject_get_numiter(struct PyGMixEMObject* self)
{
    return PyInt_FromLong((long) self->numiter);
}
static PyObject*
PyGMixEMObject_get_fdiff(struct PyGMixEMObject* self)
{
    return PyFloat_FromDouble(self->fdiff);
}








static PyMethodDef PyGMixEMObject_methods[] = {
    {"write", (PyCFunction)PyGMixEMObject_write, METH_VARARGS, 
        "print a representation\n"},
    {"get_dlist", (PyCFunction)PyGMixEMObject_get_dlist, METH_VARARGS, 
        "get the gaussian parameters as a list of dicts\n"},
    {"get_flags", (PyCFunction)PyGMixEMObject_get_flags, METH_VARARGS, 
        "get the flags from the processing\n"},
    {"get_numiter", (PyCFunction)PyGMixEMObject_get_numiter, METH_VARARGS, 
        "get the number of iterations during processing\n"},
    {"get_fdiff", (PyCFunction)PyGMixEMObject_get_fdiff, METH_VARARGS, 
        "get the number of iterations during processing\n"},
    {NULL}
};


static PyTypeObject PyGMixEMObjectType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_gmix_em.GMix",             /*tp_name*/
    sizeof(struct PyGMixEMObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyGMixEMObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    //0,                         /*tp_repr*/
    (reprfunc)PyGMixEMObject_repr,                         /*tp_repr*/
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
    PyGMixEMObject_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyGMixEMObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,                 /* tp_new */
};


static PyMethodDef gmix_module_methods[] = {
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_gmix_em",      /* m_name */
        "Defines the GMix and GMix classes",  /* m_doc */
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
init_gmix_em(void) 
{
    PyObject* m;


    PyGMixEMObjectType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyGMixEMObjectType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    if (PyType_Ready(&PyGMixEMObjectType) < 0) {
        return;
    }
    m = Py_InitModule3("_gmix_em", gmix_module_methods, 
            "This module defines a class to fit a Gaussian Mixture to an image.\n");
    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyGMixEMObjectType);
    PyModule_AddObject(m, "GMixEM", (PyObject *)&PyGMixEMObjectType);

    import_array();
}
