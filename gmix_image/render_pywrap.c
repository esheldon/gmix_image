#include <Python.h>
#include <numpy/arrayobject.h> 

#include "gmix.h"
#include "jacobian.h"
#include "render.h"
#include "image.h"
#include "bound.h"
#include "defs.h"

#include "gmix_pywrap.h"

/*

   Gaussian Mixtures

   This code defines the python gaussian mixture object

*/


static int check_numpy_image(PyObject *obj)
{
    if (!PyArray_Check(obj) 
            || NPY_DOUBLE != PyArray_TYPE(obj)
            || 2 != PyArray_NDIM(obj)) {
        return 0;
    }
    return 1;
}

static int check_numpy_array(PyObject *obj)
{
    if (!PyArray_Check(obj) || NPY_DOUBLE != PyArray_TYPE(obj)) {
        return 0;
    }
    return 1;
}




/*
 *
 *
 *
 *  code to render models and calculate likelihoods
 *
 *
 *
 *
 */

static int check_image_and_diff(PyObject *image_obj, PyObject *diff_obj)
{
    if (!check_numpy_image(image_obj)) {
        PyErr_SetString(PyExc_IOError, "image input must be a 2D double PyArrayObject");
        return 0;
    }
    // only care that diff is a double array
    if (diff_obj != Py_None && !check_numpy_array(diff_obj)) {
        PyErr_SetString(PyExc_IOError, "diff image input must be a double PyArrayObject");
        return 0;
    }
    return 1;
}
 
/*
 * no copy is made.
 */
//struct image *associate_image(PyObject* image_obj, size_t nrows, size_t ncols, double counts)
struct image *associate_image(PyObject* image_obj, size_t nrows, size_t ncols)
{
    struct image *image=NULL;
    double *data=NULL;
    int alloc_data=0; // we don't allocate
    size_t i=0;

    data = PyArray_DATA((PyArrayObject*)image_obj);

    image = _image_new(nrows, ncols, alloc_data);

    image->rows[0] = data;
    for(i = 1; i < nrows; i++) {
        image->rows[i] = image->rows[i-1] + ncols;
    }
    //IM_SET_COUNTS(image, counts);

    return image;
}




/*

   The python interfaces

*/


/*

   (model-data)/err

   for use in the LM algorithm

*/

static PyObject *
PyGMixFit_fill_ydiff(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    double ivar=0;
    PyObject* diff_obj=NULL;
    PyObject *gmix_pyobj=NULL;

    struct PyGMixObject *gmix_obj=NULL;
    struct image *image=NULL;
    struct image *diff=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OdOO", 
                &image_obj, &ivar, &gmix_pyobj, &diff_obj)) {
        return NULL;
    }

    if (!check_image_and_diff(image_obj,diff_obj)) {
        return NULL;
    }

    gmix_obj = (struct PyGMixObject *) gmix_pyobj;

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);
    diff = associate_image(diff_obj, dims[0], dims[1]);

    flags=fill_ydiff(image, ivar, gmix_obj->gmix, diff);

    // does not free underlying array
    image = image_free(image);
    diff = image_free(diff);

    return PyInt_FromLong(flags);
}


/*

   (model-data)/err

   for use in the LM algorithm

   using jacobian in u,v space
*/
static PyObject *
PyGMixFit_fill_ydiff_jacob(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    double ivar=0;
    double dudrow, dudcol, dvdrow, dvdcol;
    double row0=0, col0=0;
    PyObject *gmix_pyobj=NULL;
    PyObject* diff_obj=NULL;

    struct PyGMixObject *gmix_obj=NULL;
    struct image *image=NULL;
    struct image *diff=NULL;
    npy_intp *dims=NULL;

    struct jacobian jacob;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OdddddddOO", 
                &image_obj,
                &ivar,
                &dudrow, &dudcol, &dvdrow, &dvdcol,
                &row0,
                &col0,
                &gmix_pyobj,
                &diff_obj)) {
        return NULL;
    }

    if (!check_image_and_diff(image_obj,diff_obj)) {
        return NULL;
    }

    gmix_obj = (struct PyGMixObject *) gmix_pyobj;

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);
    diff = associate_image(diff_obj, dims[0], dims[1]);

    jacobian_set(&jacob, row0, col0, dudrow, dudcol, dvdrow, dvdcol);

    flags=fill_ydiff_jacob(image, ivar, &jacob,
                           gmix_obj->gmix, diff);
    // does not free underlying array
    image  = image_free(image);
    diff   = image_free(diff);

    return PyInt_FromLong(flags);
}



/*

   (model-data)/err

   for use in the LM algorithm

   using jacobian in u,v space and weight image
*/
static PyObject *
PyGMixFit_fill_ydiff_wt_jacob(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    PyObject* weight_obj=NULL;
    double dudrow, dudcol, dvdrow, dvdcol;
    double row0=0, col0=0;
    PyObject *gmix_pyobj=NULL;
    PyObject* diff_obj=NULL;

    struct PyGMixObject *gmix_obj=NULL;
    struct image *image=NULL, *weight=NULL;
    struct image *diff=NULL;
    npy_intp *dims=NULL;

    struct jacobian jacob;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOddddddOO", 
                &image_obj,
                &weight_obj,
                &dudrow, &dudcol, &dvdrow, &dvdcol,
                &row0,
                &col0,
                &gmix_pyobj,
                &diff_obj)) {
        return NULL;
    }

    if (!check_image_and_diff(image_obj,diff_obj)) {
        return NULL;
    }
    if (!check_numpy_image(weight_obj)) {
        PyErr_SetString(PyExc_IOError,
                "weight image must be a 2D double PyArrayObject");
        return NULL;
    }

    gmix_obj = (struct PyGMixObject *) gmix_pyobj;

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);
    weight = associate_image(weight_obj, dims[0], dims[1]);
    diff = associate_image(diff_obj, dims[0], dims[1]);

    jacobian_set(&jacob, row0, col0, dudrow, dudcol, dvdrow, dvdcol);

    flags=fill_ydiff_wt_jacob(image, weight, &jacob,
                              gmix_obj->gmix, diff);

    // does not free underlying array
    image  = image_free(image);
    weight = image_free(weight);
    diff   = image_free(diff);

    return PyInt_FromLong(flags);
}





static PyObject *
PyGMixFit_loglike_margamp(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    PyObject* gmix_pyobj=NULL;
    struct PyGMixObject *gmix_obj=NULL;
    double A=0, ierr=0;

    double loglike=0;
    PyObject *tup=NULL;

    struct image *image=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOdd", 
                                       &image_obj, &gmix_pyobj,
                                       &A, &ierr)) {
        return NULL;
    }

    if (!check_numpy_image(image_obj)) {
        PyErr_SetString(PyExc_IOError, "image input must be a 2D double PyArrayObject");
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    gmix_obj = (struct PyGMixObject *) gmix_pyobj;

    flags=calculate_loglike_margamp(image, gmix_obj->gmix, A, ierr, &loglike);

    // does not free underlying array
    image = image_free(image);

    tup = PyTuple_New(2);
    PyTuple_SetItem(tup, 0, PyFloat_FromDouble(loglike));
    PyTuple_SetItem(tup, 1, PyInt_FromLong((long)flags));

    return tup;
}




static PyObject *
PyGMixFit_loglike(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    PyObject* gmix_pyobj=NULL;
    struct PyGMixObject *gmix_obj=NULL;
    double ivar=0;

    double loglike=0, s2n_numer=0, s2n_denom=0;
    PyObject *tup=NULL;

    struct image *image=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOd", 
                                &image_obj, &gmix_pyobj, &ivar)) {
        return NULL;
    }

    if (!check_numpy_image(image_obj)) {
        PyErr_SetString(PyExc_IOError, "image input must be a 2D double PyArrayObject");
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    gmix_obj = (struct PyGMixObject *) gmix_pyobj;

    flags=calculate_loglike(image, gmix_obj->gmix, ivar, &s2n_numer, &s2n_denom, &loglike);

    // does not free underlying array
    image = image_free(image);

    tup = PyTuple_New(4);
    PyTuple_SetItem(tup, 0, PyFloat_FromDouble(loglike));
    PyTuple_SetItem(tup, 1, PyFloat_FromDouble(s2n_numer));
    PyTuple_SetItem(tup, 2, PyFloat_FromDouble(s2n_denom));
    PyTuple_SetItem(tup, 3, PyInt_FromLong((long)flags));

    return tup;
}


static PyObject *
PyGMixFit_loglike_jacob(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    double ivar=0;
    double dudrow, dudcol, dvdrow, dvdcol;
    double row0=0, col0=0;
    PyObject* gmix_pyobj=NULL;

    double loglike=0, s2n_numer=0, s2n_denom=0;
    PyObject *tup=NULL;

    struct image *image=NULL;
    struct jacobian jacob;
    struct PyGMixObject *gmix_obj=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OdddddddO", 
                &image_obj,
                &ivar,
                &dudrow, &dudcol, &dvdrow, &dvdcol,
                &row0,
                &col0,
                &gmix_pyobj)) {
        return NULL;
    }

    if (!check_numpy_image(image_obj)) {
        PyErr_SetString(PyExc_IOError, "image input must be a 2D double PyArrayObject");
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    gmix_obj = (struct PyGMixObject *) gmix_pyobj;

    jacobian_set(&jacob, row0, col0, dudrow, dudcol, dvdrow, dvdcol);

    flags=calculate_loglike_jacob(image,
                                  ivar,
                                  &jacob,
                                  gmix_obj->gmix,
                                  &s2n_numer,
                                  &s2n_denom,
                                  &loglike);

    // does not free underlying array
    image = image_free(image);

    tup = PyTuple_New(4);
    PyTuple_SetItem(tup, 0, PyFloat_FromDouble(loglike));
    PyTuple_SetItem(tup, 1, PyFloat_FromDouble(s2n_numer));
    PyTuple_SetItem(tup, 2, PyFloat_FromDouble(s2n_denom));
    PyTuple_SetItem(tup, 3, PyInt_FromLong((long)flags));

    return tup;
}



static PyObject *
PyGMixFit_loglike_wt_jacob(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    PyObject* weight_obj=NULL;
    double dudrow, dudcol, dvdrow, dvdcol;
    double row0=0, col0=0;
    PyObject* gmix_pyobj=NULL;

    double loglike=0, s2n_numer=0, s2n_denom=0;
    PyObject *tup=NULL;

    struct image *image=NULL;
    struct image *weight=NULL;
    struct jacobian jacob;
    struct PyGMixObject *gmix_obj=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOddddddO", 
                &image_obj,
                &weight_obj,
                &dudrow, &dudcol, &dvdrow, &dvdcol,
                &row0,
                &col0,
                &gmix_pyobj)) {
        return NULL;
    }

    if (!check_numpy_image(image_obj)) {
        PyErr_SetString(PyExc_IOError, "image input must be a 2D double PyArrayObject");
        return NULL;
    }
    if (!check_numpy_image(weight_obj)) {
        PyErr_SetString(PyExc_IOError, "weight input must be a 2D double PyArrayObject");
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);
    weight = associate_image(weight_obj, dims[0], dims[1]);

    gmix_obj = (struct PyGMixObject *) gmix_pyobj;

    jacobian_set(&jacob, row0, col0, dudrow, dudcol, dvdrow, dvdcol);

    flags=calculate_loglike_wt_jacob(image,
                                     weight,
                                     &jacob,
                                     gmix_obj->gmix,
                                     &s2n_numer,
                                     &s2n_denom,
                                     &loglike);

    // does not free underlying array
    image = image_free(image);
    weight = image_free(weight);

    tup = PyTuple_New(4);
    PyTuple_SetItem(tup, 0, PyFloat_FromDouble(loglike));
    PyTuple_SetItem(tup, 1, PyFloat_FromDouble(s2n_numer));
    PyTuple_SetItem(tup, 2, PyFloat_FromDouble(s2n_denom));
    PyTuple_SetItem(tup, 3, PyInt_FromLong((long)flags));

    return tup;
}









/* 
   This one can work on a subgrid.  There is not separate psf gmix, 
   you are expected to send a convolved GMix object in that case

   Simply add to the existing pixel values, so make sure you initialize

   No error checking on the gmix object is performed, do it in python!
*/

static PyObject *
PyGMixFit_fill_model(PyObject *self, PyObject *args) 
{
    PyObject *image_obj=NULL;
    PyObject *gmix_pyobj=NULL;
    struct PyGMixObject *gmix_obj=NULL;

    int nsub=0;

    struct image *image=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOi", &image_obj, &gmix_pyobj, &nsub)) {
        return NULL;
    }
    gmix_obj = (struct PyGMixObject *) gmix_pyobj;

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    flags=fill_model_subgrid(image, gmix_obj->gmix, nsub);

    // does not free underlying array
    image = image_free(image);

    return PyInt_FromLong(flags);
}


static PyObject *
PyGMixFit_fill_model_jacob(PyObject *self, PyObject *args) 
{
    PyObject *image_obj=NULL;
    PyObject *gmix_pyobj=NULL;
    double dudrow, dudcol, dvdrow, dvdcol;
    double row0=0, col0=0;
    struct jacobian jacob;
    struct PyGMixObject *gmix_obj=NULL;

    int nsub=0;

    struct image *image=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOddddddi",
                &image_obj, 
                &gmix_pyobj, 
                &dudrow, &dudcol, &dvdrow, &dvdcol,
                &row0,
                &col0,
                &nsub)) {
        return NULL;
    }
    gmix_obj = (struct PyGMixObject *) gmix_pyobj;

    jacobian_set(&jacob, row0, col0, dudrow, dudcol, dvdrow, dvdcol);

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    flags=fill_model_subgrid_jacob(image, gmix_obj->gmix, &jacob, nsub);

    // does not free underlying array
    image = image_free(image);

    return PyInt_FromLong(flags);
}



/* 
 * Generate gaussian random numbers mean 0 sigma 1
 *
 * we get two randoms for free but I'm only using one 
 */
 
/*
double randn() 
{
    double x1, x2, w, y1;//, y2;
 
    do {
        x1 = 2.*drand48() - 1.0;
        x2 = 2.*drand48() - 1.0;
        w = x1*x1 + x2*x2;
    } while ( w >= 1.0 );

    w = sqrt( (-2.*log( w ) ) / w );
    y1 = x1*w;
    //y2 = x2*w;
    return y1;
}
*/



static PyMethodDef render_module_methods[] = {
    {"fill_model", (PyCFunction)PyGMixFit_fill_model,  METH_VARARGS,  "fill the model image, possibly on a subgrid"},
    {"fill_model_jacob", (PyCFunction)PyGMixFit_fill_model_jacob,  METH_VARARGS,  "fill the model image, possibly on a subgrid"},
    {"fill_ydiff", (PyCFunction)PyGMixFit_fill_ydiff,  METH_VARARGS,  "fill diff from gmix"},
    {"fill_ydiff_jacob", (PyCFunction)PyGMixFit_fill_ydiff_jacob,  METH_VARARGS,  "fill diff with weight image and jacobian"},
    {"fill_ydiff_wt_jacob", (PyCFunction)PyGMixFit_fill_ydiff_wt_jacob,  METH_VARARGS,  "fill diff with weight image and jacobian"},

    {"loglike_margamp", (PyCFunction)PyGMixFit_loglike_margamp,  METH_VARARGS,  "calc logl, analytically marginalized over amplitude"},


    {"loglike", (PyCFunction)PyGMixFit_loglike,  METH_VARARGS,  "calc full log likelihood"},
    {"loglike_jacob", (PyCFunction)PyGMixFit_loglike_jacob,  METH_VARARGS,  "calc full log likelihood"},
    {"loglike_wt_jacob", (PyCFunction)PyGMixFit_loglike_wt_jacob,  METH_VARARGS,  "calc full log likelihood"},
    {NULL}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_render",      /* m_name */
        "Defines the some gmix fit methods",  /* m_doc */
        -1,                  /* m_size */
        render_module_methods,    /* m_methods */
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
init_render(void) 
{
    PyObject* m;

    //PyGMixType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    /*
    if (PyType_Ready(&PyGMixType) < 0) {
        return NULL;
    }
    */
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    /*
    if (PyType_Ready(&PyGMixType) < 0) {
        return;
    }
    */
    m = Py_InitModule3("_render", render_module_methods, 
            "This module gmix fit related routines.\n");
    if (m==NULL) {
        return;
    }
#endif

    //Py_INCREF(&PyGMixType);
    //PyModule_AddObject(m, "GMix", (PyObject *)&PyGMixType);
    import_array();
}
