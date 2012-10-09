#include <Python.h>
#include <numpy/arrayobject.h> 

#include "gvec.h"
#include "image.h"
#include "matrix.h"
#include "bound.h"
#include "defs.h"


static PyObject *
PyGMixFit_hello(void) {
    return PyString_FromString("hello world!");
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
int check_numpy_image(PyObject *obj)
{
    if (!PyArray_Check(obj) 
            || NPY_DOUBLE != PyArray_TYPE(obj)
            || 2 != PyArray_NDIM(obj)) {
        return 0;
    }
    return 1;
}
int check_numpy_array(PyObject *obj)
{
    if (!PyArray_Check(obj) || NPY_DOUBLE != PyArray_TYPE(obj)) {
        return 0;
    }
    return 1;
}

struct gvec *pyarray_to_gvec(PyObject *array)
{
    double *pars=NULL;
    int sz=0;
    struct gvec *gvec=NULL;
    pars = PyArray_DATA(array);
    sz = PyArray_SIZE(array);

    gvec = pars_to_gvec(pars, sz);
    return gvec;
}
struct gvec *coellip_pyarray_to_gvec(PyObject *array)
{
    double *pars=NULL;
    int sz=0;
    struct gvec *gvec=NULL;
    pars = PyArray_DATA(array);
    sz = PyArray_SIZE(array);

    gvec = coellip_pars_to_gvec(pars, sz);
    return gvec;
}
int check_image_and_diff(PyObject *image_obj, PyObject *diff_obj)
{
    if (!check_numpy_image(image_obj)) {
        PyErr_SetString(PyExc_IOError, "image input must be a 2D double PyArrayObject");
        return 0;
    }
    // only care that diff is a double array
    if (diff_obj != Py_None && !check_numpy_array(diff_obj)) {
        PyErr_SetString(PyExc_IOError, "diff image input must be a 2D double PyArrayObject");
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
   Pre-marginilized over the amplitude
         like is exp(0.5*A*B^2) where
         A is sum((model/err)^2) and is fixed
         and
           B = sum(model*image/err^2)/A
             = sum(model/err * image/err)/A
    
    The pre-marginalization may not work well
    at now S/N
*/

int calculate_loglike_old(struct image *image, 
                          struct gvec *obj_gvec, 
                          struct gvec *psf_gvec,
                          double A,
                          double ierr,
                          double *loglike)
{
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);

    struct gauss *gauss=NULL, *pgauss=NULL;
    double u=0, v=0, uv=0, u2=0, v2=0;
    double chi2=0, b=0;
    size_t i=0, j=0, col=0, row=0;
    double irr=0, irc=0, icc=0, det=0, psum=0;

    double model_val=0, tval=0;
    double ymodsum=0; // sum of (image/err)
    double ymod2sum=0; // sum of (image/err)^2
    double norm=0;
    double B=0.; // sum(model*image/err^2)/A
    int flags=0;

    *loglike=-9999.9e9;
    for (row=0; row<nrows; row++) {
        for (col=0; col<ncols; col++) {

            model_val=0;
            for (i=0; i<obj_gvec->size; i++) {
                gauss = &obj_gvec->data[i];
                if (gauss->det <= 0) {
                    DBG wlog("found det: %.16g\n", gauss->det);
                    flags |= GMIX_ERROR_NEGATIVE_DET;
                    goto _eval_model_bail;
                }

                u = row-gauss->row;
                v = col-gauss->col;

                u2 = u*u; v2 = v*v; uv = u*v;

                tval=0;
                if (psf_gvec) { 
                    psum=0;
                    for (j=0; j<psf_gvec->size; j++) {
                        pgauss=&psf_gvec->data[j];
                        irr = gauss->irr + pgauss->irr;
                        irc = gauss->irc + pgauss->irc;
                        icc = gauss->icc + pgauss->icc;
                        det = irr*icc - irc*irc;
                        if (det <= 0) {
                            DBG wlog("found convolved det: %.16g\n", det);
                            flags |= GMIX_ERROR_NEGATIVE_DET;
                            goto _eval_model_bail;
                        }
                        chi2=icc*u2 + irr*v2 - 2.0*irc*uv;
                        chi2 /= det;

                        b = M_TWO_PI*sqrt(det);
                        tval += pgauss->p*exp( -0.5*chi2 )/b;
                        psum += pgauss->p;
                    }
                    // psf always normalized to unity
                    tval *= gauss->p/psum;
                } else {
                    chi2=gauss->icc*u2 + gauss->irr*v2 - 2.0*gauss->irc*uv;
                    chi2 /= gauss->det;
                    b = M_TWO_PI*sqrt(gauss->det);
                    tval = gauss->p*exp( -0.5*chi2 )/b;
                }

                model_val += tval;
            } // gvec

            ymodsum += model_val;
            ymod2sum += model_val*model_val;
            B += IM_GET(image, row, col)*model_val;


        } // cols
    } // rows

    ymodsum *= ierr;
    ymod2sum *= ierr*ierr;
    norm = sqrt(ymodsum*ymodsum*A/ymod2sum);

    // renorm so A is fixed; also extra factor of 1/err^2 and 1/A
    B *= (norm/ymodsum*ierr*ierr/A);

    *loglike = 0.5*A*B*B;


_eval_model_bail:
    return flags;
}

int calculate_loglike(struct image *image, 
                      struct gvec *gvec, 
                      double A,
                      double ierr,
                      double *loglike)
{
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);

    struct gauss *gauss=NULL;
    double u=0, v=0, uv=0, u2=0, v2=0;
    double chi2=0, b=0;
    size_t i=0, col=0, row=0;

    double model_val=0;
    double ymodsum=0; // sum of (image/err)
    double ymod2sum=0; // sum of (image/err)^2
    double norm=0;
    double B=0.; // sum(model*image/err^2)/A
    int flags=0;

    *loglike=-9999.9e9;
    for (row=0; row<nrows; row++) {
        for (col=0; col<ncols; col++) {

            model_val=0;
            gauss=gvec->data;
            for (i=0; i<gvec->size; i++) {

                if (gauss->det <= 0) {
                    DBG wlog("found det: %.16g\n", gauss->det);
                    flags |= GMIX_ERROR_NEGATIVE_DET;
                    goto _eval_model_bail;
                }

                u = row-gauss->row;
                v = col-gauss->col;

                u2 = u*u; v2 = v*v; uv = u*v;

                chi2=gauss->icc*u2 + gauss->irr*v2 - 2.0*gauss->irc*uv;
                chi2 /= gauss->det;
                b = M_TWO_PI*sqrt(gauss->det);

                model_val += gauss->p*exp( -0.5*chi2 )/b;

                gauss++;
            } // gvec

            ymodsum += model_val;
            ymod2sum += model_val*model_val;
            B += IM_GET(image, row, col)*model_val;

        } // cols
    } // rows

    ymodsum *= ierr;
    ymod2sum *= ierr*ierr;
    norm = sqrt(ymodsum*ymodsum*A/ymod2sum);

    // renorm so A is fixed; also extra factor of 1/err^2 and 1/A
    B *= (norm/ymodsum*ierr*ierr/A);

    *loglike = 0.5*A*B*B;

_eval_model_bail:
    return flags;
}



/*
 * If diff is NULL, we fill in image.
 * If diff is not NULL, we fill in diff with model-image
 */
int fill_model(struct image *image, 
               struct gvec *obj_gvec, 
               struct gvec *psf_gvec,
               struct image *diff)
{
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);

    struct gauss *gauss=NULL, *pgauss=NULL;
    double u=0, v=0, uv=0, u2=0, v2=0;
    double chi2=0, b=0;
    size_t i=0, j=0, col=0, row=0;
    double irr=0, irc=0, icc=0, det=0, psum=0;

    double val=0, tval=0;
    int flags=0;

    int do_diff=0;
    if (diff)
        do_diff=1;

    for (row=0; row<nrows; row++) {
        for (col=0; col<ncols; col++) {

            val=0;
            for (i=0; i<obj_gvec->size; i++) {
                gauss = &obj_gvec->data[i];
                if (gauss->det <= 0) {
                    DBG wlog("found det: %.16g\n", gauss->det);
                    flags |= GMIX_ERROR_NEGATIVE_DET;
                    goto _eval_model_bail;
                }

                u = row-gauss->row;
                v = col-gauss->col;

                u2 = u*u; v2 = v*v; uv = u*v;

                tval=0;
                if (psf_gvec) { 
                    psum=0;
                    for (j=0; j<psf_gvec->size; j++) {
                        pgauss=&psf_gvec->data[j];
                        irr = gauss->irr + pgauss->irr;
                        irc = gauss->irc + pgauss->irc;
                        icc = gauss->icc + pgauss->icc;
                        det = irr*icc - irc*irc;
                        if (det <= 0) {
                            DBG wlog("found convolved det: %.16g\n", det);
                            flags |= GMIX_ERROR_NEGATIVE_DET;
                            goto _eval_model_bail;
                        }
                        chi2=icc*u2 + irr*v2 - 2.0*irc*uv;
                        chi2 /= det;

                        b = M_TWO_PI*sqrt(det);
                        tval += pgauss->p*exp( -0.5*chi2 )/b;
                        psum += pgauss->p;
                    }
                    // psf always normalized to unity
                    tval *= gauss->p/psum;
                } else {
                    chi2=gauss->icc*u2 + gauss->irr*v2 - 2.0*gauss->irc*uv;
                    chi2 /= gauss->det;
                    b = M_TWO_PI*sqrt(gauss->det);
                    tval = gauss->p*exp( -0.5*chi2 )/b;
                }

                val += tval;
            } // gvec

            if (do_diff) {
                tval = IM_GET(image, row, col);
                IM_SETFAST(diff, row, col, val-tval);
            } else {
                IM_SETFAST(image, row, col, val);
            }

        } // cols
    } // rows

_eval_model_bail:
    return flags;
}


/*
 * Note the diff object can actually have padding at the end
 * that will contain priors, so don't try to grab it's dimensions
 */
static PyObject *
PyGMixFit_coellip_fill_model(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    PyObject* diff_obj=NULL;
    PyObject* obj_pars_obj=NULL;
    PyObject* psf_pars_obj=NULL; // Can be None

    struct image *image=NULL;
    struct gvec *obj_gvec=NULL;
    struct gvec *psf_gvec=NULL;
    struct image *diff=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOOO", &image_obj, &obj_pars_obj, &psf_pars_obj, &diff_obj)) {
        return NULL;
    }

    if (!check_image_and_diff(image_obj,diff_obj)) {
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    if (diff_obj != Py_None) {
        diff = associate_image(diff_obj, dims[0], dims[1]);
    }

    obj_gvec = coellip_pyarray_to_gvec(obj_pars_obj);
    DBG2 gvec_print(obj_gvec, stderr);

    if (psf_pars_obj != Py_None) {
        // always use full gmix for psf
        psf_gvec = pyarray_to_gvec(psf_pars_obj);
        DBG2 gvec_print(psf_gvec, stderr);
    }

    flags=fill_model(image, obj_gvec, psf_gvec, diff);

    obj_gvec = gvec_free(obj_gvec);
    psf_gvec = gvec_free(psf_gvec);
    // does not free underlying array
    image = image_free(image);
    diff = image_free(diff);

    return PyInt_FromLong(flags);
}

static PyObject *
PyGMixFit_loglike_coellip(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    PyObject* obj_pars_obj=NULL;
    PyObject* psf_pars_obj=NULL; // Can be None
    double A=0, ierr=0;

    double loglike=0;
    PyObject *tup=NULL;

    struct image *image=NULL;
    struct gvec *obj_gvec=NULL;
    struct gvec *psf_gvec=NULL;
    struct gvec *gvec=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOOdd", 
                          &image_obj, &obj_pars_obj, &psf_pars_obj,
                          &A, &ierr)) {
        return NULL;
    }

    if (!check_numpy_image(image_obj)) {
        PyErr_SetString(PyExc_IOError, "image input must be a 2D double PyArrayObject");
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    obj_gvec = coellip_pyarray_to_gvec(obj_pars_obj);
    DBG2 gvec_print(obj_gvec, stderr);
    psf_gvec = pyarray_to_gvec(psf_pars_obj);
    DBG2 gvec_print(psf_gvec, stderr);

    gvec = gvec_convolve(obj_gvec, psf_gvec);

    flags=calculate_loglike(image, gvec, A, ierr, &loglike);

    obj_gvec = gvec_free(obj_gvec);
    psf_gvec = gvec_free(psf_gvec);
    gvec     = gvec_free(gvec);

    // does not free underlying array
    image = image_free(image);

    tup = PyTuple_New(2);
    PyTuple_SetItem(tup, 0, PyFloat_FromDouble(loglike));
    PyTuple_SetItem(tup, 1, PyInt_FromLong((long)flags));

    return tup;
}


static PyObject *
PyGMixFit_loglike_coellip_old(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    PyObject* obj_pars_obj=NULL;
    PyObject* psf_pars_obj=NULL; // Can be None
    double A=0, ierr=0;

    double loglike=0;
    PyObject *tup=NULL;

    struct image *image=NULL;
    struct gvec *obj_gvec=NULL;
    struct gvec *psf_gvec=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOOdd", 
                          &image_obj, &obj_pars_obj, &psf_pars_obj,
                          &A, &ierr)) {
        return NULL;
    }

    if (!check_numpy_image(image_obj)) {
        PyErr_SetString(PyExc_IOError, "image input must be a 2D double PyArrayObject");
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    obj_gvec = coellip_pyarray_to_gvec(obj_pars_obj);
    DBG2 gvec_print(obj_gvec, stderr);

    if (psf_pars_obj != Py_None) {
        // always use full gmix for psf
        psf_gvec = pyarray_to_gvec(psf_pars_obj);
        DBG2 gvec_print(psf_gvec, stderr);
    }

    flags=calculate_loglike_old(image, obj_gvec, psf_gvec, A, ierr, &loglike);

    obj_gvec = gvec_free(obj_gvec);
    psf_gvec = gvec_free(psf_gvec);
    // does not free underlying array
    image = image_free(image);

    tup = PyTuple_New(2);
    PyTuple_SetItem(tup, 0, PyFloat_FromDouble(loglike));
    PyTuple_SetItem(tup, 1, PyInt_FromLong((long)flags));

    return tup;
}



/*
 * The pars are full gaussian mixtures.
 *
 * Note the diff object can actually have padding at the end
 * that will contain priors, so don't try to grab it's dimensions
 */
static PyObject *
PyGMixFit_fill_model(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    PyObject* diff_obj=NULL;
    PyObject* obj_pars_obj=NULL;
    PyObject* psf_pars_obj=NULL; // Can be None

    struct image *image=NULL;
    struct gvec *obj_gvec=NULL;
    struct gvec *psf_gvec=NULL;
    struct image *diff=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOOO", &image_obj, &obj_pars_obj, &psf_pars_obj, &diff_obj)) {
        return NULL;
    }

    if (!check_image_and_diff(image_obj,diff_obj)) {
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    if (diff_obj != Py_None) {
        diff = associate_image(diff_obj, dims[0], dims[1]);
    }

    obj_gvec = pyarray_to_gvec(obj_pars_obj);
    DBG2 gvec_print(obj_gvec, stderr);

    if (psf_pars_obj != Py_None) {
        psf_gvec = pyarray_to_gvec(psf_pars_obj);
        DBG2 gvec_print(psf_gvec, stderr);
    }

    flags=fill_model(image, obj_gvec, psf_gvec, diff);

    obj_gvec = gvec_free(obj_gvec);
    psf_gvec = gvec_free(psf_gvec);
    // does not free underlying array
    image = image_free(image);
    diff = image_free(diff);

    return PyInt_FromLong(flags);
}




static PyMethodDef render_module_methods[] = {
    {"hello",      (PyCFunction)PyGMixFit_hello,               METH_NOARGS,  "test hello"},
    {"fill_model_coellip", (PyCFunction)PyGMixFit_coellip_fill_model,  METH_VARARGS,  "fill the model image"},
    {"fill_model", (PyCFunction)PyGMixFit_fill_model,  METH_VARARGS,  "fill the model image"},
    {"loglike_coellip_old", (PyCFunction)PyGMixFit_loglike_coellip_old,  METH_VARARGS,  "calc logl, analytically marginalized over amplitude"},
    {"loglike_coellip", (PyCFunction)PyGMixFit_loglike_coellip,  METH_VARARGS,  "calc logl, analytically marginalized over amplitude"},
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


    //PyGMixEMObjectType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    /*
    if (PyType_Ready(&PyGMixEMObjectType) < 0) {
        return NULL;
    }
    */
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    /*
    if (PyType_Ready(&PyGMixEMObjectType) < 0) {
        return;
    }
    */
    m = Py_InitModule3("_render", render_module_methods, 
            "This module gmix fit related routines.\n");
    if (m==NULL) {
        return;
    }
#endif

    /*
    Py_INCREF(&PyGMixEMObjectType);
    PyModule_AddObject(m, "GMixEM", (PyObject *)&PyGMixEMObjectType);
    */

    import_array();
}
