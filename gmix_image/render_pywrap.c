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

// pars are full gmix of size 6*ngauss
struct gvec *pars_to_gvec(PyObject *array)
{
    npy_intp sz=0;
    npy_intp ngauss=0;
    double *pars=NULL;
    struct gauss *gauss=NULL;

    int i=0, beg=0;

    pars = PyArray_DATA(array);
    sz = PyArray_SIZE(array);

    ngauss = sz/6;

    struct gvec *gvec = gvec_new(ngauss);


    for (i=0; i<ngauss; i++) {
        beg = i*6;

        gauss = &gvec->data[i];

        gauss->p   = pars[beg+0];
        gauss->row = pars[beg+1];
        gauss->col = pars[beg+2];
        gauss->irr = pars[beg+3];
        gauss->irc = pars[beg+4];
        gauss->icc = pars[beg+5];
    }

    gvec_set_dets(gvec);
    return gvec;
}



struct gvec *coellip_pars_to_gvec(PyObject *array)
{
    npy_intp sz=0;
    npy_intp ngauss=0;
    double row=0, col=0, e1=0, e2=0, Tmax=0, Ti=0, pi=0, Tfrac=0;
    double *pars=NULL;
    struct gauss *gauss=NULL;

    int i=0;

    pars = PyArray_DATA(array);
    sz = PyArray_SIZE(array);

    ngauss = (sz-4)/2;

    struct gvec * gvec = gvec_new(ngauss);

    row=pars[0];
    col=pars[1];
    e1 = pars[2];
    e2 = pars[3];
    Tmax = pars[4];

    for (i=0; i<ngauss; i++) {
        gauss = &gvec->data[i];

        if (i==0) {
            Ti = Tmax;
        } else {
            Tfrac = pars[4+i];
            Ti = Tmax*Tfrac;
        }

        pi = pars[4+ngauss+i];

        gauss->p = pi;
        gauss->row = row;
        gauss->col = col;

        gauss->irr = (Ti/2.)*(1-e1);
        gauss->irc = (Ti/2.)*e2;
        gauss->icc = (Ti/2.)*(1+e1);
    }

    gvec_set_dets(gvec);
    return gvec;
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

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    if (diff_obj != Py_None) {
        diff = associate_image(diff_obj, dims[0], dims[1]);
    }

    obj_gvec = coellip_pars_to_gvec(obj_pars_obj);
    DBG gvec_print(obj_gvec, stderr);

    if (psf_pars_obj != Py_None) {
        psf_gvec = coellip_pars_to_gvec(psf_pars_obj);
        DBG gvec_print(psf_gvec, stderr);
    }

    flags=fill_model(image, obj_gvec, psf_gvec, diff);

    obj_gvec = gvec_free(obj_gvec);
    psf_gvec = gvec_free(psf_gvec);
    // does not free underlying array
    image = image_free(image);
    diff = image_free(diff);

    return PyInt_FromLong(flags);
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

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    if (diff_obj != Py_None) {
        diff = associate_image(diff_obj, dims[0], dims[1]);
    }

    obj_gvec = pars_to_gvec(obj_pars_obj);
    DBG gvec_print(obj_gvec, stderr);

    if (psf_pars_obj != Py_None) {
        psf_gvec = pars_to_gvec(psf_pars_obj);
        DBG gvec_print(psf_gvec, stderr);
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