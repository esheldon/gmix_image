#include <Python.h>
#include <numpy/arrayobject.h> 

#include "gvec.h"
#include "image.h"
#include "bound.h"
#include "defs.h"

/*

   Gaussian Mixtures

   This code defines the python gaussian mixture object

*/

struct PyGVecObject {
    PyObject_HEAD
    struct gvec *gvec;
};


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

    if ((sz % 6) != 0) {
        PyErr_Format(PyExc_ValueError, 
                "gmix pars size not multiple of 6: %d\n", sz);
        return NULL;
    }
    gvec = gvec_from_pars(pars, sz);
    return gvec;
}
struct gvec *coellip_pyarray_to_gvec(PyObject *array)
{
    double *pars=NULL;
    int sz=0;
    struct gvec *gvec=NULL;
    pars = PyArray_DATA(array);
    sz = PyArray_SIZE(array);

    gvec = gvec_from_coellip(pars, sz);
    return gvec;
}


/*
   type
     0 full pars [pi,rowi,coli,irri,irci,icci]
     1 coellip pars
     2 exp (same layout as coellip for one gauss)
     3 dev (same layout as coellip for one gauss)
     4 turb (same layout as coellip for one gauss)
*/


static int
PyGVecObject_init(struct PyGVecObject* self, PyObject *args)
{
    int type=0;
    PyObject *pars_obj=NULL;

    npy_intp size=0;
    double *pars=NULL;

    self->gvec=NULL;

    if (!PyArg_ParseTuple(args, (char*)"iO", &type, &pars_obj)) {
        return -1;
    }

    if (!check_numpy_array(pars_obj)) {
        PyErr_SetString(PyExc_ValueError, "pars must be an array");
        return -1;
    }
    pars = PyArray_DATA(pars_obj);
    size = PyArray_SIZE(pars_obj);

    switch (type) {
        case 0:
            self->gvec = gvec_from_pars(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "full pars size not multiple of 6: %ld", size);
            }
            break;
        case 1:
            self->gvec = gvec_from_coellip(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "coellip pars wrong size: %ld", size);
            }
            break;
        case 2:
            self->gvec = gvec_from_pars_exp(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "exp pars not size 6: %ld", size);
            }
            break;
        case 3:
            self->gvec = gvec_from_pars_dev(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "dev pars not size 6: %ld", size);
            }
            break;
        case 4:
            self->gvec = gvec_from_pars_turb(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "turb pars not size 6: %ld", size);
            }
            break;

        case 5:
            self->gvec = gvec_from_pars_dev6(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "dev6 pars not size 6: %ld", size);
            }
            break;
        case 6:
            self->gvec = gvec_from_pars_dev10(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "dev10 pars not size 6: %ld", size);
            }
            break;




        default:
            PyErr_Format(PyExc_ValueError, "bad pars type value: %d", type);
            return -1;
    }

    return 0;

}
static void
PyGVecObject_dealloc(struct PyGVecObject* self)
{
    self->gvec = gvec_free(self->gvec);
#if PY_MAJOR_VERSION >= 3
    // introduced in python 2.6
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    // old way, removed in python 3
    self->ob_type->tp_free((PyObject*)self);
#endif
}

static PyObject *
PyGVecObject_repr(struct PyGVecObject* self) {
    return PyString_FromString("GVec");
}

static void add_double_to_dict(PyObject* dict, const char* key, double value) 
{
    PyObject* tobj=NULL;
    tobj=PyFloat_FromDouble(value);
    PyDict_SetItemString(dict, key, tobj);
    Py_XDECREF(tobj);
}

static PyObject *PyGVecObject_get_dlist(struct PyGVecObject* self)
{
    PyObject *list=NULL;
    PyObject *dict=NULL;
    struct gauss *gauss=NULL;
    int i=0;

    list=PyList_New(self->gvec->size);
    gauss=self->gvec->data;
    for (i=0; i<self->gvec->size; i++) {
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
static PyObject *PyGVecObject_get_T(struct PyGVecObject* self)
{
    double T=0;
    double psum=0;
    size_t i=0;
    struct gauss *gauss=NULL;

    gauss=self->gvec->data;
    for (i=0; i<self->gvec->size; i++) {
        T += (gauss->irr + gauss->icc)*gauss->p;
        psum += gauss->p;
        gauss++;
    }

    T /= psum;

    return PyFloat_FromDouble(T);
}
static PyObject *PyGVecObject_get_cen(struct PyGVecObject* self)
{
    double psum=0;
    size_t i=0;
    struct gauss *gauss=NULL;
    npy_intp dims[1] = {2};
    int npy_dtype=NPY_FLOAT64;
    double row=0, col=0;
    PyObject *cen=NULL;
    double *cendata=NULL;

    cen=PyArray_ZEROS(1, dims, npy_dtype, 0);
    cendata=PyArray_DATA(cen);

    gauss=self->gvec->data;
    for (i=0; i<self->gvec->size; i++) {
        row += gauss->row*gauss->p;
        col += gauss->col*gauss->p;
        psum += gauss->p;
        gauss++;
    }

    row /= psum;
    col /= psum;

    cendata[0] = row;
    cendata[1] = col;

    return cen;
}
static PyObject *PyGVecObject_set_cen(struct PyGVecObject* self, PyObject *args)
{
    double row=0, col=0;
    size_t i=0;
    struct gauss *gauss=NULL;

    if (!PyArg_ParseTuple(args, (char*)"dd", &row, &col)) {
        return NULL;
    }

    gauss=self->gvec->data;
    for (i=0; i<self->gvec->size; i++) {
        gauss->row=row;
        gauss->col=col;
        gauss++;
    }

    Py_XINCREF(Py_None);
    return Py_None;
}

static PyObject *PyGVecObject_get_e1e2T(struct PyGVecObject* self)
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

    gauss=self->gvec->data;
    for (i=0; i<self->gvec->size; i++) {
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
static PyObject *PyGVecObject_convolve_inplace(struct PyGVecObject* self, PyObject *args)
{
    PyObject *psf_obj=NULL;
    struct PyGVecObject *psf=NULL;
    struct gvec *gvec_new=NULL;

    if (!PyArg_ParseTuple(args, (char*)"O", &psf_obj)) {
        return NULL;
    }

    psf=(struct PyGVecObject *) psf_obj;

    gvec_new = gvec_convolve(self->gvec, psf->gvec);
    self->gvec = gvec_free(self->gvec);

    self->gvec = gvec_new;

    Py_XINCREF(Py_None);
    return Py_None;
}

static PyMethodDef PyGVecObject_methods[] = {
    {"get_dlist", (PyCFunction)PyGVecObject_get_dlist, METH_NOARGS, "get_dlist\n\nreturn list of dicts."},
    {"get_e1e2T", (PyCFunction)PyGVecObject_get_e1e2T, METH_NOARGS, "get_e1e2T\n\nreturn stats based on average moments val=sum(val_i*p)/sum(p)."},
    {"get_T", (PyCFunction)PyGVecObject_get_T, METH_NOARGS, "get_T\n\nreturn T=sum(T_i*p)/sum(p)."},
    {"get_cen", (PyCFunction)PyGVecObject_get_cen, METH_NOARGS, "get_cen\n\nreturn cen=sum(cen_i*p)/sum(p)."},
    {"set_cen", (PyCFunction)PyGVecObject_set_cen, METH_VARARGS, "set_cen\n\nSet all centers to the input row,col"},
    {"_convolve_inplace", (PyCFunction)PyGVecObject_convolve_inplace, METH_VARARGS, "convolve_inplace\n\nConvolve with the psf in place."},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyGVecType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "_gvec.GVec",             /*tp_name*/
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
    "GVecIO Class",           /* tp_doc */
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
    //PyGVecObject_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};






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

int check_image_and_diff(PyObject *image_obj, PyObject *diff_obj)
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
   fill a model with a gaussian mixture.  The model can be
   on a sub-grid (n > 1)

 */
static int
fill_model_subgrid(struct image *image, 
                   struct gvec *gvec, 
                   int nsub)
{
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);

    struct gauss *gauss=NULL;
    double u=0, v=0, uv=0, u2=0, v2=0;
    double chi2=0, b=0;
    size_t i=0, col=0, row=0, irowsub=0, icolsub=0;

    double model_val=0, tval=0;
    double stepsize=0, offset=0, trow=0, tcol=0;
    int flags=0;

    if (!gvec_verify(gvec)) {
        flags |= GMIX_ERROR_NEGATIVE_DET;
        goto _fill_model_subgrid_bail;
    }
    if (nsub < 1)
        nsub=1;
    stepsize = 1./nsub;
    offset = (nsub-1)*stepsize/2.;

    for (row=0; row<nrows; row++) {
        for (col=0; col<ncols; col++) {

            model_val=0;
            gauss=gvec->data;
            for (i=0; i<gvec->size; i++) {

                // work over the subgrid
                tval=0;
                trow = row-offset;
                for (irowsub=0; irowsub<nsub; irowsub++) {
                    tcol = col-offset;
                    for (icolsub=0; icolsub<nsub; icolsub++) {

                        u = trow-gauss->row;
                        v = tcol-gauss->col;

                        u2 = u*u; v2 = v*v; uv = u*v;

                        chi2=gauss->icc*u2 + gauss->irr*v2 - 2.0*gauss->irc*uv;
                        chi2 /= gauss->det;
                        tval += gauss->p*exp( -0.5*chi2 );

                        tcol += stepsize;
                    }
                    trow += stepsize;
                }

                b = M_TWO_PI*sqrt(gauss->det);
                tval /= (b*nsub*nsub);
                model_val += tval;

                gauss++;
            } // gvec

            IM_SETFAST(image, row, col, model_val);

        } // cols
    } // rows

_fill_model_subgrid_bail:
    return flags;
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

int calculate_loglike_old_old(struct image *image, 
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

int calculate_loglike_margamp(struct image *image, 
                              struct gvec *gvec, 
                              double A,
                              double ierr,
                              double *loglike)
{
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);

    struct gauss *gauss=NULL;
    double u=0, v=0, uv=0, u2=0, v2=0;
    double chi2=0;
    size_t i=0, col=0, row=0;

    double model_val=0;
    double ymodsum=0; // sum of (image/err)
    double ymod2sum=0; // sum of (image/err)^2
    double norm=0;
    double B=0.; // sum(model*image/err^2)/A
    double *rowdata=NULL;
    int flags=0;

    if (!gvec_verify(gvec)) {
        flags |= GMIX_ERROR_NEGATIVE_DET;
        goto _calculate_loglike_bail;
    }


    *loglike=-9999.9e9;
    for (row=0; row<nrows; row++) {
        rowdata=IM_ROW(image, row);
        for (col=0; col<ncols; col++) {

            model_val=0;
            gauss=gvec->data;
            for (i=0; i<gvec->size; i++) {
                u = row-gauss->row;
                u2=u*u;

                v = col-gauss->col;
                v2 = v*v;
                uv = u*v;

                chi2=gauss->dcc*u2 + gauss->drr*v2 - 2.0*gauss->drc*uv;
                model_val += gauss->norm*gauss->p*exp( -0.5*chi2 );

                gauss++;
            } // gvec

            ymodsum += model_val;
            ymod2sum += model_val*model_val;
            B += (*rowdata)*model_val;

            rowdata++;
        } // cols
    } // rows

    ymodsum *= ierr;
    ymod2sum *= ierr*ierr;
    norm = sqrt(ymodsum*ymodsum*A/ymod2sum);

    // renorm so A is fixed; also extra factor of 1/err^2 and 1/A
    B *= (norm/ymodsum*ierr*ierr/A);

    *loglike = 0.5*A*B*B;

    //fprintf(stderr,"OK faster\n");
_calculate_loglike_bail:
    return flags;
}

int calculate_loglike(struct image *image, 
                      struct gvec *gvec, 
                      double ivar,
                      double *loglike)
{
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);

    struct gauss *gauss=NULL;
    double u=0, v=0;
    double chi2=0, diff=0;
    size_t i=0, col=0, row=0;

    double model_val=0;
    double *rowdata=NULL;
    int flags=0;

    if (!gvec_verify(gvec)) {
        flags |= GMIX_ERROR_NEGATIVE_DET;
        goto _calculate_loglike_bail;
    }


    *loglike=-9999.9e9;
    for (row=0; row<nrows; row++) {
        rowdata=IM_ROW(image, row);
        for (col=0; col<ncols; col++) {

            model_val=0;
            gauss=gvec->data;
            for (i=0; i<gvec->size; i++) {
                u = row-gauss->row;
                v = col-gauss->col;

                chi2=gauss->dcc*u*u + gauss->drr*v*v - 2.0*gauss->drc*u*v;
                model_val += gauss->norm*gauss->p*exp( -0.5*chi2 );

                gauss++;
            } // gvec

            diff = model_val -(*rowdata);
            (*loglike) += diff*diff*ivar;

            rowdata++;
        } // cols
    } // rows

    (*loglike) *= (-0.5);

    //fprintf(stderr,"OK faster\n");
_calculate_loglike_bail:
    return flags;
}





int calculate_loglike_old(struct image *image, 
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

    if (!gvec_verify(gvec)) {
        flags |= GMIX_ERROR_NEGATIVE_DET;
        goto _calculate_loglike_bail;
    }

    *loglike=-9999.9e9;
    for (row=0; row<nrows; row++) {
        for (col=0; col<ncols; col++) {

            model_val=0;
            gauss=gvec->data;
            for (i=0; i<gvec->size; i++) {

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

_calculate_loglike_bail:
    return flags;
}



/*
 * If diff is NULL, we fill in image.
 * If diff is not NULL, we fill in diff with model-image
 */
int fill_model_old(struct image *image, 
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
PyGMixFit_coellip_fill_model_old(PyObject *self, PyObject *args) 
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

    if (!PyArg_ParseTuple(args, (char*)"OOOO", &image_obj, &obj_pars_obj, 
                &psf_pars_obj, &diff_obj)) {
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

    flags=fill_model_old(image, obj_gvec, psf_gvec, diff);

    obj_gvec = gvec_free(obj_gvec);
    psf_gvec = gvec_free(psf_gvec);
    // does not free underlying array
    image = image_free(image);
    diff = image_free(diff);

    return PyInt_FromLong(flags);
}


static PyObject *
PyGMixFit_fill_ydiff_dev6(PyObject *self, PyObject *args) 
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
    double *pars=NULL;
    size_t npars=0;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOOO", 
                &image_obj, &obj_pars_obj, &psf_pars_obj, &diff_obj)) {
        return NULL;
    }

    if (!check_image_and_diff(image_obj,diff_obj)) {
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);
    diff = associate_image(diff_obj, dims[0], dims[1]);

    if (!check_numpy_array(obj_pars_obj) || PyArray_SIZE(obj_pars_obj) != 6) {
        PyErr_SetString(PyExc_ValueError, "pars must be double array of length 6: %d\n");
        return NULL;
    }
    pars = PyArray_DATA(obj_pars_obj);
    npars = PyArray_SIZE(obj_pars_obj);
    obj_gvec = gvec_from_pars_dev6(pars,npars);
    DBG2 gvec_print(obj_gvec, stderr);

    // always use full gmix for psf
    psf_gvec = pyarray_to_gvec(psf_pars_obj);
    DBG2 gvec_print(psf_gvec, stderr);

    flags=fill_model_old(image, obj_gvec, psf_gvec, diff);

    obj_gvec = gvec_free(obj_gvec);
    psf_gvec = gvec_free(psf_gvec);
    // does not free underlying array
    image = image_free(image);
    diff = image_free(diff);

    return PyInt_FromLong(flags);
}


static PyObject *
PyGMixFit_fill_ydiff_dev10(PyObject *self, PyObject *args) 
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
    double *pars=NULL;
    size_t npars=0;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOOO", 
                &image_obj, &obj_pars_obj, &psf_pars_obj, &diff_obj)) {
        return NULL;
    }

    if (!check_image_and_diff(image_obj,diff_obj)) {
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);
    diff = associate_image(diff_obj, dims[0], dims[1]);

    if (!check_numpy_array(obj_pars_obj) || PyArray_SIZE(obj_pars_obj) != 6) {
        PyErr_SetString(PyExc_ValueError, "pars must be double array of length 6: %d\n");
        return NULL;
    }
    pars = PyArray_DATA(obj_pars_obj);
    npars = PyArray_SIZE(obj_pars_obj);
    obj_gvec = gvec_from_pars_dev10(pars,npars);
    DBG2 gvec_print(obj_gvec, stderr);

    // always use full gmix for psf
    psf_gvec = pyarray_to_gvec(psf_pars_obj);
    DBG2 gvec_print(psf_gvec, stderr);

    flags=fill_model_old(image, obj_gvec, psf_gvec, diff);

    obj_gvec = gvec_free(obj_gvec);
    psf_gvec = gvec_free(psf_gvec);
    // does not free underlying array
    image = image_free(image);
    diff = image_free(diff);

    return PyInt_FromLong(flags);
}




static PyObject *
PyGMixFit_fill_ydiff_dev(PyObject *self, PyObject *args) 
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
    double *pars=NULL;
    size_t npars=0;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOOO", 
                &image_obj, &obj_pars_obj, &psf_pars_obj, &diff_obj)) {
        return NULL;
    }

    if (!check_image_and_diff(image_obj,diff_obj)) {
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);
    diff = associate_image(diff_obj, dims[0], dims[1]);

    if (!check_numpy_array(obj_pars_obj) || PyArray_SIZE(obj_pars_obj) != 6) {
        PyErr_SetString(PyExc_ValueError, "pars must be double array of length 6: %d\n");
        return NULL;
    }
    pars = PyArray_DATA(obj_pars_obj);
    npars = PyArray_SIZE(obj_pars_obj);
    obj_gvec = gvec_from_pars_dev(pars,npars);
    DBG2 gvec_print(obj_gvec, stderr);

    // always use full gmix for psf
    psf_gvec = pyarray_to_gvec(psf_pars_obj);
    DBG2 gvec_print(psf_gvec, stderr);

    flags=fill_model_old(image, obj_gvec, psf_gvec, diff);

    obj_gvec = gvec_free(obj_gvec);
    psf_gvec = gvec_free(psf_gvec);
    // does not free underlying array
    image = image_free(image);
    diff = image_free(diff);

    return PyInt_FromLong(flags);
}



static PyObject *
PyGMixFit_fill_ydiff_dev_galsim(PyObject *self, PyObject *args) 
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
    double *pars=NULL;
    size_t npars=0;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOOO", 
                &image_obj, &obj_pars_obj, &psf_pars_obj, &diff_obj)) {
        return NULL;
    }

    if (!check_image_and_diff(image_obj,diff_obj)) {
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);
    diff = associate_image(diff_obj, dims[0], dims[1]);

    if (!check_numpy_array(obj_pars_obj) || PyArray_SIZE(obj_pars_obj) != 6) {
        PyErr_SetString(PyExc_ValueError, "pars must be double array of length 6: %d\n");
        return NULL;
    }
    pars = PyArray_DATA(obj_pars_obj);
    npars = PyArray_SIZE(obj_pars_obj);
    obj_gvec = gvec_from_pars_dev_galsim(pars,npars);
    DBG2 gvec_print(obj_gvec, stderr);

    // always use full gmix for psf
    psf_gvec = pyarray_to_gvec(psf_pars_obj);
    DBG2 gvec_print(psf_gvec, stderr);

    flags=fill_model_old(image, obj_gvec, psf_gvec, diff);

    obj_gvec = gvec_free(obj_gvec);
    psf_gvec = gvec_free(psf_gvec);
    // does not free underlying array
    image = image_free(image);
    diff = image_free(diff);

    return PyInt_FromLong(flags);
}

static PyObject *
PyGMixFit_fill_ydiff_exp(PyObject *self, PyObject *args) 
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
    double *pars=NULL;
    size_t npars=0;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOOO", 
                &image_obj, &obj_pars_obj, &psf_pars_obj, &diff_obj)) {
        return NULL;
    }

    if (!check_image_and_diff(image_obj,diff_obj)) {
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);
    diff = associate_image(diff_obj, dims[0], dims[1]);

    if (!check_numpy_array(obj_pars_obj) || PyArray_SIZE(obj_pars_obj) != 6) {
        PyErr_SetString(PyExc_ValueError, "pars must be double array of length 6: %d\n");
        return NULL;
    }
    pars = PyArray_DATA(obj_pars_obj);
    npars = PyArray_SIZE(obj_pars_obj);
    obj_gvec = gvec_from_pars_exp(pars,npars);
    DBG2 gvec_print(obj_gvec, stderr);

    // always use full gmix for psf
    psf_gvec = pyarray_to_gvec(psf_pars_obj);
    DBG2 gvec_print(psf_gvec, stderr);

    flags=fill_model_old(image, obj_gvec, psf_gvec, diff);

    obj_gvec = gvec_free(obj_gvec);
    psf_gvec = gvec_free(psf_gvec);
    // does not free underlying array
    image = image_free(image);
    diff = image_free(diff);

    return PyInt_FromLong(flags);
}



static PyObject *
PyGMixFit_loglike_coellip_margamp(PyObject *self, PyObject *args) 
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

    flags=calculate_loglike_old(image, gvec, A, ierr, &loglike);

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
PyGMixFit_loglike_margamp(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    PyObject* gvec_pyobj=NULL;
    struct PyGVecObject *gvec_obj=NULL;
    double A=0, ierr=0;

    double loglike=0;
    PyObject *tup=NULL;

    struct image *image=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOdd", 
                                       &image_obj, &gvec_pyobj,
                                       &A, &ierr)) {
        return NULL;
    }

    if (!check_numpy_image(image_obj)) {
        PyErr_SetString(PyExc_IOError, "image input must be a 2D double PyArrayObject");
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    gvec_obj = (struct PyGVecObject *) gvec_pyobj;

    flags=calculate_loglike_margamp(image, gvec_obj->gvec, A, ierr, &loglike);

    // does not free underlying array
    image = image_free(image);

    tup = PyTuple_New(2);
    PyTuple_SetItem(tup, 0, PyFloat_FromDouble(loglike));
    PyTuple_SetItem(tup, 1, PyInt_FromLong((long)flags));

    return tup;
}


static PyObject *
PyGMixFit_loglike_old(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    PyObject* gvec_pyobj=NULL;
    struct PyGVecObject *gvec_obj=NULL;
    double A=0, ierr=0;

    double loglike=0;
    PyObject *tup=NULL;

    struct image *image=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOdd", 
                                       &image_obj, &gvec_pyobj,
                                       &A, &ierr)) {
        return NULL;
    }

    if (!check_numpy_image(image_obj)) {
        PyErr_SetString(PyExc_IOError, "image input must be a 2D double PyArrayObject");
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    gvec_obj = (struct PyGVecObject *) gvec_pyobj;

    flags=calculate_loglike_old(image, gvec_obj->gvec, A, ierr, &loglike);

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
    PyObject* gvec_pyobj=NULL;
    struct PyGVecObject *gvec_obj=NULL;
    double ivar=0;

    double loglike=0;
    PyObject *tup=NULL;

    struct image *image=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOd", 
                                &image_obj, &gvec_pyobj, &ivar)) {
        return NULL;
    }

    if (!check_numpy_image(image_obj)) {
        PyErr_SetString(PyExc_IOError, "image input must be a 2D double PyArrayObject");
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    gvec_obj = (struct PyGVecObject *) gvec_pyobj;

    flags=calculate_loglike(image, gvec_obj->gvec, ivar, &loglike);

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

    flags=calculate_loglike_old_old(image, obj_gvec, psf_gvec, A, ierr, &loglike);

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
PyGMixFit_fill_model_old(PyObject *self, PyObject *args) 
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

    flags=fill_model_old(image, obj_gvec, psf_gvec, diff);

    obj_gvec = gvec_free(obj_gvec);
    psf_gvec = gvec_free(psf_gvec);
    // does not free underlying array
    image = image_free(image);
    diff = image_free(diff);

    return PyInt_FromLong(flags);
}


/* 
   This one can work on a subgrid.  There is not separate psf gmix, 
   you are expected to send a convolved GMix object in that case

   No error checking on the gvec object is performed, do it in python!
*/

static PyObject *
PyGMixFit_fill_model(PyObject *self, PyObject *args) 
{
    PyObject *image_obj=NULL;
    PyObject *gvec_pyobj=NULL;
    struct PyGVecObject *gvec_obj=NULL;

    int nsub=0;

    struct image *image=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOi", &image_obj, &gvec_pyobj, &nsub)) {
        return NULL;
    }
    gvec_obj = (struct PyGVecObject *) gvec_pyobj;

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    flags=fill_model_subgrid(image, gvec_obj->gvec, nsub);

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
    {"fill_model_coellip_old", (PyCFunction)PyGMixFit_coellip_fill_model_old,  METH_VARARGS,  "fill the model image"},
    {"fill_ydiff_dev_galsim", (PyCFunction)PyGMixFit_fill_ydiff_dev_galsim,  METH_VARARGS,  "fill the dev model image"},
    {"fill_ydiff_dev", (PyCFunction)PyGMixFit_fill_ydiff_dev,  METH_VARARGS,  "fill the dev model image"},
    {"fill_ydiff_dev6", (PyCFunction)PyGMixFit_fill_ydiff_dev6,  METH_VARARGS,  "fill the dev model image"},
    {"fill_ydiff_dev10", (PyCFunction)PyGMixFit_fill_ydiff_dev10,  METH_VARARGS,  "fill the dev model image"},
    {"fill_ydiff_exp", (PyCFunction)PyGMixFit_fill_ydiff_exp,  METH_VARARGS,  "fill the exp model image"},
    {"fill_model_old", (PyCFunction)PyGMixFit_fill_model_old,  METH_VARARGS,  "fill the model image"},

    {"loglike_coellip_old", (PyCFunction)PyGMixFit_loglike_coellip_old,  METH_VARARGS,  "calc logl, analytically marginalized over amplitude"},
    {"loglike_coellip_margamp", (PyCFunction)PyGMixFit_loglike_coellip_margamp,  METH_VARARGS,  "calc logl, analytically marginalized over amplitude"},
    {"loglike_old", (PyCFunction)PyGMixFit_loglike_old,  METH_VARARGS,  "calc logl, analytically marginalized over amplitude"},
    {"loglike_margamp", (PyCFunction)PyGMixFit_loglike_margamp,  METH_VARARGS,  "calc logl, analytically marginalized over amplitude"},

    {"loglike", (PyCFunction)PyGMixFit_loglike,  METH_VARARGS,  "calc full log likelihood"},
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

    PyGVecType.tp_new = PyType_GenericNew;

#if PY_MAJOR_VERSION >= 3
    if (PyType_Ready(&PyGVecType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else
    if (PyType_Ready(&PyGVecType) < 0) {
        return;
    }
    m = Py_InitModule3("_render", render_module_methods, 
            "This module gmix fit related routines.\n");
    if (m==NULL) {
        return;
    }
#endif

    Py_INCREF(&PyGVecType);
    PyModule_AddObject(m, "GVec", (PyObject *)&PyGVecType);
    import_array();
}
