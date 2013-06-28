#include <Python.h>
#include <numpy/arrayobject.h> 
#include <alloca.h>

#include "gvec.h"
#include "jacobian.h"
#include "render.h"
#include "image.h"
#include "bound.h"
#include "defs.h"

struct PyGMixRenderer {
    PyObject_HEAD
    struct gvec *gvec;
};

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

/*
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

struct gvec *coellip_pyarray_to_gvec_Tfrac(PyObject *array)
{
    double *pars=NULL;
    int sz=0;
    struct gvec *gvec=NULL;
    pars = PyArray_DATA(array);
    sz = PyArray_SIZE(array);

    gvec = gvec_new_coellip_Tfrac(pars, sz);
    return gvec;
}
struct gvec *coellip_pyarray_to_gvec(PyObject *array)
{
    double *pars=NULL;
    int sz=0;
    struct gvec *gvec=NULL;
    pars = PyArray_DATA(array);
    sz = PyArray_SIZE(array);

    gvec = gvec_new_coellip(pars, sz);
    return gvec;
}
*/



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

    enum gvec_model model=type;

    self->gvec = gvec_new_model(model, pars, size);
    if (!self->gvec) {
        PyErr_Format(PyExc_ValueError, 
                "error constructing gmix");
        return -1;
    }
    /*
    switch (type) {
        case 0:
            self->gvec = gvec_from_pars(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing gmix from full array");
                return -1;
            }
            break;
        case 1:
            self->gvec = gvec_new_coellip(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing gmix from coellip");
                return -1;
            }
            break;
        case 5:
            fprintf(stderr,"Using old Tfrac\n");
            self->gvec = gvec_new_coellip_Tfrac(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing gmix from coellip Tfrac");
                return -1;
            }
            break;

        case 2:
            self->gvec = gvec_from_pars_turb(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing gmix from turb");
                return -1;
            }
            break;

        case 3:
            self->gvec = gvec_from_pars_exp6(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing exp6 gmix");
                return -1;
            }
            break;
        case 4:
            self->gvec = gvec_from_pars_dev10(pars, size);
            if (self->gvec == NULL) {
                PyErr_Format(PyExc_ValueError, 
                        "error constructing dev10 gmix");
                return -1;
            }
            break;

        case 6:
            self->gvec = gvec_from_pars_bd(pars, size);
            if (self->gvec == NULL) {
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

static PyObject *PyGVecObject_get_size(struct PyGVecObject* self)
{
    return PyLong_FromLong((long)self->gvec->size);
}
static PyObject *PyGVecObject_get_pars(struct PyGVecObject* self)
{
    PyObject *pars_array=NULL;
    npy_intp dims[1];
    int npy_dtype=NPY_FLOAT64;
    double *pars=NULL;
    struct gauss *gauss=NULL;
    int i=0, ii=0, ngauss=0;

    ngauss=self->gvec->size;
    dims[0] = 6*ngauss;

    pars_array=PyArray_ZEROS(1, dims, npy_dtype, 0);
    pars=PyArray_DATA(pars_array);

    gauss=self->gvec->data;
    for (i=0; i<self->gvec->size; i++) {
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
    T = gvec_get_T(self->gvec);
    return PyFloat_FromDouble(T);
}
static PyObject *PyGVecObject_get_psum(struct PyGVecObject* self)
{
    double psum=0;
    psum = gvec_get_psum(self->gvec);
    return PyFloat_FromDouble(psum);
}
static PyObject *PyGVecObject_set_psum(struct PyGVecObject* self, PyObject *args)
{
    double psum=0;

    if (!PyArg_ParseTuple(args, (char*)"d", &psum)) {
        return NULL;
    }

    gvec_set_psum(self->gvec, psum);

    Py_XINCREF(Py_None);
    return Py_None;
}


static PyObject *PyGVecObject_get_cen(struct PyGVecObject* self)
{
    npy_intp dims[1] = {2};
    int npy_dtype=NPY_FLOAT64;
    PyObject *cen=NULL;
    double *cendata=NULL;

    cen=PyArray_ZEROS(1, dims, npy_dtype, 0);
    cendata=PyArray_DATA(cen);

    gvec_get_cen(self->gvec, &cendata[0], &cendata[1]);

    return cen;
}
static PyObject *PyGVecObject_set_cen(struct PyGVecObject* self, PyObject *args)
{
    double row=0, col=0;

    if (!PyArg_ParseTuple(args, (char*)"dd", &row, &col)) {
        return NULL;
    }

    gvec_set_cen(self->gvec, row, col);

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
/*
static PyObject *PyGVecObject_convolve_replace_wrong(struct PyGVecObject* self, PyObject *args)
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
*/
/* error checking should happen in python */
static PyObject *PyGVecObject_convolve_replace(struct PyGVecObject* self, PyObject *args)
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
    {"get_size", (PyCFunction)PyGVecObject_get_size, METH_NOARGS, "get_size\n\nreturn number of gaussians."},
    {"get_dlist", (PyCFunction)PyGVecObject_get_dlist, METH_NOARGS, "get_dlist\n\nreturn list of dicts."},
    {"get_e1e2T", (PyCFunction)PyGVecObject_get_e1e2T, METH_NOARGS, "get_e1e2T\n\nreturn stats based on average moments val=sum(val_i*p)/sum(p)."},
    {"get_T", (PyCFunction)PyGVecObject_get_T, METH_NOARGS, "get_T\n\nreturn T=sum(T_i*p)/sum(p)."},
    {"get_psum", (PyCFunction)PyGVecObject_get_psum, METH_NOARGS, "get_psum\n\nreturn sum(p)."},
    {"set_psum", (PyCFunction)PyGVecObject_set_psum, METH_VARARGS, "set_psum\n\nset new sum(p)."},
    {"get_cen", (PyCFunction)PyGVecObject_get_cen, METH_NOARGS, "get_cen\n\nreturn cen=sum(cen_i*p)/sum(p)."},
    {"set_cen", (PyCFunction)PyGVecObject_set_cen, METH_VARARGS, "set_cen\n\nSet all centers to the input row,col"},
    {"get_pars", (PyCFunction)PyGVecObject_get_pars, METH_NOARGS, "get_pars\n\nreturn full pars."},
    {"_convolve_replace", (PyCFunction)PyGVecObject_convolve_replace, METH_VARARGS, "convolve_inplace\n\nConvolve with the psf in place."},
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
    PyObject *gvec_pyobj=NULL;

    struct PyGVecObject *gvec_obj=NULL;
    struct image *image=NULL;
    struct image *diff=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OdOO", 
                &image_obj, &ivar, &gvec_pyobj, &diff_obj)) {
        return NULL;
    }

    if (!check_image_and_diff(image_obj,diff_obj)) {
        return NULL;
    }

    gvec_obj = (struct PyGVecObject *) gvec_pyobj;

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);
    diff = associate_image(diff_obj, dims[0], dims[1]);

    flags=fill_ydiff(image, ivar, gvec_obj->gvec, diff);

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
    PyObject *gvec_pyobj=NULL;
    PyObject* diff_obj=NULL;

    struct PyGVecObject *gvec_obj=NULL;
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
                &gvec_pyobj,
                &diff_obj)) {
        return NULL;
    }

    if (!check_image_and_diff(image_obj,diff_obj)) {
        return NULL;
    }

    gvec_obj = (struct PyGVecObject *) gvec_pyobj;

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);
    diff = associate_image(diff_obj, dims[0], dims[1]);

    jacobian_set(&jacob, row0, col0, dudrow, dudcol, dvdrow, dvdcol);

    flags=fill_ydiff_jacob(image, ivar, &jacob,
                           gvec_obj->gvec, diff);
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
    PyObject *gvec_pyobj=NULL;
    PyObject* diff_obj=NULL;

    struct PyGVecObject *gvec_obj=NULL;
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
                &gvec_pyobj,
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

    gvec_obj = (struct PyGVecObject *) gvec_pyobj;

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);
    weight = associate_image(weight_obj, dims[0], dims[1]);
    diff = associate_image(diff_obj, dims[0], dims[1]);

    jacobian_set(&jacob, row0, col0, dudrow, dudcol, dvdrow, dvdcol);

    flags=fill_ydiff_wt_jacob(image, weight, &jacob,
                              gvec_obj->gvec, diff);

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
PyGMixFit_loglike(PyObject *self, PyObject *args) 
{
    PyObject* image_obj=NULL;
    PyObject* gvec_pyobj=NULL;
    struct PyGVecObject *gvec_obj=NULL;
    double ivar=0;

    double loglike=0, s2n_numer=0, s2n_denom=0;
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

    flags=calculate_loglike(image, gvec_obj->gvec, ivar, &s2n_numer, &s2n_denom, &loglike);

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
    PyObject* gvec_pyobj=NULL;

    double loglike=0, s2n_numer=0, s2n_denom=0;
    PyObject *tup=NULL;

    struct image *image=NULL;
    struct jacobian jacob;
    struct PyGVecObject *gvec_obj=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OdddddddO", 
                &image_obj,
                &ivar,
                &dudrow, &dudcol, &dvdrow, &dvdcol,
                &row0,
                &col0,
                &gvec_pyobj)) {
        return NULL;
    }

    if (!check_numpy_image(image_obj)) {
        PyErr_SetString(PyExc_IOError, "image input must be a 2D double PyArrayObject");
        return NULL;
    }

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    gvec_obj = (struct PyGVecObject *) gvec_pyobj;

    jacobian_set(&jacob, row0, col0, dudrow, dudcol, dvdrow, dvdcol);

    flags=calculate_loglike_jacob(image,
                                  ivar,
                                  &jacob,
                                  gvec_obj->gvec,
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
    PyObject* gvec_pyobj=NULL;

    double loglike=0, s2n_numer=0, s2n_denom=0;
    PyObject *tup=NULL;

    struct image *image=NULL;
    struct image *weight=NULL;
    struct jacobian jacob;
    struct PyGVecObject *gvec_obj=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOddddddO", 
                &image_obj,
                &weight_obj,
                &dudrow, &dudcol, &dvdrow, &dvdcol,
                &row0,
                &col0,
                &gvec_pyobj)) {
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

    gvec_obj = (struct PyGVecObject *) gvec_pyobj;

    jacobian_set(&jacob, row0, col0, dudrow, dudcol, dvdrow, dvdcol);

    flags=calculate_loglike_wt_jacob(image,
                                     weight,
                                     &jacob,
                                     gvec_obj->gvec,
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


static PyObject *
PyGMixFit_fill_model_jacob(PyObject *self, PyObject *args) 
{
    PyObject *image_obj=NULL;
    PyObject *gvec_pyobj=NULL;
    double dudrow, dudcol, dvdrow, dvdcol;
    double row0=0, col0=0;
    struct jacobian jacob;
    struct PyGVecObject *gvec_obj=NULL;

    int nsub=0;

    struct image *image=NULL;
    npy_intp *dims=NULL;

    int flags=0;

    if (!PyArg_ParseTuple(args, (char*)"OOddddddi",
                &image_obj, 
                &gvec_pyobj, 
                &dudrow, &dudcol, &dvdrow, &dvdcol,
                &row0,
                &col0,
                &nsub)) {
        return NULL;
    }
    gvec_obj = (struct PyGVecObject *) gvec_pyobj;

    jacobian_set(&jacob, row0, col0, dudrow, dudcol, dvdrow, dvdcol);

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    image = associate_image(image_obj, dims[0], dims[1]);

    flags=fill_model_subgrid_jacob(image, gvec_obj->gvec, &jacob, nsub);

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
