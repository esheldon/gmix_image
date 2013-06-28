#include <Python.h>
#include <numpy/arrayobject.h> 
#include "py_helpers.h"


struct image *pyhelp_associate_image(PyObject* image_obj)
{
    struct image *image=NULL;
    double *data=NULL;
    int alloc_data=0; // we don't allocate
    size_t i=0;
    npy_intp *dims=NULL;

    dims = PyArray_DIMS((PyArrayObject*)image_obj);
    data = PyArray_DATA((PyArrayObject*)image_obj);

    image = _image_new(dims[0], dims[1], alloc_data);

    image->rows[0] = data;
    for(i = 1; i < dims[0]; i++) {
        image->rows[i] = image->rows[i-1] + dims[1];
    }

    return image;
}

double pyhelp_dict_get_double(PyObject *dict, const char *name, long *status)
{
    PyObject *tmp=NULL;
    double val=0;

    // borrowed ref
    tmp = PyDict_GetItemString(dict, name);
    if (!tmp) {
        PyErr_Format(PyExc_KeyError, "Key not found: '%s'", name);
        (*status)=1;
        return -1;
    }
    val = PyFloat_AsDouble(tmp);
    return val;
}

long pyhelp_dict_to_jacob(PyObject *dict, struct jacobian *jacob)
{
    long status=0;

    if (!PyDict_Check(dict)) {
        PyErr_Format(PyExc_TypeError, "jacobian is not a dict");
        return 0;
    }
    jacob->row0 = pyhelp_dict_get_double(dict,"row0",&status);
    if (status) {
        return 0;
    }

    jacob->col0 = pyhelp_dict_get_double(dict,"col0",&status);
    if (status) {
        return 0;
    }

    jacob->dudrow = pyhelp_dict_get_double(dict,"dudrow",&status);
    if (status) {
        return 0;
    }

    jacob->dudcol = pyhelp_dict_get_double(dict,"dudcol",&status);
    if (status) {
        return 0;
    }

    jacob->dvdrow = pyhelp_dict_get_double(dict,"dvdrow",&status);
    if (status) {
        return 0;
    }

    jacob->dvdcol = pyhelp_dict_get_double(dict,"dvdcol",&status);
    if (status) {
        return 0;
    }

    return 1;

}
