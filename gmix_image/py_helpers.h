#ifndef _py_helpers_header_guard
#define _py_helpers_header_guard

#include <Python.h>
#include <numpy/arrayobject.h> 
#include "image.h"

int pyhelp_check_numpy_image(PyObject *obj);
int pyhelp_check_numpy_array(PyObject *obj);

/*
 * no copy is made, nor ownership granted, but you do need to free the image
 */
struct image *pyhelp_associate_image(PyObject* image_obj, size_t nrows, size_t ncols);

#endif
