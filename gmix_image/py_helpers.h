#ifndef _py_helpers_header_guard
#define _py_helpers_header_guard

#include <Python.h>
#include <numpy/arrayobject.h> 
#include "image.h"
#include "jacobian.h"

//int pyhelp_check_numpy_image(PyObject *obj);
//int pyhelp_check_numpy_array(PyObject *obj);

/*
 * no copy is made, nor ownership granted, but you do need to free the image
 */
struct image *pyhelp_associate_image(PyObject* image_obj);

double pyhelp_dict_get_double(PyObject *dict, const char *name, long *status);
long pyhelp_dict_to_jacob(PyObject *dict, struct jacobian *jacob);

#endif
