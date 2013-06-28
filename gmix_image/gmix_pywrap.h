#ifndef _gmix_pywrap_header_guard
#define _gmix_pywrap_header_guard

#include "gmix.h"

struct PyGMixObject {
    PyObject_HEAD
    struct gmix *gmix;
};

#endif
