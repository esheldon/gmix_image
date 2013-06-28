#ifndef _GMIX_PYWRAP_HEADER_GUARD
#define _GMIX_PYWRAP_HEADER_GUARD

#include "gmix.h"

struct PyGMixObject {
    PyObject_HEAD
    struct gmix *gmix;
};

#endif
