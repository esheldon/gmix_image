#ifndef _BOUND_HEADER_GUARD_H
#define _BOUND_HEADER_GUARD_H

#include <stdlib.h>

struct bound {
    size_t rowmin;
    size_t rowmax;
    size_t colmin;
    size_t colmax;
};


struct bound *bound_new(size_t rowmin, 
                        size_t rowmax, 
                        size_t colmin, 
                        size_t colmax);
struct bound *bound_free(struct bound *self);

void bound_set(struct bound* self,
               size_t rowmin, 
               size_t rowmax, 
               size_t colmin, 
               size_t colmax);

#endif
