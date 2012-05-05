#include <stdlib.h>
#include <stdio.h>
#include "bound.h"

struct bound *bound_new(size_t rowmin, 
                        size_t rowmax, 
                        size_t colmin, 
                        size_t colmax)
{
    struct bound *self=NULL;

    self = calloc(1, sizeof(struct bound));
    if (self==NULL) {
        fprintf(stderr,"could not allocate struct bound\n");
        exit(EXIT_FAILURE);
    }

    bound_set(self, rowmin, rowmax, colmin, colmax);
    return self;
}

struct bound *bound_free(struct bound *self) {
    if (self) {
        free(self);
    }
    return NULL;
}

void bound_set(struct bound* self,
               size_t rowmin, 
               size_t rowmax, 
               size_t colmin, 
               size_t colmax)
{
    self->rowmin=rowmin;
    self->rowmax=rowmax;
    self->colmin=colmin;
    self->colmax=colmax;
}
