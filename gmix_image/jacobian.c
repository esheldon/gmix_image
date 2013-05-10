#include <stdlib.h>
#include <stdio.h>
#include "jacobian.h"

struct jacobian *jacobian_new(double row0,
                              double col0,
                              double dudrow,
                              double dudcol,
                              double dvdrow,
                              double dvdcol)
{
    struct jacobian *self=NULL;

    self=calloc(1, sizeof(struct jacobian));
    if (!self) {
        fprintf(stderr,"could not allocate struct jacobian\n");
        exit(1);
    }

    jacobian_set(self,
                 row0,
                 col0,
                 dudrow,
                 dudcol,
                 dvdrow,
                 dvdcol);
    return self;
}


void jacobian_set(struct jacobian *self, 
                  double row0,
                  double col0,
                  double dudrow,
                  double dudcol,
                  double dvdrow,
                  double dvdcol)
{
    self->row0=row0;
    self->col0=col0;
    self->dudrow=dudrow;
    self->dudcol=dudcol;
    self->dvdrow=dvdrow;
    self->dvdcol=dvdcol;
}

