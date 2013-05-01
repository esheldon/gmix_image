#include "jacobian.h"

void jacobian_set(struct jacobian *self, 
                  double dudrow,
                  double dudcol,
                  double dvdrow,
                  double dvdcol)
{
    self->dudrow=dudrow;
    self->dudcol=dudcol;
    self->dvdrow=dvdrow;
    self->dvdcol=dvdcol;
}

