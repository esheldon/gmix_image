#ifndef _JACOBIAN_HEADER_GUARD
#define _JACOBIAN_HEADER_GUARD

struct jacobian {
    double dudrow;
    double dudcol;
    double dvdrow;
    double dvdcol;
};

// row col here are relative to the "center"
#define JACOB_PIX2U(jacob, row, col)  \
    ( (jacob)->dudrow*(row) + (jacob)->dudcol*(col) )

#define JACOB_PIX2V(jacob, row, col)  \
    ( (jacob)->dvdrow*(row) + (jacob)->dvdcol*(col) )

void jacobian_set(struct jacobian *self, 
                  double dudrow,
                  double dudcol,
                  double dvdrow,
                  double dvdcol);
#endif
