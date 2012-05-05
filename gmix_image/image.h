/*
  This is basically modeled after PHOTO regions
 */
#ifndef _IMAGE_HEADER_GUARD_H
#define _IMAGE_HEADER_GUARD_H

#include <stdlib.h>
#include <stdio.h>
#include "bound.h"

struct image {
    size_t size;
    size_t nrows;
    size_t ncols;

    struct image *parent; // != NULL if a sub image
    size_t row0;          // > 0 if this is a sub-image
    size_t col0;

    double **rows;        // not owned if a sub-image

    double sky;
    double counts;        // will get recalculated for sub-images

    int is_owner;         // ==0 for sub-images
};


#define IM_IS_OWNER(im) ((im)->is_owner)
#define IM_PARENT(im) ((im)->parent)

#define IM_SIZE(im) ((im)->size)
#define IM_NROWS(im) ((im)->nrows)
#define IM_NCOLS(im) ((im)->ncols)

#define IM_ROW(im,row) \
    ((im)->rows[(im)->row0 + (row)])
#define IM_ROW_ITER(im,row) \
    ((im)->rows[(im)->row0 + (row)] + (im)->col0)
#define IM_ROW_END(im,row) \
    ((im)->rows[(im)->row0 + (row)] + (im)->col0 + (im)->ncols)

#define IM_GET(im, row, col)                  \
    ( *((im)->rows[(im)->row0 + (row)] + (im)->col0 + col) )
#define IM_SET(im, row, col, val)             \
    ( *((im)->rows[(im)->row0 + (row)] + (im)->col0 + col) = (val) )
#define IM_GETP(im, row, col)                 \
    (  ((im)->rows[(im)->row0 + (row)] + (im)->col0 + col) )

#define IM_SKY(im)                  \
    ( (im)->sky )
#define IM_SET_SKY(im, val)             \
    ( (im)->sky = (val) )

#define IM_COUNTS(im)                  \
    ( (im)->counts )
#define IM_SET_COUNTS(im, val)             \
    ( (im)->counts = (val) )


struct image *image_new(size_t nrows, size_t ncols);
struct image *image_free(struct image *self);
struct image *image_read_text(const char* filename);

void image_write(struct image *self, FILE* stream);

void image_calc_counts(struct image *self);
void image_add_scalar(struct image *self, double val);

struct image* image_sub(struct image *parent, struct bound* bound);

#endif
