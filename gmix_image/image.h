#ifndef _IMAGE_HEADER_GUARD_H
#define _IMAGE_HEADER_GUARD_H

#include <stdlib.h>

struct image {
    size_t size;
    size_t nrows;
    size_t ncols;

    int has_sky;
    double sky;

    int has_counts;
    double counts;

    double *data;
};


#define IM_SIZE(im) ((im)->size)
#define IM_NROWS(im) ((im)->nrows)
#define IM_NCOLS(im) ((im)->ncols)

#define IM_SET(im, row, col, val) \
    ( *((im)->data + (row)*(im)->ncols + (col)) = (val) )
#define IM_GET(im, row, col) \
    ( *((im)->data + (row)*(im)->ncols + (col)) )

#define IM_GETP(im, row, col) \
    ( ( (im)->data + (row)*(im)->ncols + (col)) )


struct image *image_new(size_t nrows, size_t ncols);
struct image *image_free(struct image *self);
struct image *image_read_text(const char* filename);

void image_set_sky(struct image *self, double sky);
double image_sky(struct image *self);

void image_calc_counts(struct image *self);
double image_counts(struct image *self);

void image_normalize(struct image *self, double norm);


#endif
