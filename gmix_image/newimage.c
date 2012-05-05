#include <stdlib.h>
#include <stdio.h>
#include "newimage.h"
#include "bound.h"

struct image *image_new(size_t nrows, size_t ncols)
{
    struct image *self=NULL;
    size_t nel = nrows*ncols, i=0;
    if (nel == 0) {
        fprintf(stderr,"image size must be > 0\n");
        exit(EXIT_FAILURE);
    }

    self = calloc(1, sizeof(struct image));
    if (self==NULL) {
        fprintf(stderr,"could not allocate struct image\n");
        exit(EXIT_FAILURE);
    }

    self->nrows=nrows;
    self->ncols=ncols;
    self->size=nel;


    self->rows = calloc(nrows,sizeof(double *));
    if (self->rows==NULL) {
        fprintf(stderr,"could not allocate image of dimensions [%lu,%lu]\n",
                nrows,ncols);
        exit(EXIT_FAILURE);
    }
    self->rows[0] = calloc(self->size,sizeof(double));
    if (self->rows[0]==NULL) {
        fprintf(stderr,"could not allocate image of dimensions [%lu,%lu]\n",
                nrows,ncols);
        exit(EXIT_FAILURE);
    }

    for(i = 1; i < nrows; i++) {
        self->rows[i] = self->rows[i-1] + ncols;
    }

    self->is_owner=1;
    return self;
}

struct image *image_free(struct image *self)
{
    if (self) {
        if (self->rows && self->is_owner) {
            free(self->rows[0]);
            free(self->rows);
        }
        free(self);
        self=NULL;
    }
    return self;
}

void image_write(struct image *self, FILE* stream)
{
    size_t row=0;
    double *col=NULL, *end=NULL;
    fprintf(stream,"%lu\n", IM_NROWS(self));
    fprintf(stream,"%lu\n", IM_NCOLS(self));
    fprintf(stream,"%.16g\n", IM_SKY(self));

    for (row=0; row<IM_NROWS(self); row++) {
        col    = IM_ROW_ITER(self,row);
        end = IM_ROW_END(self,row);
        for (; col != end; col++) {
            fprintf(stream,"%.16g ",*col);
        }
        fprintf(stream,"\n");
    }
}
struct image* image_sub(struct image *parent, struct bound* bound)
{
    struct image *self=NULL;
    size_t rowmax=0, colmax=0;

    self = calloc(1, sizeof(struct image));
    if (self==NULL) {
        fprintf(stderr,"could not allocate struct sub-image\n");
        exit(EXIT_FAILURE);
    }

    self->parent=parent;
    self->rows=parent->rows;
    IM_SET_SKY(self, IM_SKY(parent));
    self->is_owner=0;

    // note using size_t takes care of min values being >= 0
    rowmax = IM_NROWS(parent) -1;
    colmax = IM_NCOLS(parent) -1;

    self->row0 = bound->rowmin;
    self->col0 = bound->colmin;

    rowmax = bound->rowmax > rowmax ? rowmax : bound->rowmax;
    colmax = bound->colmax > colmax ? colmax : bound->colmax;

    self->nrows = rowmax - bound->rowmin + 1;
    self->ncols = colmax - bound->colmin + 1;
    self->size = self->nrows*self->ncols;

    // we keep the counts for the sub-image region
    image_calc_counts(self);

    return self;
}

void image_calc_counts(struct image *self)
{
    double counts=0;
    double *col=NULL, *end=NULL;
    size_t nrows = IM_NROWS(self);
    size_t row=0;
    for (row=0;row<nrows;row++) {

        col    = IM_ROW_ITER(self,row);
        end = IM_ROW_END(self,row);
        for (; col != end; col++) {
            counts += *col;
        }
    }
    self->counts=counts;
}

void image_add_scalar(struct image *self, double val)
{
    double *col=NULL, *end=NULL;
    size_t row=0, nrows = IM_NROWS(self);
    for (row=0;row<nrows;row++) {
        col    = IM_ROW_ITER(self,row);
        end = IM_ROW_END(self,row);
        for (; col != end; col++) {
            *col += val;
        }
    }

    image_calc_counts(self);
}

