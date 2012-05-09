#ifndef _GVEC_HEADER_GUARD
#define _GVEC_HEADER_GUARD

#include "matrix.h"

struct gauss {
    double p;
    double row;
    double col;
    double irr;
    double irc;
    double icc;
    double det;
};

struct gvec {
    size_t size;
    struct gauss* data;

    // these only make sense for same-center gaussians
    // e.g. the psf
    double total_irr;
    double total_irc;
    double total_icc;
};

struct gvec *gvec_new(size_t n);
struct gvec *gvec_free(struct gvec *self);
void gvec_set_dets(struct gvec *self);

// only makes sense for same center, e.g. psf
void gvec_set_total_moms(struct gvec *self);

// this is actually kind of unclear to use in practice since it is easy to
// screw up which parameters go where
void gauss_set(struct gauss* self, 
               double p, 
               double row, 
               double col,
               double irr,
               double irc,
               double icc);

int gvec_copy(struct gvec *self, struct gvec* dest);
void gvec_print(struct gvec *self, FILE* fptr);

// calculate the weighted sum of the moments
//  sum_gi( p*(irr + icc )
double gvec_wmomsum(struct gvec* gvec);

// 0 returned if a zero determinant is found somewhere, else 1
int gvec_wmean_center(const struct gvec* gvec, struct vec2* mu_new);

void gvec_wmean_covar(const struct gvec* gvec, struct mtx2 *cov);

#endif
