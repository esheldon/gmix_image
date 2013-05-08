#ifndef _GMIX_EM_HEADER_GUARD
#define _GMIX_EM_HEADER_GUARD

#include "image.h"
#include "gvec.h"

struct gmix {
    size_t maxiter;
    double tol;
    int coellip;
    int cocenter;
    int verbose;
};

struct sums {
    // scratch on a given pixel
    double gi;
    double trowsum;
    double tcolsum;
    double tu2sum;
    double tuvsum;
    double tv2sum;

    // sums over all pixels
    double pnew;
    double rowsum;
    double colsum;
    double u2sum;
    double uvsum;
    double v2sum;
};

struct iter {
    size_t ngauss;

    // sums over all pixels and all gaussians
    double skysum;
    double psum;

    double psky;
    double nsky;

    struct sums *sums;
};

int gmix_em(struct gmix* self,
        struct image *image, 
        struct gvec *gvec,
        size_t *iter,
        double *fdiff);
int gmix_em_cocenter(struct gmix* self,
        struct image *image, 
        struct gvec *gvec,
        size_t *iter,
        double *fdiff);

/*
int gmix_em_coellip(struct gmix* self,
        struct image *image, 
        struct gvec *gvec,
        struct gvec *gvec_psf, // can be NULL
        size_t *iter,
        double *fdiff);
*/
int gmix_get_sums(struct gmix* self,
                  struct image *image,
                  struct gvec *gvec,
                  struct iter* iter);

/*
double gmix_evaluate_convolved(struct gmix* self,
                               struct gauss *gauss,
                               struct gvec *gvec_psf,
                               double u2, double uv, double v2,
                               int *flags);
*/


struct iter *iter_new(size_t ngauss);
struct iter *iter_free(struct iter *self);
void iter_clear(struct iter *self);

void gmix_set_gvec_fromiter(struct gvec *gvec, 
                            struct iter *iter);

/*
void gmix_set_gvec_fromiter_convolved(struct gvec *gvec, 
                                      struct gvec *gvec_psf,
                                      struct iter* iter);
*/
/* 
 * when we are using same center, we want to be able to just set a new p and
 * center *
 */

/*
void gmix_set_p_and_cen(struct gvec* gvec, 
                        double* pnew,
                        double* rowsum,
                        double* colsum);
*/
#endif
