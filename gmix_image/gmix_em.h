#ifndef _GMIX_EM_HEADER_GUARD
#define _GMIX_EM_HEADER_GUARD

#include "image.h"
#include "gmix.h"
#include "jacobian.h"

struct gmix_em {
    size_t maxiter;
    double tol;
    int cocenter;
    int verbose;

    int has_jacobian;
    struct jacobian jacob;
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

struct gmix_em *gmix_em_new(size_t maxiter,
                            double tol,
                            int cocenter,
                            int verbose);

void gmix_em_add_jacobian(struct gmix_em *self,
                          double row0,
                          double col0,
                          double dudrow,
                          double dudcol,
                          double dvdrow,
                          double dvdcol);

int gmix_em_run(struct gmix_em* self,
                const struct image *image, 
                struct gmix *gmix,
                size_t *iter,
                double *fdiff);
int gmix_em_cocenter_run(struct gmix_em* self,
                         const struct image *image, 
                         struct gmix *gmix,
                         size_t *iter,
                         double *fdiff);

/*
int gmix_em_coellip(struct gmix_em* self,
        struct image *image, 
        struct gmix *gmix,
        struct gmix *gmix_psf, // can be NULL
        size_t *iter,
        double *fdiff);
*/
int gmix_get_sums(struct gmix_em* self,
                  const struct image *image,
                  struct gmix *gmix,
                  struct iter* iter);

int gmix_get_sums_pix(struct gmix_em* self,
                      const struct image *image,
                      struct gmix *gmix,
                      struct iter* iter);

/*
   Must set the jacobian and center
*/
int gmix_get_sums_jacobian(struct gmix_em* self,
                           const struct image *image,
                           struct gmix *gmix,
                           struct iter* iter);

/*
double gmix_evaluate_convolved(struct gmix_em* self,
                               struct gauss *gauss,
                               struct gmix *gmix_psf,
                               double u2, double uv, double v2,
                               int *flags);
*/


struct iter *iter_new(size_t ngauss);
struct iter *iter_free(struct iter *self);
void iter_clear(struct iter *self);

void gmix_set_gmix_fromiter(struct gmix *gmix, 
                            struct iter *iter);

/*
void gmix_set_gmix_fromiter_convolved(struct gmix *gmix, 
                                      struct gmix *gmix_psf,
                                      struct iter* iter);
*/
/* 
 * when we are using same center, we want to be able to just set a new p and
 * center *
 */

/*
void gmix_set_p_and_cen(struct gmix* gmix, 
                        double* pnew,
                        double* rowsum,
                        double* colsum);
*/
#endif
