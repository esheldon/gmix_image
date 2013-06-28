#ifndef _RENDER_HEADER_GUARD
#define _RENDER_HEADER_GUARD

#include "gvec.h"
#include "image.h"
#include "jacobian.h"

int fill_model_subgrid_jacob(struct image *image, 
                             const struct gvec *gvec, 
                             const struct jacobian *jacob,
                             int nsub);

int fill_model_subgrid(struct image *image, 
                       struct gvec *gvec, 
                       int nsub);

int fill_ydiff_wt_jacob_generic(const struct image *image,
                                const struct image *weight, // either
                                double ivar,                // or
                                const struct jacobian *jacob,
                                const struct gvec *gvec,
                                struct image *diff_image);

int fill_ydiff_jacob(const struct image *image,
                     double ivar,
                     const struct jacobian *jacob,
                     const struct gvec *gvec,
                     struct image *diff_image);

int fill_ydiff_wt_jacob(const struct image *image,
                        const struct image *weight,
                        const struct jacobian *jacob,
                        const struct gvec *gvec,
                        struct image *diff_image);

int fill_ydiff(struct image *image,
               double ivar,
               struct gvec *gvec,
               struct image *diff_image);

int calculate_loglike_margamp(struct image *image, 
                              struct gvec *gvec, 
                              double A,
                              double ierr,
                              double *loglike);




int calculate_loglike_wt_jacob_generic(const struct image *image, 
                                       const struct image *weight, // either
                                       double ivar,                // or
                                       const struct jacobian *jacob,
                                       const struct gvec *gvec, 
                                       double *s2n_numer,
                                       double *s2n_denom,
                                       double *loglike);

int calculate_loglike_wt_jacob(const struct image *image, 
                               const struct image *weight,
                               const struct jacobian *jacob,
                               const struct gvec *gvec, 
                               double *s2n_numer,
                               double *s2n_denom,
                               double *loglike);

int calculate_loglike_jacob(const struct image *image, 
                            double ivar,
                            const struct jacobian *jacob,
                            const struct gvec *gvec, 
                            double *s2n_numer,
                            double *s2n_denom,
                            double *loglike);

int calculate_loglike_wt(const struct image *image, 
                         const struct image *weight,
                         const struct gvec *gvec, 
                         double *s2n_numer,
                         double *s2n_denom,
                         double *loglike);
int calculate_loglike(struct image *image, 
                      struct gvec *gvec, 
                      double ivar,
                      double *s2n_numer,
                      double *s2n_denom,
                      double *loglike);

#endif
