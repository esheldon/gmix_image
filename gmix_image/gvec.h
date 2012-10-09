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

enum gapprox {
    GAPPROX_EXP,
    GAPPROX_DEV
};
 
struct gvec *gvec_new(size_t n);
struct gvec *gvec_free(struct gvec *self);
void gvec_set_dets(struct gvec *self);

// make sure pointer not null and det>0 for all gauss
int gvec_verify(struct gvec *self);

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

/* convolution results in an nobj*npsf total gaussians */
struct gvec *gvec_convolve(struct gvec *obj_gvec, 
                           struct gvec *psf_gvec);


/* full parameters list
   [pi,rowi,coli,irri,irci,icci,...]
*/
struct gvec *pars_to_gvec(double *pars, int sz);
/* coellip list
   [row,col,e1,e2,Tmax,f2,f3,...,p1,p2,p3..]
 */
struct gvec *coellip_pars_to_gvec(double *pars, int sz);

/* 
   Generate a gvec from the inputs pars assuming an appoximate
   3-gaussian representation of an exponential disk or devauc.

   One component is nearly a delta function

   pars should be [row,col,e1,e2,T,p]

   T = sum(pi*Ti)/sum(pi)

   The p and F values are chosen to make this so
*/
struct gvec *gapprox_pars_to_gvec(double *pars, enum gapprox type);

#endif
