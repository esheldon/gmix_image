#ifndef _GVEC_HEADER_GUARD
#define _GVEC_HEADER_GUARD

//#include "matrix.h"

/*
struct gauss2 {
    double p;
    double norm;

    double cen[2];
    double cov[2][2];
};
struct gauss2 {
    double p;
    double norm;

    struct vec2;
    struct mtx2;
};
*/

struct gauss {
    double p;
    double row;
    double col;
    double irr;
    double irc;
    double icc;
    double det;

    double drr;
    double drc;
    double dcc;

    double norm; // 1/(2*pi*sqrt(det))
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
    GAPPROX_DEV,
    GAPPROX_BD
};
 
struct gvec *gvec_new(size_t n);
struct gvec *gvec_free(struct gvec *self);
void gvec_set_dets(struct gvec *self);

// make sure pointer not null and det>0 for all gauss
int gvec_verify(const struct gvec *self);

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

int gvec_copy(const struct gvec *self, struct gvec* dest);
void gvec_print(const struct gvec *self, FILE* fptr);

// calculate the weighted sum of the moments
//  sum_gi( p*(irr + icc )
double gvec_wmomsum(const struct gvec* gvec);

void gvec_get_cen(const struct gvec *gvec, double *row, double *col);
// set the overall centroid.  Note individual gaussians can have
// a different center
void gvec_set_cen(struct gvec *gvec, double row, double col);

double gvec_get_T(const struct gvec *self);
double gvec_get_psum(const struct gvec *gvec);
// set the overall sum(p)
void gvec_set_psum(struct gvec *gvec, double psum);

// 0 returned if a zero determinant is found somewhere, else 1
//int gvec_wmean_center(const struct gvec* gvec, struct vec2* mu_new);

//void gvec_wmean_covar(const struct gvec* gvec, struct mtx2 *cov);

/* convolution results in an nobj*npsf total gaussians */
struct gvec *gvec_convolve(struct gvec *obj_gvec, 
                           struct gvec *psf_gvec);


/* full parameters list
   [pi,rowi,coli,irri,irci,icci,...]
*/
struct gvec *gvec_from_pars(double *pars, int size);

/* coellip list
   [row,col,e1,e2,Tmax,f2,f3,...,p1,p2,p3..]
 */
struct gvec *gvec_from_coellip(double *pars, int size);
struct gvec *gvec_from_coellip_Tfrac(double *pars, int size);

/* 
   Generate a gvec from the inputs pars assuming an appoximate
   gaussian representation of an exponential disk.  Values
   from Hogg and Lang

   pars should be [row,col,e1,e2,T,p]

   T = sum(pi*Ti)/sum(pi)

   The p and F values are chosen to make this so
*/
struct gvec *gvec_from_pars_exp4(double *pars, int size);
struct gvec *gvec_from_pars_exp6(double *pars, int size);



/* 
   Generate a gvec from the inputs pars assuming an appoximate
   10-gaussian representation of a devauc profile.

   pars should be [row,col,e1,e2,T,p]

   T = sum(pi*Ti)/sum(pi)

   The p and F values are chosen to make this so
*/

struct gvec *gvec_from_pars_dev10(double *pars, int size);

/* similar to above but for a turbulent psf */
struct gvec *gvec_from_pars_turb(double *pars, int size);

/*
   co-elliptical bulg+disk

   pars should be [row,col,e1,e2,Texp,Tdev,Fexp,Fdev]

   npars is 8
*/

struct gvec *gvec_from_pars_bd(double *pars, int size);


#define GAUSS_EVAL(gauss, rowval, colval) ({                   \
    double _u = (rowval)-(gauss)->row;                         \
    double _v = (colval)-(gauss)->col;                         \
                                                               \
    double _chi2 =                                             \
        (gauss)->dcc*_u*_u                                     \
        + (gauss)->drr*_v*_v                                   \
        - 2.0*(gauss)->drc*_u*_v;                              \
                                                               \
    double _val=0.0;                                           \
    if (_chi2 < EXP_MAX_CHI2) {                                \
        _val = (gauss)->norm*(gauss)->p*expd( -0.5*_chi2 );    \
    }                                                          \
                                                               \
    _val;                                                      \
})


#define GVEC_EVAL(gmix, rowval, colval) ({                     \
    int _i=0;                                                  \
    double _val=0.0;                                           \
    struct gauss *_gauss=(gmix)->data;                         \
    for (_i=0; _i<(gmix)->size; _i++) {                        \
        _val += GAUSS_EVAL(_gauss, (rowval), (colval));        \
        _gauss++;                                              \
    }                                                          \
    _val;                                                      \
})


// evaluate the gvec and store in "val"
// also store the number of evaluations that were done
#define GVEC_EVAL_COUNT(gmix, rowval, colval, val, count) {    \
    int _i=0;                                                  \
    double _u;                                                 \
    double _v;                                                 \
    double _chi2;                                              \
                                                               \
    (val)=0;                                                   \
    (count)=0;                                                 \
    struct gauss *_gauss=(gmix)->data;                         \
    for (_i=0; _i<(gmix)->size; _i++) {                        \
                                                               \
        _u = (rowval)-(gauss)->row;                            \
        _v = (colval)-(gauss)->col;                            \
                                                               \
        _chi2=(gauss)->dcc*_u*_u                               \
            + (gauss)->drr*_v*_v                               \
        - 2.0*(gauss)->drc*_u*_v;                              \
                                                               \
        if (_chi2 < EXP_MAX_CHI2) {                                  \
            (val) += (gauss)->norm*(gauss)->p*expd( -0.5*_chi2 );    \
            (count) += 1;                                            \
        }                                                            \
                                                                     \
        _gauss++;                                              \
    }                                                          \
}



#endif
