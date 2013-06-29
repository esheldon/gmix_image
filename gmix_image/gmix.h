#ifndef _GMIX_HEADER_GUARD
#define _GMIX_HEADER_GUARD

#include <stdint.h>


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

struct gmix {
    size_t size;
    struct gauss* data;

    // these only make sense for same-center gaussians
    // e.g. the psf
    double total_irr;
    double total_irc;
    double total_icc;
};

enum gmix_model {
    GMIX_FULL=0,
    GMIX_COELLIP=1,
    GMIX_TURB=2,
    GMIX_EXP=3,
    GMIX_DEV=4,
    GMIX_COELLIP_TFRAC=5,
    GMIX_BD=6
};

long gmix_get_simple_ngauss(enum gmix_model model, long *flags);
long gmix_get_coellip_ngauss(long npars, long *flags);
long gmix_get_full_ngauss(long npars, long *flags);

struct gmix *gmix_new(size_t n, long *flags);

struct gmix* gmix_new_empty_simple(enum gmix_model model, long *flags);
struct gmix* gmix_new_empty_coellip(long npars, long *flags);
struct gmix* gmix_new_empty_full(long npars, long *flags);

struct gmix* gmix_new_model(enum gmix_model model, double *pars, long npars, long *flags);
struct gmix *gmix_new_coellip(double *pars, long npars, long *flags);


void gmix_fill_model(struct gmix *self,
                     enum gmix_model model,
                     double *pars,
                     long npars,
                     long *flags);

void gmix_fill_full(struct gmix *self, double *pars, long npars, long *flags);
void gmix_fill_coellip(struct gmix *self, double *pars, long npars, long *flags);
void gmix_fill_exp6(struct gmix *self, double *pars, long npars, long *flags);
void gmix_fill_dev10(struct gmix *self, double *pars, long npars, long *flags);
void gmix_fill_bd(struct gmix *self, double *pars, long npars, long *flags);
void gmix_fill_turb3(struct gmix *self, double *pars, long npars, long *flags);


struct gmix *gmix_free(struct gmix *self);
void gmix_set_dets(struct gmix *self);

// make sure pointer not null and det>0 for all gauss
long gmix_verify(const struct gmix *self);

// only makes sense for same center, e.g. psf
void gmix_set_total_moms(struct gmix *self);

// this is actually kind of unclear to use in practice since it is easy to
// screw up which parameters go where
void gauss_set(struct gauss* self, 
               double p, 
               double row, 
               double col,
               double irr,
               double irc,
               double icc,
               long *flags);

void gmix_copy(const struct gmix *self, struct gmix* dest, long *flags);
struct gmix *gmix_newcopy(const struct gmix *self, long *flags);
void gmix_print(const struct gmix *self, FILE* fptr);

// calculate the weighted sum of the moments
//  sum_gi( p*(irr + icc )
double gmix_wmomsum(const struct gmix* gmix);

void gmix_get_cen(const struct gmix *gmix, double *row, double *col);
// set the overall centroid.  Note individual gaussians can have
// a different center
void gmix_set_cen(struct gmix *gmix, double row, double col);

double gmix_get_T(const struct gmix *self);
double gmix_get_psum(const struct gmix *gmix);
// set the overall sum(p)
void gmix_set_psum(struct gmix *gmix, double psum);

// 0 returned if a zero determinant is found somewhere, else 1
//int gmix_wmean_center(const struct gmix* gmix, struct vec2* mu_new);

//void gmix_wmean_covar(const struct gmix* gmix, struct mtx2 *cov);

/* convolution results in an nobj*npsf total gaussians */
struct gmix *gmix_convolve(const struct gmix *obj_gmix, 
                           const struct gmix *psf_gmix,
                           long *flags);

void gmix_convolve_fill(struct gmix *self, 
                        const struct gmix *obj_gmix, 
                        const struct gmix *psf_gmix,
                        long *flags);

// old
//struct gmix *gmix_from_pars(double *pars, long npars);
//struct gmix *gmix_new_coellip_Tfrac(double *pars, long npars);

/* 
   Generate a gmix from the inputs pars assuming an appoximate
   gaussian representation of an exponential disk.  Values
   from Hogg and Lang

   pars should be [row,col,e1,e2,T,p]

   T = sum(pi*Ti)/sum(pi)

   The p and F values are chosen to make this so
*/
//struct gmix *gmix_from_pars_exp4(double *pars, long npars);
//struct gmix *gmix_from_pars_exp6(double *pars, long npars);



/* 
   Generate a gmix from the inputs pars assuming an appoximate
   10-gaussian representation of a devauc profile.

   pars should be [row,col,e1,e2,T,p]

   T = sum(pi*Ti)/sum(pi)

   The p and F values are chosen to make this so
*/

//struct gmix *gmix_from_pars_dev10(double *pars, long npars);

/* similar to above but for a turbulent psf */
//struct gmix *gmix_from_pars_turb(double *pars, long npars);

/*
   co-elliptical bulg+disk

   pars should be [row,col,e1,e2,Texp,Tdev,Fexp,Fdev]

   npars is 8
*/

//struct gmix *gmix_from_pars_bd(double *pars, long npars);


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


#define GMIX_EVAL(gmix, rowval, colval) ({                     \
    int _i=0;                                                  \
    double _val=0.0;                                           \
    struct gauss *_gauss=(gmix)->data;                         \
    for (_i=0; _i<(gmix)->size; _i++) {                        \
        _val += GAUSS_EVAL(_gauss, (rowval), (colval));        \
        _gauss++;                                              \
    }                                                          \
    _val;                                                      \
})


// evaluate the gmix and store in "val"
// also store the number of evaluations that were done
#define GMIX_EVAL_COUNT(gmix, rowval, colval, val, count) {    \
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
