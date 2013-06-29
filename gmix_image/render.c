#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "defs.h"
#include "gmix.h"
#include "image.h"
#include "render.h"

#include "fmath.h"

/*
   fill a model with a gaussian mixture.  The model can be
   on a sub-grid (n > 1)

   Simply add to the existing pixel values!

 */
int fill_model_subgrid_jacob(struct image *image, 
                             const struct gmix *gmix, 
                             const struct jacobian *jacob,
                             int nsub)
{
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);

    double u=0, v=0;
    size_t col=0, row=0, irowsub=0, icolsub=0;

    double model_val=0, tval=0;
    double stepsize=0, ucolstep=0, vcolstep=0, offset=0, trow=0, tcol=0;
    int flags=0;

    if (!gmix_verify(gmix)) {
        flags |= GMIX_ERROR_NEGATIVE_DET;
        goto _fill_model_subgrid_bail;
    }
    if (nsub < 1) nsub=1;

    stepsize = 1./nsub;
    offset = (nsub-1)*stepsize/2.;

    // sub-step sizes in column direction
    ucolstep = stepsize*jacob->dudcol;
    vcolstep = stepsize*jacob->dvdcol;

    for (row=0; row<nrows; row++) {
        for (col=0; col<ncols; col++) {

            // start with existing value!
            model_val=IM_GET(image, row, col);

            // work over the subgrid
            tval=0;
            trow = row-offset;
            for (irowsub=0; irowsub<nsub; irowsub++) {

                tcol = col-offset;

                u=JACOB_PIX2U(jacob, trow, tcol);
                v=JACOB_PIX2V(jacob, trow, tcol);

                for (icolsub=0; icolsub<nsub; icolsub++) {
                    tval += GMIX_EVAL(gmix, u, v);
                    u += ucolstep;
                    v += vcolstep;
                }
                trow += stepsize;
            }

            tval /= (nsub*nsub);
            model_val += tval;


            if (!isfinite(model_val)) {
                model_val=0;
            }
            IM_SETFAST(image, row, col, model_val);

        } // cols
    } // rows

_fill_model_subgrid_bail:
    return flags;
}

/*
   fill a model with a gaussian mixture.  The model can be
   on a sub-grid (n > 1)

   Simply add to the existing pixel values!

 */
int fill_model_subgrid(struct image *image, 
                       struct gmix *gmix, 
                       int nsub)
{

    struct jacobian jacob;

    jacobian_set_identity(&jacob);

    return fill_model_subgrid_jacob(image,
                                    gmix, 
                                    &jacob,
                                    nsub);
}



// internal generic routine with optional pars
// fill in (model-data)/err
// gmix centers should be in the u,v plane
int
fill_ydiff_wt_jacob_generic(const struct image *image,
                            const struct image *weight, // either
                            double ivar,                // or
                            const struct jacobian *jacob,
                            const struct gmix *gmix,
                            struct image *diff_image)
{
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);

    double u=0, v=0;
    double diff=0;
    ssize_t col=0, row=0;

    double model_val=0, pixval=0;
    int flags=0;

    if (!gmix_verify(gmix)) {
        flags |= GMIX_ERROR_NEGATIVE_DET;
        goto _fill_ydiff_wt_jacob_bail;
    }

    if (ivar < 0) ivar=0.0;
    for (row=0; row<nrows; row++) {
        u=JACOB_PIX2U(jacob, row, 0);
        v=JACOB_PIX2V(jacob, row, 0);
        for (col=0; col<ncols; col++) {

            if (weight) {
                ivar=IM_GET(weight, row, col);
                if (ivar < 0) ivar=0.0; // fpack...
            }

            if (ivar > 0) {
                model_val=GMIX_EVAL(gmix, u, v);
                pixval=IM_GET(image, row, col);

                diff = model_val - pixval;
                diff *= sqrt(ivar);

                if (!isfinite(diff)) {
                    diff=GMIX_IMAGE_BIGNUM;
                }
                IM_SETFAST(diff_image, row, col, diff);
            }

            u += jacob->dudcol; v += jacob->dvdcol;
        } // cols
    } // rows


_fill_ydiff_wt_jacob_bail:
    return flags;

}

// fill in (model-data)/err
// gmix centers should be in the u,v plane
int fill_ydiff_jacob(const struct image *image,
                     double ivar,
                     const struct jacobian *jacob,
                     const struct gmix *gmix,
                     struct image *diff_image)
{

    struct image *junk_weight=NULL;
    return fill_ydiff_wt_jacob_generic(image,
                                       junk_weight,
                                       ivar,
                                       jacob,
                                       gmix,
                                       diff_image);

}



// fill in (model-data)/err
// gmix centers should be in the u,v plane
int fill_ydiff_wt_jacob(const struct image *image,
                        const struct image *weight,
                        const struct jacobian *jacob,
                        const struct gmix *gmix,
                        struct image *diff_image)
{

    double junk_ivar=0;
    return fill_ydiff_wt_jacob_generic(image,
                                       weight,
                                       junk_ivar,
                                       jacob,
                                       gmix,
                                       diff_image);
}

/*
   fill in (model-data)/err
*/
int fill_ydiff(struct image *image,
               double ivar,
               struct gmix *gmix,
               struct image *diff_image)
{

    struct image *junk_weight=NULL;
    struct jacobian jacob;
    jacobian_set_identity(&jacob);

    return fill_ydiff_wt_jacob_generic(image,
                                       junk_weight,
                                       ivar,
                                       &jacob,
                                       gmix,
                                       diff_image);
}



int calculate_loglike_margamp(struct image *image, 
                              struct gmix *gmix, 
                              double A,
                              double ierr,
                              double *loglike)
{
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);

    struct gauss *gauss=NULL;
    double u=0, v=0, uv=0, u2=0, v2=0;
    double chi2=0;
    size_t i=0, col=0, row=0;

    double model_val=0;
    double ymodsum=0; // sum of (image/err)
    double ymod2sum=0; // sum of (image/err)^2
    double norm=0;
    double B=0.; // sum(model*image/err^2)/A
    double *rowdata=NULL;
    int flags=0;

    if (!gmix_verify(gmix)) {
        flags |= GMIX_ERROR_NEGATIVE_DET;
        goto _calculate_loglike_bail;
    }


    *loglike=-9999.9e9;
    for (row=0; row<nrows; row++) {
        rowdata=IM_ROW(image, row);
        for (col=0; col<ncols; col++) {

            model_val=0;
            gauss=gmix->data;
            for (i=0; i<gmix->size; i++) {
                u = row-gauss->row;
                u2=u*u;

                v = col-gauss->col;
                v2 = v*v;
                uv = u*v;

                chi2=gauss->dcc*u2 + gauss->drr*v2 - 2.0*gauss->drc*uv;
                //model_val += gauss->norm*gauss->p*exp( -0.5*chi2 );
                if (chi2 < EXP_MAX_CHI2) {
                    model_val += gauss->norm*gauss->p*expd( -0.5*chi2 );
                }

                gauss++;
            } // gmix

            ymodsum += model_val;
            ymod2sum += model_val*model_val;
            B += (*rowdata)*model_val;

            rowdata++;
        } // cols
    } // rows

    ymodsum *= ierr;
    ymod2sum *= ierr*ierr;
    norm = sqrt(ymodsum*ymodsum*A/ymod2sum);

    // renorm so A is fixed; also extra factor of 1/err^2 and 1/A
    B *= (norm/ymodsum*ierr*ierr/A);

    *loglike = 0.5*A*B*B;

_calculate_loglike_bail:
    return flags;
}


// using a weight image and jacobian.  Not tested.
// row0,col0 is center of coordinate system
// gmix centers should be in the u,v plane
// combine s2n_numer and s2n_denom as below
// can sum over multiple images
//s2n = s2n_numer/sqrt(s2n_denom);

/*
int calculate_loglike_wt_jacob_generic(const struct image *image, 
                                       const struct image *weight, // either
                                       double ivar,                // or
                                       const struct jacobian *jacob,
                                       const struct gmix *gmix, 
                                       double *s2n_numer,
                                       double *s2n_denom,
                                       double *loglike)
{
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);

    double u=0, v=0;
    double diff=0;
    ssize_t col=0, row=0;

    double model_val=0;
    double pixval=0, *ptr=NULL;
    int flags=0;

    double _u=0;
    double _v=0;
    double _chi2=0;
    double tmp=0, tloglike=0, ts2n_numer=0, ts2n_denom=0;


    (*s2n_numer)=0;
    (*s2n_denom)=0;
    (*loglike)=0;

    if (!gmix_verify(gmix)) {
        *loglike=-9999.9e9;
        flags |= GMIX_ERROR_NEGATIVE_DET;
        goto _calculate_loglike_wt_jacob_bail;
    }

    if (ivar < 0) ivar=0.0;

    ptr=IM_GETP(image,0,0);

    for (row=0; row<nrows; row++) {
        u=JACOB_PIX2U(jacob, row, 0);
        v=JACOB_PIX2V(jacob, row, 0);
        for (col=0; col<ncols; col++) {

            if (weight) {
                ivar=IM_GET(weight, row, col);
                if (ivar < 0) ivar=0.0; // fpack...
            }

            if (ivar > 0) {
                model_val=0.0;
                int _i=0;
                struct gauss *gauss=gmix->data;

                for (_i=0; _i<gmix->size; _i++) {
                    _u = u-gauss->row;
                    _v = v-gauss->col;
                    _chi2 = gauss->dcc*_u*_u + gauss->drr*_v*_v - 2.0*gauss->drc*_u*_v;

                    if (_chi2 < EXP_MAX_CHI2) {
                        _chi2 *= (-0.5);
                        tmp = expd(_chi2);
                        tmp *= gauss->pnorm;
                        model_val += tmp;
                    }
                    gauss++;
                }

                // (model_val-pixval)**2*ivar
                //pixval=IM_GET(image, row, col);
                //diff = model_val - pixval;

                pixval=*ptr;

                diff=model_val;
                diff -= pixval;

                // diff*diff*ivar;
                diff *= diff;
                diff *= ivar;
                tloglike += diff;

                /// pixval*model_val*ivar
                tmp = pixval;
                tmp *= model_val;
                tmp *= ivar;
                ts2n_numer += tmp;

                //  model_val*model_val*ivar;
                tmp = model_val;
                tmp *= model_val;
                tmp *= ivar;
                ts2n_denom += tmp;
            }

            u += jacob->dudcol; v += jacob->dvdcol;
            ptr ++;
        } // cols
    } // rows

    //(*loglike) *= (-0.5);
    (*loglike) = -0.5*tloglike;
    (*s2n_numer) = ts2n_numer;
    (*s2n_denom) = ts2n_denom;

    if (!isfinite((*loglike))) {
        (*loglike) = -GMIX_IMAGE_BIGNUM;
    }

_calculate_loglike_wt_jacob_bail:
    return flags;
}
*/

int calculate_loglike_wt_jacob_generic(const struct image *image, 
                                       const struct image *weight, // either
                                       double ivar,                // or
                                       const struct jacobian *jacob,
                                       const struct gmix *gmix, 
                                       double *s2n_numer,
                                       double *s2n_denom,
                                       double *loglike)
{
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);

    double u=0, v=0;
    double diff=0;
    ssize_t col=0, row=0;

    double model_val=0;
    double pixval=0;
    int flags=0;

    (*s2n_numer)=0;
    (*s2n_denom)=0;
    (*loglike)=0;

    if (!gmix_verify(gmix)) {
        *loglike=-9999.9e9;
        flags |= GMIX_ERROR_NEGATIVE_DET;
        goto _calculate_loglike_wt_jacob_bail;
    }

    if (ivar < 0) ivar=0.0;
    for (row=0; row<nrows; row++) {
        u=JACOB_PIX2U(jacob, row, 0);
        v=JACOB_PIX2V(jacob, row, 0);
        for (col=0; col<ncols; col++) {

            if (weight) {
                ivar=IM_GET(weight, row, col);
                if (ivar < 0) ivar=0.0; // fpack...
            }

            if (ivar > 0) {
                model_val=GMIX_EVAL(gmix, u, v);
                pixval=IM_GET(image, row, col);
                diff = model_val - pixval;

                (*loglike) += diff*diff*ivar;

                (*s2n_numer) += pixval*model_val*ivar;
                (*s2n_denom) += model_val*model_val*ivar;
            }

            u += jacob->dudcol; v += jacob->dvdcol;
        } // cols
    } // rows

    (*loglike) *= (-0.5);
    if (!isfinite((*loglike))) {
        (*loglike) = -GMIX_IMAGE_BIGNUM;
    }

_calculate_loglike_wt_jacob_bail:
    return flags;
}



// using a weight image and jacobian.  Not tested.
// row0,col0 is center of coordinate system
// gmix centers should be in the u,v plane
// combine s2n_numer and s2n_denom as below
// can sum over multiple images
//s2n = s2n_numer/sqrt(s2n_denom);

int calculate_loglike_wt_jacob(const struct image *image, 
                               const struct image *weight,
                               const struct jacobian *jacob,
                               const struct gmix *gmix, 
                               double *s2n_numer,
                               double *s2n_denom,
                               double *loglike)
{

    double junk_ivar=0;
    return calculate_loglike_wt_jacob_generic(image, 
                                              weight,
                                              junk_ivar,
                                              jacob,
                                              gmix, 
                                              s2n_numer,
                                              s2n_denom,
                                              loglike);

}

// row0,col0 is center of coordinate system
// gmix centers should be in the u,v plane
// combine s2n_numer and s2n_denom as below
// can sum over multiple images
//s2n = s2n_numer/sqrt(s2n_denom);

int calculate_loglike_jacob(const struct image *image, 
                            double ivar,
                            const struct jacobian *jacob,
                            const struct gmix *gmix, 
                            double *s2n_numer,
                            double *s2n_denom,
                            double *loglike)
{

    struct image *junk_weight=NULL;

    return calculate_loglike_wt_jacob_generic(image, 
                                              junk_weight,
                                              ivar,
                                              jacob,
                                              gmix, 
                                              s2n_numer,
                                              s2n_denom,
                                              loglike);

}

// using a weight image.  Not tested.
// combine s2n_numer and s2n_denom as below
// can sum over multiple images
//s2n = s2n_numer/sqrt(s2n_denom);
int calculate_loglike_wt(const struct image *image, 
                         const struct image *weight,
                         const struct gmix *gmix, 
                         double *s2n_numer,
                         double *s2n_denom,
                         double *loglike)
{

    double junk_ivar=0;
    struct jacobian jacob;
    jacobian_set_identity(&jacob);

    return calculate_loglike_wt_jacob_generic(image, 
                                              weight,
                                              junk_ivar,
                                              &jacob,
                                              gmix, 
                                              s2n_numer,
                                              s2n_denom,
                                              loglike);

}


int calculate_loglike(struct image *image, 
                      struct gmix *gmix, 
                      double ivar,
                      double *s2n_numer,
                      double *s2n_denom,
                      double *loglike)
{

    int flags=0;
    struct image *junk_weight=NULL;

    struct jacobian jacob;
    jacobian_set_identity(&jacob);

    flags=calculate_loglike_wt_jacob_generic(image, 
                                             junk_weight,
                                             ivar,
                                             &jacob,
                                             gmix, 
                                             s2n_numer,
                                             s2n_denom,
                                             loglike);


    return flags;

}

