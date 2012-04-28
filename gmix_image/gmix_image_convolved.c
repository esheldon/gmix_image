#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "gmix_image.h"
#include "gmix_image_convolved.h"
#include "image.h"
#include "gvec.h"
#include "defs.h"

int gmix_image_convolved(struct gmix* self,
                         struct image *image, 
                         struct gvec *gvec,
                         struct gvec *gvec_psf,
                         size_t *iter,
                         double *fdiff)
{
    int flags=0;
    size_t ngauss = gvec->size;
    double wmomlast=0, wmom=0, wmomdiff=0;

    double sky=image_sky(image);
    double counts=image_counts(image);
    size_t npoints = IMSIZE(image);

    struct iter *iter_struct = iter_new(ngauss);

    iter_struct->nsky = sky/counts;
    iter_struct->psky = sky/(counts/npoints);

    wmomlast=-9999;
    *iter=0;
    while (*iter < self->maxiter) {
        if (self->verbose)
            gvec_print(gvec,stderr);

        flags = gmix_get_sums_convolved(image, gvec, gvec_psf, iter_struct);

        if (flags!=0) {
            fprintf(stderr,"error found at iter %lu\n", *iter);
            goto _gmix_image_convolved_bail;
        }

        gmix_set_gvec_fromiter_convolved(gvec, gvec_psf, iter_struct);

        iter_struct->psky = iter_struct->skysum;
        iter_struct->nsky = iter_struct->psky/npoints;

        wmom = gvec_wmomsum(gvec);

        wmom /= iter_struct->psum;
        wmomdiff = fabs(wmom-wmomlast);
        *fdiff = fabs(wmomdiff/wmom);
        if (*fdiff < self->tol) {
            break;
        }
        wmomlast = wmom;
        (*iter)++;
    }

_gmix_image_convolved_bail:
    if (self->maxiter == (*iter)) {
        flags += GMIX_ERROR_MAXIT;
    }
    iter_struct = iter_free(iter_struct);

    return flags;
}

int gmix_get_sums_convolved(struct image *image,
                            struct gvec *gvec,
                            struct gvec *gvec_psf,
                            struct iter* iter)
{
    int flags=0;
    double igrat=0, imnorm=0, gtot=0, wtau=0;
    double u=0, v=0, uv=0, u2=0, v2=0;
    size_t i=0, col=0, row=0;
    struct gauss* gauss=NULL;
    struct sums *sums=NULL;
    //double chi2=0,b=0;

    iter_clear(iter);
    for (col=0; col<image->ncols; col++) {
        for (row=0; row<image->nrows; row++) {

            imnorm=IMGET(image,row,col);
            imnorm /= image->counts;

            gtot=0;
            gauss = &gvec->data[0];
            sums = &iter->sums[0];
            for (i=0; i<gvec->size; i++) {
                u = (row-gauss->row);
                v = (col-gauss->col);

                u2 = u*u; v2 = v*v; uv = u*v;
                sums->gi = gmix_evaluate_convolved(gauss,
                                                   gvec_psf,
                                                   u2,uv,v2,
                                                   &flags);

                if (flags != 0) {
                    goto _gmix_get_sums_convolved_bail;
                }

                gtot += sums->gi;

                sums->trowsum = row*sums->gi;
                sums->tcolsum = col*sums->gi;
                sums->tu2sum  = u2*sums->gi;
                sums->tuvsum  = uv*sums->gi;
                sums->tv2sum  = v2*sums->gi;

                gauss++;
                sums++;
            }
            gtot += iter->nsky;
            igrat = imnorm/gtot;
            sums = &iter->sums[0];
            for (i=0; i<gvec->size; i++) {
                // this is gi/gtot*imnorm
                wtau = sums->gi*igrat;  

                iter->psum += wtau;
                sums->pnew += wtau;

                sums->rowsum += sums->trowsum*igrat;
                sums->colsum += sums->tcolsum*igrat;
                sums->u2sum  += sums->tu2sum*igrat;
                sums->uvsum  += sums->tuvsum*igrat;
                sums->v2sum  += sums->tv2sum*igrat;
                sums++;
            }
            iter->skysum += iter->nsky*imnorm/gtot;

        } // rows
    } // cols

_gmix_get_sums_convolved_bail:
    return flags;
}

/* mathematically this comes down to averaging over
   the convolved gaussians.

       sum_i( g*psf_i*p_i )/sum(p_i)

   where g*psf_i is a convolution with psf i.
   The convolution comes down to adding the covariance
   matrices and then we have to do the proper normalization
   using the convolved determinant

   u is (row-rowcen) v is (col-colcen)

   We will bail if *either* the gaussian determinant is
   zero or the convolved gaussian determinant is zero
*/
double gmix_evaluate_convolved(struct gauss *gauss,
                               struct gvec *gvec_psf,
                               double u2, double uv, double v2,
                               int *flags)
{
    size_t i=0;
    struct gauss *psf=NULL;
    double irr=0, irc=0, icc=0, det=0, chi2=0, b=0;
    double psum=0;
    double val=0;

    if (gauss->det <= 0) {
        wlog("found obj det: %.16g\n", det);
        *flags |= GMIX_ERROR_NEGATIVE_DET;
        val=-9999;
        goto _gmix_eval_conv_bail;
    }

    psf = &gvec_psf->data[0];
    for (i=0; i<gvec_psf->size; i++) {

        irr = gauss->irr + psf->irr;
        irc = gauss->irc + psf->irc;
        icc = gauss->icc + psf->icc;

        det = irr*icc - irc*irc;
        if (det <= 0) {
            wlog("found convolved det: %.16g\n", det);
            *flags |= GMIX_ERROR_NEGATIVE_DET;
            val=-9999;
            goto _gmix_eval_conv_bail;
        }

        chi2=icc*u2 + irr*v2 - 2.0*irc*uv;
        chi2 /= det;

        b = M_TWO_PI*sqrt(det);
        val += psf->p*exp( -0.5*chi2 )/b;
        psum += psf->p;
        psf++;
    }

    val *= gauss->p/psum;

_gmix_eval_conv_bail:
    return val;
}


void gmix_set_gvec_fromiter_convolved(struct gvec *gvec, 
                                      struct gvec *gvec_psf,
                                      struct iter* iter)
{
    struct sums *sums=NULL;
    struct gauss *gauss=NULL;
    size_t i=0;

    double psf_irr_mean=0,psf_irc_mean=0, psf_icc_mean=0;
    double psum=0;

    // average the moments.  We should store this in the gvec
    gauss=gvec_psf->data;
    for (i=0; i<gvec_psf->size; i++) {
        psf_irr_mean += gauss->p*gauss->irr;
        psf_irc_mean += gauss->p*gauss->irc;
        psf_icc_mean += gauss->p*gauss->icc;
        psum+=gauss->p;
        gauss++;
    }
    psf_irr_mean /= psum;
    psf_irc_mean /= psum;
    psf_icc_mean /= psum;

    sums=iter->sums;
    gauss=gvec->data;
    for (i=0; i<gvec->size; i++) {
        gauss->p   = sums->pnew;
        gauss->row = sums->rowsum/sums->pnew;
        gauss->col = sums->colsum/sums->pnew;

        gauss->irr = sums->u2sum/sums->pnew - psf_irr_mean;
        gauss->irc = sums->uvsum/sums->pnew - psf_irc_mean;
        gauss->icc = sums->v2sum/sums->pnew - psf_icc_mean;

        gauss->det = gauss->irr*gauss->icc - gauss->irc*gauss->irc;

        sums++;
        gauss++;
    }
}


