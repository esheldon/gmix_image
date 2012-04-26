/*
 
    Algorithm is simple:

    Start with a guess for the N gaussians.

    Then the new estimate for the gaussian weight "p" for gaussian gi is

        pnew[gi] = sum_pix( gi[pix]/gtot[pix]*imnorm[pix] )

    where imnorm is image/sum(image) and gtot[pix] is
        
        gtot[pix] = sum_gi(gi[pix]) + nsky

    and nsky is the sky/sum(image)
    
    These new p[gi] can then be used to update the mean and covariance
    as well.  To update the mean in coordinate x

        mx[gi] = sum_pix( gi[pix]/gtot[pix]*imnorm[pix]*x )/pnew[gi]
 
    where x is the pixel value in either row or column.

    Similarly for the covariance for coord x and coord y.

        cxy = sum_pix(  gi[pix]/gtot[pix]*imnorm[pix]*(x-xc)*(y-yc) )/pcen[gi]

    setting x==y gives the diagonal terms.

    Then repeat until some tolerance in the moments is achieved.

    These calculations can be done very efficiently within a single loop,
    with a pixel lookup only once per loop.
    
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "gmix_image.h"
#include "image.h"
#include "gvec.h"
#include "defs.h"


int gmix_image(struct gmix* self,
               struct image *image, 
               struct gvec *gvec,
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

        flags = gmix_get_sums(image, gvec, iter_struct);

        if (flags!=0)
            goto _gmix_image_new_bail;

        gmix_set_gvec_fromiter(gvec, iter_struct);

        iter_struct->psky = iter_struct->skysum;
        iter_struct->nsky = iter_struct->psky/npoints;

        wmom = gvec_wmomsum(gvec);

        wmom /= iter_struct->psum;
        wmomdiff = fabs(wmom-wmomlast);
        *fdiff = wmomdiff/wmom;
        if (*fdiff < self->tol) {
            break;
        }
        wmomlast = wmom;
        (*iter)++;
    }

_gmix_image_new_bail:
    if (self->maxiter == (*iter)) {
        flags += GMIX_ERROR_MAXIT;
    }
    iter_struct = iter_free(iter_struct);

    return flags;
}






int gmix_get_sums(struct image *image,
                  struct gvec *gvec,
                  struct iter* iter)
{
    int flags=0;
    double igrat=0, imnorm=0, gtot=0, tau=0, b=0, chi2=0;
    double u=0, v=0, uv=0, u2=0, v2=0;
    size_t i=0, col=0, row=0;
    struct gauss* gauss=NULL;
    struct sums *sums=NULL;

    iter_clear(iter);
    for (col=0; col<image->ncols; col++) {
        for (row=0; row<image->nrows; row++) {

            imnorm=IMGET(image,row,col);
            imnorm /= image->counts;

            gtot=0;
            gauss = &gvec->data[0];
            sums = &iter->sums[0];
            for (i=0; i<gvec->size; i++) {
                if (gauss->det <= 0) {
                    wlog("found det: %.16g\n", gauss->det);
                    flags+=GMIX_ERROR_NEGATIVE_DET;
                    goto _gmix_get_sums_bail;
                }
                u = (row-gauss->row);
                v = (col-gauss->col);

                u2 = u*u; v2 = v*v; uv = u*v;

                chi2=gauss->icc*u2 + gauss->irr*v2 - 2.0*gauss->irc*uv;
                chi2 /= gauss->det;
                b = M_TWO_PI*sqrt(gauss->det);

                sums->gi = gauss->p*exp( -0.5*chi2 )/b;
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
                tau = sums->gi*igrat;  // Dave's tau*imnorm
                iter->psum += tau;

                sums->pnew += tau;
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

_gmix_get_sums_bail:
    return flags;
}



void gmix_set_gvec_fromiter(struct gvec* gvec, struct iter* iter)
{
    struct sums *sums=iter->sums;
    struct gauss *gauss = gvec->data;
    size_t i=0;
    for (i=0; i<gvec->size; i++) {
        gauss->p   = sums->pnew;
        gauss->row = sums->rowsum/sums->pnew;
        gauss->col = sums->colsum/sums->pnew;
        gauss->irr = sums->u2sum/sums->pnew;
        gauss->irc = sums->uvsum/sums->pnew;
        gauss->icc = sums->v2sum/sums->pnew;
        gauss->det = gauss->irr*gauss->icc - gauss->irc*gauss->irc;

        sums++;
        gauss++;
    }
}


struct iter *iter_new(size_t ngauss)
{
    struct iter *self=calloc(1,sizeof(struct iter));
    if (self == NULL) {
        wlog("could not allocate iter struct, bailing\n");
        exit(EXIT_FAILURE);
    }

    self->ngauss=ngauss;

    self->sums = calloc(ngauss, sizeof(struct sums));
    if (self->sums == NULL) {
        wlog("could not allocate iter struct, bailing\n");
        exit(EXIT_FAILURE);
    }

    return self;
}

struct iter *iter_free(struct iter *self)
{
    if (self) {
        free(self->sums);
        free(self);
    }
    return NULL;
}

/* we don't clear psky or nsky or sums */
void iter_clear(struct iter *self)
{
    self->gtot=0;
    self->skysum=0;
    self->psum=0;
    memset(self->sums,0,self->ngauss*sizeof(struct sums));
}














/*
int gmix_image_old(struct gmix* self,
               struct image *image, 
               struct gvec *gvec,
               size_t *iter)
{
    int flags=0;
    size_t i=0;
    size_t ngauss = gvec->size;
    size_t nbytes = gvec->size*sizeof(double);
    double chi2=0, b=0;
    double u=0, v=0, uv=0, u2=0, v2=0, igrat=0;
    double gtot=0, imnorm=0, skysum=0.0, tau=0;
    double wmomlast=0, wmom=0, wmomdiff=0, psum=0;

    struct gauss* gauss=NULL;

    double sky=image_sky(image);
    double counts=image_counts(image);
    size_t npoints = IMSIZE(image);

    double nsky = sky/counts;
    double psky = sky/(counts/npoints);

    // these are all stack allocated

    double *gi = alloca(nbytes);
    double *trowsum = alloca(nbytes);
    double *tcolsum = alloca(nbytes);
    double *tu2sum = alloca(nbytes);
    double *tuvsum = alloca(nbytes);
    double *tv2sum = alloca(nbytes);

    // these need to be zeroed on each iteration
    double *pnew = alloca(nbytes);
    double *rowsum = alloca(nbytes);
    double *colsum = alloca(nbytes);
    double *u2sum = alloca(nbytes);
    double *uvsum = alloca(nbytes);
    double *v2sum = alloca(nbytes);

    wmomlast=-9999;
    *iter=0;
    while (*iter < self->maxiter) {
        if (self->verbose)
            gvec_print(gvec,stderr);

        skysum=0;
        psum=0;
        memset(pnew,0,nbytes);
        memset(rowsum,0,nbytes);
        memset(colsum,0,nbytes);
        memset(u2sum,0,nbytes);
        memset(uvsum,0,nbytes);
        memset(v2sum,0,nbytes);

        for (size_t col=0; col<image->ncols; col++) {
            for (size_t row=0; row<image->nrows; row++) {

                imnorm=IMGET(image,row,col);
                imnorm /= counts;

                gtot=0;
                gauss = &gvec->data[0];
                for (i=0; i<ngauss; i++) {
                    if (gauss->det <= 0) {
                        wlog("found det: %.16g\n", gauss->det);
                        flags+=GMIX_ERROR_NEGATIVE_DET;
                        goto _gmix_image_bail;
                    }
                    u = (row-gauss->row);
                    v = (col-gauss->col);

                    u2 = u*u; v2 = v*v; uv = u*v;

                    chi2=gauss->icc*u2 + gauss->irr*v2 - 2.0*gauss->irc*uv;
                    chi2 /= gauss->det;
                    b = M_TWO_PI*sqrt(gauss->det);

                    gi[i] = gauss->p*exp( -0.5*chi2 )/b;
                    gtot += gi[i];

                    trowsum[i] = row*gi[i];
                    tcolsum[i] = col*gi[i];
                    tu2sum[i]  = u2*gi[i];
                    tuvsum[i]  = uv*gi[i];
                    tv2sum[i]  = v2*gi[i];

                    gauss++;
                }
                gtot += nsky;
                igrat = imnorm/gtot;
                for (i=0; i<ngauss; i++) {
                    tau = gi[i]*igrat;  // Dave's tau*imnorm
                    pnew[i] += tau;
                    psum += tau;

                    rowsum[i] += trowsum[i]*igrat;
                    colsum[i] += tcolsum[i]*igrat;
                    u2sum[i]  += tu2sum[i]*igrat;
                    uvsum[i]  += tuvsum[i]*igrat;
                    v2sum[i]  += tv2sum[i]*igrat;
                }
                skysum += nsky*imnorm/gtot;

            } // rows
        } // cols

        psky = skysum;
        nsky = psky/npoints;

        gmix_set_gvec(gvec,pnew,rowsum,colsum,u2sum,uvsum,v2sum);

        wmom=0;
        gauss=gvec->data;
        for (i=0; i<ngauss; i++) {
            wmom += gauss->p*gauss->irr + gauss->p*gauss->icc;
            gauss++;
        }

        wmom /= psum;
        wmomdiff = fabs(wmom-wmomlast);
        //wlog("iter: %lu  diffrat: %.16g\n", *iter, wmomdiff/wmom);
        if (wmomdiff/wmom < self->tol) {
            break;
        }
        wmomlast = wmom;
        (*iter)++;
    }

_gmix_image_bail:
    if (self->maxiter == (*iter)) {
        flags += GMIX_ERROR_MAXIT;
    }
    return flags;
}
*/


