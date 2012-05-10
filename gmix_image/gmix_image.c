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
               struct gvec *gvec_psf,
               size_t *iter,
               double *fdiff)
{
    int flags=0;
    double wmomlast=0, wmom=0;
    double sky     = IM_SKY(image);
    double counts  = IM_COUNTS(image);
    size_t npoints = IM_SIZE(image);

    struct iter *iter_struct = iter_new(gvec->size);

    iter_struct->nsky = sky/counts;
    iter_struct->psky = sky/(counts/npoints);

    if (gvec_psf)
        gvec_set_total_moms(gvec_psf);

    wmomlast=-9999;
    *iter=0;
    while (*iter < self->maxiter) {
        if (self->verbose > 1) gvec_print(gvec,stderr);

        flags = gmix_get_sums(self, image, gvec, gvec_psf, iter_struct);
        if (flags!=0)
            goto _gmix_image_bail;

        gmix_set_gvec_fromiter(gvec, gvec_psf, iter_struct);

        // fixing sky doesn't work, need correct starting value
        if (!self->fixsky) {
            iter_struct->psky = iter_struct->skysum;
            iter_struct->nsky = iter_struct->psky/npoints;
        }

        wmom = gvec_wmomsum(gvec);
        wmom /= iter_struct->psum;
        *fdiff = fabs((wmom-wmomlast)/wmom);

        if (*fdiff < self->tol) {
            break;
        }
        wmomlast = wmom;
        (*iter)++;
    }

_gmix_image_bail:
    if (self->maxiter == (*iter)) {
        flags += GMIX_ERROR_MAXIT;
    }
    if (flags!=0 && self->verbose) wlog("error found at iter %lu\n", *iter);

    iter_struct = iter_free(iter_struct);

    return flags;
}

static void set_means(struct gvec *gvec, struct vec2 *cen)
{
    size_t i=0;
    for (i=0; i<gvec->size; i++) {
        gvec->data[i].row = cen->v1;
        gvec->data[i].col = cen->v2;
    }
}

/*
 * this could be cleaned up, some repeated code
 */
int gmix_image_samecen(struct gmix* self,
                       struct image *image, 
                       struct gvec *gvec,
                       struct gvec *gvec_psf,
                       size_t *iter,
                       double *fdiff)
{
    int flags=0;
    double wmomlast=0, wmom=0;
    double sky     = IM_SKY(image);
    double counts  = IM_COUNTS(image);
    size_t npoints = IM_SIZE(image);
    struct vec2 cen_new;
    struct gvec *gcopy=NULL;
    struct iter *iter_struct = iter_new(gvec->size);

    gcopy = gvec_new(gvec->size);

    iter_struct->nsky = sky/counts;
    iter_struct->psky = sky/(counts/npoints);

    if (gvec_psf)
        gvec_set_total_moms(gvec_psf);

    wmomlast=-9999;
    *iter=0;
    while (*iter < self->maxiter) {
        if (self->verbose > 1) gvec_print(gvec,stderr);

        // first pass to get centers
        flags = gmix_get_sums(self, image, gvec, gvec_psf, iter_struct);
        if (flags!=0)
            goto _gmix_image_samecen_bail;

        // copy for getting centers only
        gvec_copy(gvec, gcopy);
        gmix_set_gvec_fromiter(gcopy, gvec_psf, iter_struct);

        if (!gvec_wmean_center(gcopy, &cen_new)) {
            flags += GMIX_ERROR_NEGATIVE_DET_SAMECEN;
            goto _gmix_image_samecen_bail;
        }
        set_means(gvec, &cen_new);

        // now that we have fixed centers, we re-calculate everything
        flags = gmix_get_sums(self, image, gvec, gvec_psf, iter_struct);
        if (flags!=0)
            goto _gmix_image_samecen_bail;
 

        gmix_set_gvec_fromiter(gvec, gvec_psf, iter_struct);
        // we only wanted to update the moments, set these back.
        // Should do with extra par in above function or something
        set_means(gvec, &cen_new);

        // fixing sky doesn't work, need correct starting value
        if (!self->fixsky) {
            iter_struct->psky = iter_struct->skysum;
            iter_struct->nsky = iter_struct->psky/npoints;
        }

        wmom = gvec_wmomsum(gvec);
        wmom /= iter_struct->psum;
        *fdiff = fabs((wmom-wmomlast)/wmom);

        if (*fdiff < self->tol) {
            break;
        }
        wmomlast = wmom;
        (*iter)++;
    }

_gmix_image_samecen_bail:
    if (self->maxiter == (*iter)) {
        flags += GMIX_ERROR_MAXIT;
    }
    if (flags!=0 && self->verbose) wlog("error found at iter %lu\n", *iter);

    gcopy = gvec_free(gcopy);
    iter_struct = iter_free(iter_struct);

    return flags;
}

// set all the covariances equal to the input covariance
// scaled to their own size
//   cov_i = cov*(irr_i+icc_i)/(irr+icc)
static void force_coellip(struct gvec *gvec, struct mtx2 *cov)
{
    double size = cov->m11+cov->m22;
    double size_i=0.0;
    struct gauss *gauss=gvec->data;
    struct gauss *end=gvec->data+gvec->size;

    for (; gauss != end; gauss++) {
        size_i = gauss->irr + gauss->icc;

        gauss->irr = cov->m11*size_i/size;
        gauss->irc = cov->m12*size_i/size;
        gauss->icc = cov->m22*size_i/size;

        gauss->det = gauss->irr*gauss->icc - gauss->irc*gauss->irc;
    }
}
int gmix_image_coellip(struct gmix* self,
                       struct image *image, 
                       struct gvec *gvec,
                       struct gvec *gvec_psf,
                       size_t *iter,
                       double *fdiff)
{
    int flags=0;
    double wmomlast=0, wmom=0;
    double sky     = IM_SKY(image);
    double counts  = IM_COUNTS(image);
    size_t npoints = IM_SIZE(image);
    struct vec2 cen_new={0};
    struct mtx2 cov={0};
    struct gvec *gcopy=NULL;
    struct iter *iter_struct = iter_new(gvec->size);

    gcopy = gvec_new(gvec->size);

    iter_struct->nsky = sky/counts;
    iter_struct->psky = sky/(counts/npoints);

    if (gvec_psf)
        gvec_set_total_moms(gvec_psf);

    wmomlast=-9999;
    *iter=0;
    while (*iter < self->maxiter) {
        if (self->verbose > 1) gvec_print(gvec,stderr);

        // first pass to get centers
        flags = gmix_get_sums(self, image, gvec, gvec_psf, iter_struct);
        if (flags!=0)
            goto _gmix_image_coellip_bail;

        // copy for getting centers only
        gvec_copy(gvec, gcopy);
        gmix_set_gvec_fromiter(gcopy, gvec_psf, iter_struct);

        if (!gvec_wmean_center(gcopy, &cen_new)) {
            flags += GMIX_ERROR_NEGATIVE_DET_SAMECEN;
            goto _gmix_image_coellip_bail;
        }
        set_means(gvec, &cen_new);

        // now that we have fixed centers, we re-calculate everything
        flags = gmix_get_sums(self, image, gvec, gvec_psf, iter_struct);
        if (flags!=0)
            goto _gmix_image_coellip_bail;
 

        gmix_set_gvec_fromiter(gvec, gvec_psf, iter_struct);
        // we only wanted to update the moments, set these back.
        // Should do with extra par in above function or something
        set_means(gvec, &cen_new);

        // now force the covariance matrices to be proportional
        gvec_wmean_covar(gvec, &cov);
        force_coellip(gvec, &cov);

        // fixing sky doesn't work, need correct starting value
        if (!self->fixsky) {
            iter_struct->psky = iter_struct->skysum;
            iter_struct->nsky = iter_struct->psky/npoints;
        }

        wmom = gvec_wmomsum(gvec);
        wmom /= iter_struct->psum;
        *fdiff = fabs((wmom-wmomlast)/wmom);

        if (*fdiff < self->tol) {
            break;
        }
        wmomlast = wmom;
        (*iter)++;
    }

_gmix_image_coellip_bail:
    if (self->maxiter == (*iter)) {
        flags += GMIX_ERROR_MAXIT;
    }
    if (flags!=0 && self->verbose) wlog("error found at iter %lu\n", *iter);

    gcopy = gvec_free(gcopy);
    iter_struct = iter_free(iter_struct);

    return flags;
}



int gmix_get_sums(struct gmix* self,
                  struct image *image,
                  struct gvec *gvec,
                  struct gvec *gvec_psf,
                  struct iter* iter)
{
    int flags=0;
    double igrat=0, imnorm=0, gtot=0, wtau=0, b=0, chi2=0;
    double u=0, v=0, uv=0, u2=0, v2=0;
    size_t i=0, col=0, row=0;
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);
    size_t row0=IM_ROW0(image), col0=IM_COL0(image); // could be a subimage
    struct gauss* gauss=NULL;
    struct sums *sums=NULL;

    iter_clear(iter);
    for (row=0; row<nrows; row++) {
        for (col=0; col<ncols; col++) {

            imnorm = IM_GET(image, row, col);
            imnorm /= IM_COUNTS(image);

            gtot=0;
            gauss = &gvec->data[0];
            sums = &iter->sums[0];
            for (i=0; i<gvec->size; i++) {
                if (gauss->det <= 0) {
                    if (self->verbose) wlog("found det: %.16g\n", gauss->det);
                    flags+=GMIX_ERROR_NEGATIVE_DET;
                    goto _gmix_get_sums_bail;
                }
                // w.r.t. row0,col0 in case this is a masked image
                // centers will be w.r.t. the main image
                u = (row-(gauss->row-row0));
                v = (col-(gauss->col-col0));

                u2 = u*u; v2 = v*v; uv = u*v;

                if (gvec_psf) { 
                    sums->gi = gmix_evaluate_convolved(self,
                                                       gauss,
                                                       gvec_psf,
                                                       u2,uv,v2,
                                                       &flags);
                    if (flags != 0) {
                        goto _gmix_get_sums_bail;
                    }
                } else {
                    chi2=gauss->icc*u2 + gauss->irr*v2 - 2.0*gauss->irc*uv;
                    chi2 /= gauss->det;
                    b = M_TWO_PI*sqrt(gauss->det);
                    sums->gi = gauss->p*exp( -0.5*chi2 )/b;
                }
                gtot += sums->gi;

                // keep row units in unmasked frame
                sums->trowsum = (row0+row)*sums->gi;
                sums->tcolsum = (col0+col)*sums->gi;
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
                // wtau is gi[pix]/gtot[pix]*imnorm[pix]
                // which is Dave's tau*imnorm = wtau
                wtau = sums->gi*igrat;  
                //wtau = sums->gi*imnorm/gtot;  

                iter->psum += wtau;
                sums->pnew += wtau;

                // row*gi/gtot*imnorm
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


/* 
   mathematically this comes down to averaging over
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


double gmix_evaluate_convolved(struct gmix* self,
                               struct gauss *gauss,
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
        if (self->verbose) wlog("found obj det: %.16g\n", det);
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
            if (self->verbose) wlog("found convolved det: %.16g\n", det);
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




void gmix_set_gvec_fromiter(struct gvec *gvec, 
                            struct gvec *gvec_psf, 
                            struct iter* iter)
{
    if (gvec_psf) {
        gmix_set_gvec_fromiter_convolved(gvec, gvec_psf, iter);
    } else {
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
}

void gmix_set_gvec_fromiter_convolved(struct gvec *gvec, 
                                      struct gvec *gvec_psf,
                                      struct iter* iter)
{
    struct sums *sums=NULL;
    struct gauss *gauss=NULL;
    size_t i=0;

    sums=iter->sums;
    gauss=gvec->data;
    for (i=0; i<gvec->size; i++) {
        gauss->p   = sums->pnew;
        gauss->row = sums->rowsum/sums->pnew;
        gauss->col = sums->colsum/sums->pnew;

        gauss->irr = sums->u2sum/sums->pnew - gvec_psf->total_irr;
        gauss->irc = sums->uvsum/sums->pnew - gvec_psf->total_irc;
        gauss->icc = sums->v2sum/sums->pnew - gvec_psf->total_icc;

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
    self->skysum=0;
    self->psum=0;
    memset(self->sums,0,self->ngauss*sizeof(struct sums));
}
