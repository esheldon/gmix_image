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
#include "gmix_em.h"
#include "image.h"
#include "gvec.h"
#include "matrix.h"
#include "defs.h"

#include "fmath.h"

struct gmix_em *gmix_em_new(size_t maxiter,
                            double tol,
                            int cocenter,
                            int verbose)
{
    struct gmix_em *self=NULL;

    self=calloc(1, sizeof(struct gmix_em));
    if (!self) {
        fprintf(stderr,"could not allocate struct gmix_em\n");
        exit(1);
    }

    self->maxiter=maxiter;
    self->tol=tol;
    self->cocenter=cocenter;
    self->verbose=verbose;

    // has_jacobian is not set, but make the jacobian default to something sane
    // anyway, just in case someone tries to use it
    self->jacob.dudrow=1;
    self->jacob.dvdcol=1;

    return self;
}

void gmix_em_add_jacobian(struct gmix_em *self,
                          double row0,
                          double col0,
                          double dudrow,
                          double dudcol,
                          double dvdrow,
                          double dvdcol)
{

    self->has_jacobian=1;
    jacobian_set(&self->jacob, row0, col0, dudrow, dudcol, dvdrow, dvdcol);
}


static double get_effective_scale(struct gmix_em* self)
{
    double scale=1.0;
    if (self->has_jacobian) {
        scale=sqrt(
                self->jacob.dudrow*self->jacob.dvdcol
                -
                self->jacob.dudcol*self->jacob.dvdrow);
    }
    return scale;
}
int gmix_em_run(struct gmix_em* self,
                const struct image *image, 
                struct gvec *gvec,
                size_t *iter,
                double *fdiff)
{
    int flags=0;
    double wmomlast=0, wmom=0;
    double scale=1;
    double area=0;

    double sky     = IM_SKY(image);
    double counts  = IM_COUNTS(image);
    size_t npoints = IM_SIZE(image);

    // we may not be working in pixel coordinates
    scale=get_effective_scale(self);
    area = npoints*scale*scale;

    struct iter *iter_struct = iter_new(gvec->size);

    iter_struct->nsky = sky/counts;
    iter_struct->psky = sky/(counts/area);

    wmomlast=-9999;
    *iter=0;
    while (*iter < self->maxiter) {
        if (self->verbose > 1) gvec_print(gvec,stderr);

        flags = gmix_get_sums(self, image, gvec, iter_struct);
        if (flags!=0)
            goto _gmix_em_bail;

        gmix_set_gvec_fromiter(gvec, iter_struct);

        iter_struct->psky = iter_struct->skysum;
        iter_struct->nsky = iter_struct->psky/area;

        wmom = gvec_wmomsum(gvec);
        wmom /= iter_struct->psum;
        *fdiff = fabs((wmom-wmomlast)/wmom);

        if (*fdiff < self->tol) {
            break;
        }
        wmomlast = wmom;
        (*iter)++;
    }

_gmix_em_bail:
    if (self->maxiter == (*iter)) {
        flags += GMIX_ERROR_MAXIT;
    }
    if (flags!=0 && self->verbose) wlog("error found at iter %lu\n", *iter);

    iter_struct = iter_free(iter_struct);

    return flags;
}
/*
 
 Find the center for the next step

 for j gaussians

     munew = sum_j( C_j^-1 p_j mu_j )/sum( C_j^-1 p_j )

 where the mus are mean vectors and the C are the covarance
 matrices.

 The following would be a lot simpler if we use vec2 and mtx2
 types in the gaussian!  Maybe some other day.

 */
static int get_cen_new(const struct gvec* gvec, struct vec2* mu_new)
{
    int status=1;
    struct vec2 mu_Cinvp, mu_Cinvpsum;
    struct mtx2 Cinvpsum, Cinvpsum_inv, C, Cinvp;
    size_t i=0;

    memset(&Cinvpsum,0,sizeof(struct mtx2));
    memset(&mu_Cinvpsum,0,sizeof(struct vec2));

    const struct gauss* gauss = gvec->data;
    for (i=0; i<gvec->size; i++) {
        // p*C^-1
        mtx2_set(&C, gauss->irr, gauss->irc, gauss->icc);
        if (!mtx2_invert(&C, &Cinvp)) {
            wlog("gvec_fix_centers: zero determinant found in C\n");
            status=0;
            goto _get_cen_new_bail;
        }
        mtx2_sprodi(&Cinvp, gauss->p);

        // running sum of p*C^-1
        mtx2_sumi(&Cinvpsum, &Cinvp);

        // set the center as a vec2
        vec2_set(&mu_Cinvp, gauss->row, gauss->col);
        // p*C^-1 * mu in place on mu
        mtx2_vec2prodi(&Cinvp, &mu_Cinvp);

        // running sum of p*C^-1 * mu
        vec2_sumi(&mu_Cinvpsum, &mu_Cinvp);
        gauss++;
    }

    if (!mtx2_invert(&Cinvpsum, &Cinvpsum_inv)) {
        wlog("gvec_fix_centers: zero determinant found in Cinvpsum\n");
        status=0;
        goto _get_cen_new_bail;
    }

    mtx2_vec2prod(&Cinvpsum_inv, &mu_Cinvpsum, mu_new);

_get_cen_new_bail:
    return status;
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
int gmix_em_cocenter_run(struct gmix_em* self,
                         const struct image *image, 
                         struct gvec *gvec,
                         size_t *iter,
                         double *fdiff)
{
    int flags=0;
    double wmomlast=0, wmom=0;
    double scale=1;
    double area=0;

    double sky     = IM_SKY(image);
    double counts  = IM_COUNTS(image);
    size_t npoints = IM_SIZE(image);

    struct vec2 cen_new;
    struct gvec *gcopy=NULL;
    struct iter *iter_struct = iter_new(gvec->size);

    // we may not be working in pixel coordinates
    scale=get_effective_scale(self);
    area = npoints*scale*scale;

    gcopy = gvec_new(gvec->size);

    iter_struct->nsky = sky/counts;
    iter_struct->psky = sky/(counts/area);


    wmomlast=-9999;
    *iter=0;
    while (*iter < self->maxiter) {
        if (self->verbose > 1) gvec_print(gvec,stderr);

        // first pass to get centers
        flags = gmix_get_sums(self, image, gvec, iter_struct);
        if (flags!=0)
            goto _gmix_em_cocenter_bail;

        // copy for getting centers only
        gvec_copy(gvec, gcopy);
        gmix_set_gvec_fromiter(gcopy, iter_struct);

        if (!get_cen_new(gcopy, &cen_new)) {
            flags += GMIX_ERROR_NEGATIVE_DET_COCENTER;
            goto _gmix_em_cocenter_bail;
        }
        set_means(gvec, &cen_new);

        // now that we have fixed centers, we re-calculate everything
        flags = gmix_get_sums(self, image, gvec, iter_struct);
        if (flags!=0)
            goto _gmix_em_cocenter_bail;
 

        gmix_set_gvec_fromiter(gvec, iter_struct);
        // we only wanted to update the moments, set these back.
        // Should do with extra par in above function or something
        set_means(gvec, &cen_new);

        iter_struct->psky = iter_struct->skysum;
        iter_struct->nsky = iter_struct->psky/area;

        wmom = gvec_wmomsum(gvec);
        wmom /= iter_struct->psum;
        *fdiff = fabs((wmom-wmomlast)/wmom);

        if (*fdiff < self->tol) {
            break;
        }
        wmomlast = wmom;
        (*iter)++;
    }

_gmix_em_cocenter_bail:
    if (self->maxiter == (*iter)) {
        flags += GMIX_ERROR_MAXIT;
    }
    if (flags!=0 && self->verbose) wlog("error found at iter %lu\n", *iter);

    gcopy = gvec_free(gcopy);
    iter_struct = iter_free(iter_struct);

    return flags;
}

int gmix_get_sums(struct gmix_em* self,
                  const struct image *image,
                  struct gvec *gvec,
                  struct iter* iter)
{
    int flags=0;
    if (self->has_jacobian) {
        flags=gmix_get_sums_jacobian(self, image, gvec, iter);
    } else {
        flags=gmix_get_sums_pix(self, image, gvec, iter);
    }
    return flags;
}

int gmix_get_sums_pix(struct gmix_em* self,
                      const struct image *image,
                      struct gvec *gvec,
                      struct iter* iter)
{
    int flags=0;
    double igrat=0, imnorm=0, gtot=0, wtau=0, chi2=0;
    double u=0, v=0, uv=0, u2=0, v2=0;
    size_t i=0, col=0, row=0;
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);
    const struct gauss* gauss=NULL;
    struct sums *sums=NULL;

    iter_clear(iter);
    for (row=0; row<nrows; row++) {
        for (col=0; col<ncols; col++) {

            imnorm = IM_GET(image, row, col);
            imnorm /= IM_COUNTS(image);

            gtot=0;
            for (i=0; i<gvec->size; i++) {
                gauss = &gvec->data[i];
                sums  = &iter->sums[i];

                if (gauss->det <= 0) {
                    if (self->verbose) wlog("found det: %.16g\n", gauss->det);
                    flags+=GMIX_ERROR_NEGATIVE_DET;
                    goto _gmix_get_sums_bail;
                }

                u = row-gauss->row;
                v = col-gauss->col;

                u2 = u*u; v2 = v*v; uv = u*v;

                chi2=gauss->dcc*u2 + gauss->drr*v2 - 2.0*gauss->drc*uv;

                if (chi2 < EXP_MAX_CHI2) {
                    sums->gi = gauss->norm*gauss->p*expd( -0.5*chi2 );
                } else {
                    sums->gi=0;
                }

                gtot += sums->gi;

                sums->trowsum = row*sums->gi;
                sums->tcolsum = col*sums->gi;
                sums->tu2sum  = u2*sums->gi;
                sums->tuvsum  = uv*sums->gi;
                sums->tv2sum  = v2*sums->gi;

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


int gmix_get_sums_jacobian(struct gmix_em* self,
                           const struct image *image,
                           struct gvec *gvec,
                           struct iter* iter)
{
    int flags=0;
    double igrat=0, imnorm=0, gtot=0, wtau=0, chi2=0;
    double u=0, v=0, uv=0, u2=0, v2=0;
    double uabs=0, vabs=0;
    size_t i=0, col=0, row=0;
    size_t nrows=IM_NROWS(image), ncols=IM_NCOLS(image);

    const struct gauss* gauss=NULL;
    const struct jacobian *jacob=&self->jacob;

    struct sums *sums=NULL;

    iter_clear(iter);
    for (row=0; row<nrows; row++) {
        uabs=JACOB_PIX2U(jacob, row, 0);
        vabs=JACOB_PIX2V(jacob, row, 0);
        for (col=0; col<ncols; col++) {

            imnorm = IM_GET(image, row, col);
            imnorm /= IM_COUNTS(image);

            gtot=0;
            for (i=0; i<gvec->size; i++) {
                gauss = &gvec->data[i];
                sums  = &iter->sums[i];

                if (gauss->det <= 0) {
                    if (self->verbose) wlog("found det: %.16g\n", gauss->det);
                    flags+=GMIX_ERROR_NEGATIVE_DET;
                    goto _gmix_get_sums_bail;
                }

                // "row" and "col" are really "u" and "v"
                u = uabs-gauss->row;
                v = vabs-gauss->col;

                u2 = u*u; v2 = v*v; uv = u*v;

                chi2=gauss->dcc*u2 + gauss->drr*v2 - 2.0*gauss->drc*uv;

                if (chi2 < EXP_MAX_CHI2) {
                    sums->gi = gauss->norm*gauss->p*expd( -0.5*chi2 );
                } else {
                    sums->gi=0;
                }

                gtot += sums->gi;

                sums->trowsum = uabs*sums->gi;
                sums->tcolsum = vabs*sums->gi;
                sums->tu2sum  = u2*sums->gi;
                sums->tuvsum  = uv*sums->gi;
                sums->tv2sum  = v2*sums->gi;

            }
            gtot += iter->nsky;
            igrat = imnorm/gtot;
            for (i=0; i<gvec->size; i++) {
                sums = &iter->sums[i];

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
            }
            iter->skysum += iter->nsky*imnorm/gtot;

            uabs += jacob->dudcol; vabs += jacob->dvdcol;

        } // cols
    } // rows

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



void gmix_set_gvec_fromiter(struct gvec *gvec, 
                            struct iter* iter)
{
    size_t i=0;
    struct sums *sums   = NULL;
    struct gauss *gauss = NULL;
    for (i=0; i<gvec->size; i++) {
        sums  = &iter->sums[i];
        gauss = &gvec->data[i];

        gauss_set(gauss,
                  sums->pnew,               // p
                  sums->rowsum/sums->pnew,  // row
                  sums->colsum/sums->pnew,  // col
                  sums->u2sum/sums->pnew,   // irr
                  sums->uvsum/sums->pnew,   // irc
                  sums->v2sum/sums->pnew);  // icc
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
