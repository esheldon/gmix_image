#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "gvec.h"
#include "defs.h"
#include "matrix.h"

struct gvec* gvec_new(size_t ngauss)
{
    struct gvec*self=NULL;
    if (ngauss == 0) {
        wlog("number of gaussians must be > 0\n");
        return NULL;
    }

    self = calloc(1, sizeof(struct gvec));
    if (self==NULL) {
        wlog("could not allocate struct gvec\n");
        exit(EXIT_FAILURE);
    }

    self->size=ngauss;

    self->data = calloc(self->size, sizeof(struct gauss));
    if (self->data==NULL) {
        wlog("could not allocate %lu gaussian structs\n",ngauss);
        free(self);
        exit(EXIT_FAILURE);
    }

    return self;

}

struct gvec *gvec_free(struct gvec *self)
{
    if (self) {
        free(self->data);
        self->data=NULL;
        free(self);
        self=NULL;
    }
    return self;
}

void gauss_set(struct gauss* self, 
               double p, 
               double row, 
               double col,
               double irr,
               double irc,
               double icc)
{
    self->p   = p;
    self->row = row;
    self->col = col;
    self->irr = irr;
    self->irc = irc;
    self->icc = icc;
    self->det = irr*icc - irc*irc;
}
void gvec_set_dets(struct gvec *self)
{
    struct gauss *gptr = self->data;
    size_t i=0;
    for (i=0; i<self->size; i++) {
        gptr->det = gptr->irr*gptr->icc - gptr->irc*gptr->irc;
        gptr++;
    }
}
int gvec_copy(struct gvec *self, struct gvec* dest)
{
    if (dest->size != self->size) {
        wlog("gvec are not same size\n");
        return 0;
    }
    memcpy(dest->data, self->data, self->size*sizeof(struct gauss));
    return 1;
}

void gvec_print(struct gvec *self, FILE* fptr)
{
    struct gauss *gptr = self->data;
    size_t i=0;
    for (i=0; i<self->size; i++) {
        fprintf(fptr,
             "%lu p: %9.6lf row: %9.6lf col: %9.6lf " 
             "irr: %9.6lf irc: %9.6lf icc: %9.6lf\n",
             i, gptr->p, gptr->row, gptr->col,
             gptr->irr,gptr->irc, gptr->icc);
        gptr++;
    }
}

double gvec_wmomsum(struct gvec* gvec)
{
    double wmom=0;
    struct gauss* gauss=gvec->data;
    size_t i=0;
    for (i=0; i<gvec->size; i++) {
        wmom += gauss->p*(gauss->irr + gauss->icc);
        gauss++;
    }
    return wmom;
}

void gvec_set_total_moms(struct gvec *self)
{
    size_t i=0;
    double p=0, psum=0;
    struct gauss *gauss=NULL;

    self->total_irr=0;
    self->total_irc=0;
    self->total_icc=0;

    gauss = self->data;
    for (i=0; i<self->size; i++) {
        p = gauss->p;
        psum += p;

        self->total_irr += p*gauss->irr;
        self->total_irc += p*gauss->irc;
        self->total_icc += p*gauss->icc;
        gauss++;
    }

    self->total_irr /= psum;
    self->total_irc /= psum;
    self->total_icc /= psum;
}


/*
 
 Find the weighted average center

 for j gaussians

     munew = sum_j( C_j^-1 p_j mu_j )/sum( C_j^-1 p_j )

 where the mus are mean vectors and the C are the covarance
 matrices.

 The following would be a lot simpler if we use vec2 and mtx2
 types in the gaussian!  Maybe some other day.

 */
int gvec_wmean_center(const struct gvec* gvec, struct vec2* mu_new)
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
            goto _gvec_wmean_center_bail;
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
        goto _gvec_wmean_center_bail;
    }

    mtx2_vec2prod(&Cinvpsum_inv, &mu_Cinvpsum, mu_new);

_gvec_wmean_center_bail:
    return status;
}

/*
 * calculate the mean covariance matrix
 *
 *   sum(p*Covar)/sum(p)
 */
void gvec_wmean_covar(const struct gvec* gvec, struct mtx2 *cov)
{
    double psum=0.0;
    struct gauss *gauss=gvec->data;
    struct gauss *end=gvec->data+gvec->size;

    mtx2_sprodi(cov, 0.0);
    
    for (; gauss != end; gauss++) {
        psum += gauss->p;
        cov->m11 += gauss->p*gauss->irr;
        cov->m12 += gauss->p*gauss->irc;
        cov->m22 += gauss->p*gauss->icc;
    }

    cov->m11 /= psum;
    cov->m12 /= psum;
    cov->m22 /= psum;
}
