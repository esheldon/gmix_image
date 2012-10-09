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

/* convolution results in an nobj*npsf total gaussians */
struct gvec *gvec_convolve(struct gvec *obj_gvec, 
                           struct gvec *psf_gvec)
{
    struct gauss *psf=NULL, *obj=NULL, *comb=NULL;

    int ntot=0, iobj=0, ipsf=0;
    double psum=0;

    ntot = obj_gvec->size*psf_gvec->size;
    struct gvec *gvec = gvec_new(ntot);

    for (ipsf=0; ipsf<psf_gvec->size; ipsf++) {
        psf = &psf_gvec->data[ipsf];
        psum += psf->p;
    }

    obj = obj_gvec->data;
    comb = gvec->data;
    for (iobj=0; iobj<obj_gvec->size; iobj++) {

        psf = psf_gvec->data;
        for (ipsf=0; ipsf<psf_gvec->size; ipsf++) {

            comb->row = obj->row;
            comb->col = obj->col;

            comb->irr = obj->irr + psf->irr;
            comb->irc = obj->irc + psf->irc;
            comb->icc = obj->icc + psf->icc;

            comb->det = comb->irr*comb->icc - comb->irc*comb->irc;

            comb->p = obj->p*psf->p/psum;

            psf++;
            comb++;
        }

        obj++;
    }

    return gvec;
}


// pars are full gmix of size 6*ngauss
struct gvec *pars_to_gvec(double *pars, int sz)
{
    int ngauss=0;
    struct gauss *gauss=NULL;

    int i=0, beg=0;

    //pars = PyArray_DATA(array);
    //sz = PyArray_SIZE(array);

    ngauss = sz/6;

    struct gvec *gvec = gvec_new(ngauss);


    for (i=0; i<ngauss; i++) {
        beg = i*6;

        gauss = &gvec->data[i];

        gauss->p   = pars[beg+0];
        gauss->row = pars[beg+1];
        gauss->col = pars[beg+2];
        gauss->irr = pars[beg+3];
        gauss->irc = pars[beg+4];
        gauss->icc = pars[beg+5];
    }

    gvec_set_dets(gvec);
    return gvec;
}



struct gvec *coellip_pars_to_gvec(double *pars, int sz)
{
    int ngauss=0;
    double row=0, col=0, e1=0, e2=0, Tmax=0, Ti=0, pi=0, Tfrac=0;
    struct gauss *gauss=NULL;

    int i=0;

    //pars = PyArray_DATA(array);
    //sz = PyArray_SIZE(array);

    ngauss = (sz-4)/2;

    struct gvec * gvec = gvec_new(ngauss);

    row=pars[0];
    col=pars[1];
    e1 = pars[2];
    e2 = pars[3];
    Tmax = pars[4];

    for (i=0; i<ngauss; i++) {
        gauss = &gvec->data[i];

        if (i==0) {
            Ti = Tmax;
        } else {
            Tfrac = pars[4+i];
            Ti = Tmax*Tfrac;
        }

        pi = pars[4+ngauss+i];

        gauss->p = pi;
        gauss->row = row;
        gauss->col = col;

        gauss->irr = (Ti/2.)*(1-e1);
        gauss->irc = (Ti/2.)*e2;
        gauss->icc = (Ti/2.)*(1+e1);
    }

    gvec_set_dets(gvec);
    return gvec;
}

/* 
   Generate a gvec from the inputs pars assuming an appoximate
   3-gaussian representation of an exponential disk or devauc.

   One component is nearly a delta function

   pars should be [row,col,e1,e2,T,p]

   T = sum(pi*Ti)/sum(pi)

   The p and F values are chosen to make this so
*/

/* helper function */
static struct gvec *_gapprox_pars_to_gvec(double *pars, 
                                          const double *Fvals, 
                                          const double *pvals)
{
    double row=0, col=0, e1=0, e2=0;
    double T=0, Tvals[3]={0};
    double p=0, counts[3]={0};

    struct gauss *gauss=NULL;
    struct gvec * gvec = NULL;

    int i=0;

    row=pars[0];
    col=pars[1];
    e1=pars[2];
    e2=pars[3];
    T=pars[4];
    p=pars[5];

    Tvals[0] = Fvals[0]*T;
    Tvals[1] = Fvals[1]*T;
    Tvals[2] = Fvals[2]*T;
    counts[0] = pvals[0]*p;
    counts[1] = pvals[1]*p;
    counts[2] = pvals[2]*p;

    gvec = gvec_new(3);

    gauss=gvec->data;
    for (i=0; i<gvec->size; i++) {
        gauss->p = counts[i];
        gauss->row = row;
        gauss->col = col;
        gauss->irr = (Tvals[i]/2.)*(1-e1);
        gauss->irc = (Tvals[i]/2.)*e2;
        gauss->icc = (Tvals[i]/2.)*(1+e1);

        gauss->det = gauss->irr*gauss->icc - gauss->irc*gauss->irc;
        gauss++;
    }

    return gvec;
}
struct gvec *gapprox_pars_to_gvec(double *pars, enum gapprox type)
{
    /* pvals are normalized */
    static const double exp_Fvals[3] = 
        {3.947146384343532e-05, 0.5010756804049256, 1.911515572152285};
    static const double exp_pvals[3] = 
        {0.06031348356539361,   0.5645244398053312, 0.3751620766292753};

    /* seems to me more a function of size than for exp */
    static const double dev_Fvals[3] = 
        {0.003718633817323675, 0.9268795541243965, 9.627400726500005};
    static const double dev_pvals[3] = 
        {0.659318547053916,    0.2623209100496331, 0.07836054289645095};

    if (type==GAPPROX_EXP) {
        return _gapprox_pars_to_gvec(pars, exp_Fvals, exp_pvals);
    } else if (type==GAPPROX_DEV) {
        return _gapprox_pars_to_gvec(pars, dev_Fvals, dev_pvals);
    } else {
        return NULL;
    }
}


