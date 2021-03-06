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

    self->drr = self->irr/self->det;
    self->drc = self->irc/self->det;
    self->dcc = self->icc/self->det;
    self->norm = 1./(M_TWO_PI*sqrt(self->det));

}
void gvec_set_dets(struct gvec *self)
{
    struct gauss *gauss = NULL;
    size_t i=0;
    for (i=0; i<self->size; i++) {
        gauss = &self->data[i];
        gauss->det = gauss->irr*gauss->icc - gauss->irc*gauss->irc;
    }
}

int gvec_verify(const struct gvec *self)
{
    size_t i=0;
    const struct gauss *gauss=NULL;

    if (!self || !self->data) {
        DBG wlog("gvec is not initialized\n");
        return 0;
    }

    for (i=0; i<self->size; i++) {
        gauss=&self->data[i];
        if (gauss->det <= 0) {
            DBG wlog("gvec_verify found det: %.16g\n", gauss->det);
            return 0;
        }
    }
    return 1;
}


int gvec_copy(const struct gvec *self, struct gvec* dest)
{
    if (dest->size != self->size) {
        wlog("gvec are not same size\n");
        return 0;
    }
    memcpy(dest->data, self->data, self->size*sizeof(struct gauss));
    return 1;
}

void gvec_print(const struct gvec *self, FILE* fptr)
{
    const struct gauss *gauss = NULL;
    size_t i=0;
    for (i=0; i<self->size; i++) {
        gauss = &self->data[i];
        fprintf(fptr,
             "%lu p: %9.6lf row: %9.6lf col: %9.6lf " 
             "irr: %9.6lf irc: %9.6lf icc: %9.6lf\n",
             i, 
             gauss->p, gauss->row, gauss->col,
             gauss->irr,gauss->irc, gauss->icc);
    }
}

double gvec_wmomsum(const struct gvec* self)
{
    double wmom=0;
    const struct gauss* gauss=NULL;
    size_t i=0;
    for (i=0; i<self->size; i++) {
        gauss=&self->data[i];
        wmom += gauss->p*(gauss->irr + gauss->icc);
    }
    return wmom;
}

void gvec_get_cen(const struct gvec *self, double *row, double *col)
{
    int i=0;
    const struct gauss *gauss=NULL;
    double psum=0;
    (*row)=0;
    (*col)=0;

    for (i=0; i<self->size; i++) {
        gauss=&self->data[i];

        psum += gauss->p;
        (*row) += gauss->p*gauss->row;
        (*col) += gauss->p*gauss->col;
    }

    (*row) /= psum;
    (*col) /= psum;
}

void gvec_set_cen(struct gvec *self, double row, double col)
{
    int i=0;
    struct gauss *gauss=NULL;

    double row_cur=0, col_cur=0;
    double row_shift=0, col_shift=0;

    gvec_get_cen(self, &row_cur, &col_cur);

    row_shift = row - row_cur;
    col_shift = col - col_cur;

    for (i=0; i<self->size; i++) {
        gauss=&self->data[i];

        gauss->row += row_shift;
        gauss->col += col_shift;
    }
}


double gvec_get_T(const struct gvec *self)
{
    int i=0;
    const struct gauss *gauss=NULL;
    double T=0, psum=0;

    for (i=0; i<self->size; i++) {
        gauss=&self->data[i];

        T += (gauss->irr + gauss->icc)*gauss->p;
        psum += gauss->p;
    }
    T /= psum;
    return T;
}
double gvec_get_psum(const struct gvec *self)
{
    int i=0;
    const struct gauss *gauss=NULL;
    double psum=0;

    for (i=0; i<self->size; i++) {
        gauss=&self->data[i];

        psum += gauss->p;
    }
    return psum;
}
void gvec_set_psum(struct gvec *self, double psum)
{
    int i=0;
    double psum_cur=0, rat=0;
    struct gauss *gauss=NULL;

    psum_cur=gvec_get_psum(self);
    rat=psum/psum_cur;

    for (i=0; i<self->size; i++) {
        gauss=&self->data[i];

        gauss->p *= rat;
    }
}

void gvec_set_total_moms(struct gvec *self)
{
    size_t i=0;
    double p=0, psum=0;
    struct gauss *gauss=NULL;

    self->total_irr=0;
    self->total_irc=0;
    self->total_icc=0;

    for (i=0; i<self->size; i++) {
        gauss = &self->data[i];
        p = gauss->p;
        psum += p;

        self->total_irr += p*gauss->irr;
        self->total_irc += p*gauss->irc;
        self->total_icc += p*gauss->icc;
    }

    self->total_irr /= psum;
    self->total_irc /= psum;
    self->total_icc /= psum;
}


/* convolution results in nobj*npsf total gaussians */
/*
struct gvec *gvec_convolve_wrong(struct gvec *obj_gvec, 
                           struct gvec *psf_gvec)
{
    struct gauss *psf=NULL, *obj=NULL, *comb=NULL;
    struct gvec *gvec=NULL;

    int ntot=0, iobj=0, ipsf=0;
    double irr=0, irc=0, icc=0, psum=0;

    ntot = obj_gvec->size*psf_gvec->size;

    gvec = gvec_new(ntot);

    for (ipsf=0; ipsf<psf_gvec->size; ipsf++) {
        psf = &psf_gvec->data[ipsf];
        psum += psf->p;
    }

    comb = gvec->data;
    for (iobj=0; iobj<obj_gvec->size; iobj++) {
        obj = &obj_gvec->data[iobj];

        for (ipsf=0; ipsf<psf_gvec->size; ipsf++) {
            psf = &psf_gvec->data[ipsf];

            irr = obj->irr + psf->irr;
            irc = obj->irc + psf->irc;
            icc = obj->icc + psf->icc;

            gauss_set(comb,
                      obj->p*psf->p/psum,
                      obj->row, obj->col, 
                      irr, irc, icc);

            comb++;
        }
    }

    return gvec;
}
*/
struct gvec *gvec_convolve(struct gvec *obj_gvec, 
                           struct gvec *psf_gvec)
{
    struct gauss *psf=NULL, *obj=NULL, *comb=NULL;
    struct gvec *gvec=NULL;

    int ntot=0, iobj=0, ipsf=0;
    double irr=0, irc=0, icc=0, psum=0;
    double row=0, col=0;
    double psf_rowcen=0, psf_colcen=0;

    ntot = obj_gvec->size*psf_gvec->size;

    gvec = gvec_new(ntot);

    gvec_get_cen(psf_gvec, &psf_rowcen, &psf_colcen);

    for (ipsf=0; ipsf<psf_gvec->size; ipsf++) {
        psf = &psf_gvec->data[ipsf];
        psum += psf->p;
    }

    comb = gvec->data;
    for (iobj=0; iobj<obj_gvec->size; iobj++) {
        obj = &obj_gvec->data[iobj];

        for (ipsf=0; ipsf<psf_gvec->size; ipsf++) {
            psf = &psf_gvec->data[ipsf];

            irr = obj->irr + psf->irr;
            irc = obj->irc + psf->irc;
            icc = obj->icc + psf->icc;

            // off-center psf components shift the
            // convolved center
            row = obj->row + (psf->row-psf_rowcen);
            col = obj->col + (psf->col-psf_colcen);

            gauss_set(comb,
                      obj->p*psf->p/psum,
                      row, col, 
                      irr, irc, icc);

            comb++;
        }
    }

    return gvec;
}



// pars are full gmix of size 6*ngauss
struct gvec *gvec_from_pars(double *pars, int size)
{
    int ngauss=0;
    struct gauss *gauss=NULL;
    struct gvec *gvec = NULL;

    int i=0, beg=0;

    if ( (size % 6) != 0) {
        return NULL;
    }
    ngauss = size/6;

    gvec = gvec_new(ngauss);

    for (i=0; i<ngauss; i++) {
        gauss = &gvec->data[i];

        beg = i*6;

        gauss_set(gauss,
                  pars[beg+0],  // p
                  pars[beg+1],  // row
                  pars[beg+2],  // col
                  pars[beg+3],  // irr
                  pars[beg+4],  // irc
                  pars[beg+5]); // icc
    }

    return gvec;
}

// out of range inputs are scaled to be in range
static int g1g2_to_e1e2(double g1, double g2, double *e1, double *e2)
{
    double g=0,e=0,fac=0;

    g = sqrt(g1*g1 + g2*g2);

    if (g == 0) {
        *e1=0;
        *e2=0;
    } else {

        if (g >= 1.) {
            *e1=-9999;
            *e2=-9999;
            return 0;
        }

        e = tanh(2*atanh(g));
        if (e >= 1.) {
            // round off error most likely
            e = 0.99999999;
        }
        fac = e/g;

        *e1 = fac*g1;
        *e2 = fac*g2;
    }
    return 1;
}


struct gvec *gvec_from_coellip_Tfrac(double *pars, int size)
{
    int ngauss=0;
    double row=0, col=0, g1=0, g2=0, e1=0, e2=0;
    double Tmax=0, Ti=0, pi=0, Tfrac=0;
    struct gauss *gauss=NULL;

    int i=0;

    if ( ((size-4) % 2) != 0) {
        return NULL;
    }
    ngauss = (size-4)/2;


    row=pars[0];
    col=pars[1];
    g1 = pars[2];
    g2 = pars[3];
    Tmax = pars[4];

    if (!g1g2_to_e1e2(g1,g2,&e1,&e2)) {
        return NULL;
    }
    struct gvec * gvec = gvec_new(ngauss);

    for (i=0; i<ngauss; i++) {
        gauss = &gvec->data[i];

        if (i==0) {
            Ti = Tmax;
        } else {
            Tfrac = pars[4+i];
            Ti = Tmax*Tfrac;
        }

        pi = pars[4+ngauss+i];

        gauss_set(gauss,
                  pi,
                  row, 
                  col, 
                  (Ti/2.)*(1-e1),
                  (Ti/2.)*e2,
                  (Ti/2.)*(1+e1));
    }

    return gvec;
}


struct gvec *gvec_from_coellip(double *pars, int size)
{
    int ngauss=0, Tstart=0, Astart=0;
    double row=0, col=0, g1=0, g2=0, e1=0, e2=0, Ti=0, Ai=0;
    struct gauss *gauss=NULL;

    int i=0;

    if ( ((size-4) % 2) != 0) {
        return NULL;
    }
    ngauss = (size-4)/2;


    row=pars[0];
    col=pars[1];
    g1 = pars[2];
    g2 = pars[3];

    Tstart=4;
    Astart=Tstart+ngauss;

    if (!g1g2_to_e1e2(g1,g2,&e1,&e2)) {
        return NULL;
    }

    struct gvec * gvec = gvec_new(ngauss);

    for (i=0; i<ngauss; i++) {
        gauss = &gvec->data[i];

        Ti = pars[Tstart+i];
        Ai = pars[Astart+i];

        gauss_set(gauss,
                  Ai,
                  row, 
                  col, 
                  (Ti/2.)*(1-e1),
                  (Ti/2.)*e2,
                  (Ti/2.)*(1+e1));
    }

    return gvec;
}

static struct gvec *_gapprox_pars_to_gvec(double *pars, 
                                          const double *Fvals, 
                                          const double *pvals,
                                          int ngauss)
{
    double row=0, col=0, g1=0, g2=0, e1=0, e2=0;
    double T=0, T_i=0;
    double counts=0, counts_i=0;

    struct gauss *gauss=NULL;
    struct gvec  *gvec=NULL;

    int i=0;

    row=pars[0];
    col=pars[1];
    g1=pars[2];
    g2=pars[3];
    T=pars[4];
    counts=pars[5];

    if (!g1g2_to_e1e2(g1,g2,&e1,&e2)) {
        return NULL;
    }

    gvec = gvec_new(ngauss);

    for (i=0; i<gvec->size; i++) {
        gauss=&gvec->data[i];

        T_i = T*Fvals[i];
        counts_i=counts*pvals[i];

        gauss_set(gauss,
                  counts_i,
                  row, col, 
                  (T_i/2.)*(1-e1), 
                  (T_i/2.)*e2,
                  (T_i/2.)*(1+e1));
    }

    return gvec;
}

struct gvec *gvec_from_pars_dev6(double *pars, int size)
{
    if (size != 6) {
        fprintf(stderr,"wrong par len for dev6: %d\n", size);
        return NULL;
    }

    // from Hogg & Lang, normalized
    static const double Fvals[6] = 
        {8.094092042722281e-06, 
         0.0001690696202011497, 
         0.0019014383023117968, 
         0.017212145881988744, 
         0.15359226155140734, 
         1.7878113597233429};
    static const double pvals[6] = 
        {0.00070880632246569225, 
         0.0067331181625659223, 
         0.034438341436557503, 
         0.12060545499079853, 
         0.30562612308952852, 
         0.53188815599808381};

    return _gapprox_pars_to_gvec(pars, Fvals, pvals, 6);
}

struct gvec *gvec_from_pars_dev10(double *pars, int size)
{
    if (size != 6) {
        fprintf(stderr,"wrong par len for dev10: %d\n", size);
        return NULL;
    }

    // from Hogg & Lang, normalized
    static const double Fvals[10] = 
        {2.9934935706271918e-07, 
         3.4651596338231207e-06, 
         2.4807910570562753e-05, 
         0.00014307404300535354, 
         0.000727531692982395, 
         0.003458246439442726, 
         0.0160866454407191, 
         0.077006776775654429, 
         0.41012562102501476, 
         2.9812509778548648};
    static const double pvals[10] = 
        {6.5288960012625658e-05, 
         0.00044199216814302695, 
         0.0020859587871659754, 
         0.0075913681418996841, 
         0.02260266219257237, 
         0.056532254390212859, 
         0.11939049233042602, 
         0.20969545753234975, 
         0.29254151133139222, 
         0.28905301416582552};

    return _gapprox_pars_to_gvec(pars, Fvals, pvals, 10);
}

struct gvec *gvec_from_pars_exp4(double *pars, int size)
{
    if (size != 6) {
        fprintf(stderr,"wrong par len for exp4: %d\n", size);
        return NULL;
    }

    // from Hogg & Lang, normalized
    static const double Fvals[4] = 
        {0.01474041913425168, 
         0.10842545172416658, 
         0.47550246740469826, 
         1.6605918409918259};
    static const double pvals[4] = 
        {0.008206472936682925, 
         0.095111781891460051, 
         0.4214499816612774, 
         0.47523176351057955};

    return _gapprox_pars_to_gvec(pars, Fvals, pvals, 4);
}

struct gvec *gvec_from_pars_exp6(double *pars, int size)
{
    if (size != 6) {
        fprintf(stderr,"wrong par len for exp6: %d\n", size);
        return NULL;
    }

    // from Hogg & Lang, normalized
    static const double Fvals[6] = 
        {0.002467115141477932, 
         0.018147435573256168, 
         0.07944063151366336, 
         0.27137669897479122, 
         0.79782256866993773, 
         2.1623306025075739};
    static const double pvals[6] = 
        {0.00061601229677880041, 
         0.0079461395724623237, 
         0.053280454055540001, 
         0.21797364640726541, 
         0.45496740582554868, 
         0.26521634184240478};

    return _gapprox_pars_to_gvec(pars, Fvals, pvals, 6);
}

struct gvec *gvec_from_pars_bd(double *pars, int size)
{
    int ngauss_exp=6, ngauss_dev=10;

    double pars_exp[6], pars_dev[6];
    struct gvec *gvec_exp=NULL, *gvec_dev=NULL, *gvec=NULL;

    if (size != 8) {
        fprintf(stderr,"wrong par len for bulge+disk: %d\n", size);
        return NULL;
    }

    pars_exp[0] = pars[0];
    pars_exp[1] = pars[1];
    pars_exp[2] = pars[2];
    pars_exp[3] = pars[3];
    pars_exp[4] = pars[4];
    pars_exp[5] = pars[6];

    pars_dev[0] = pars[0];
    pars_dev[1] = pars[1];
    pars_dev[2] = pars[2];
    pars_dev[3] = pars[3];
    pars_dev[4] = pars[5];
    pars_dev[5] = pars[7];

    gvec_exp=gvec_from_pars_exp6(pars_exp, 6);
    gvec_dev=gvec_from_pars_dev10(pars_dev, 6);

    gvec=gvec_new(16);
    memcpy(gvec->data,
           gvec_exp->data,
           ngauss_exp*sizeof(struct gauss));
    memcpy(gvec->data+ngauss_exp,
           gvec_dev->data,
           ngauss_dev*sizeof(struct gauss));

    gvec_exp=gvec_free(gvec_exp);
    gvec_dev=gvec_free(gvec_dev);

    return gvec;
}

struct gvec *gvec_from_pars_turb(double *pars, int size)
{
    if (size != 6) {
        fprintf(stderr,"wrong par len for turb: %d\n", size);
        return NULL;
    }

    static const double Fvals[3] = 
        {0.5793612389470884,1.621860687127999,7.019347162356363};
    static const double pvals[3] = 
        {0.596510042804182,0.4034898268889178,1.303069003078001e-07};

    return _gapprox_pars_to_gvec(pars, Fvals, pvals, 3);
}


