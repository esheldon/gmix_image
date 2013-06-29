#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "gmix.h"
#include "defs.h"
#include "matrix.h"
#include "convert.h"

long gmix_get_simple_ngauss(enum gmix_model model, long *flags)
{
    long ngauss=0;
    switch (model) {
        case GMIX_EXP:
            ngauss=6;
            break;
        case GMIX_DEV:
            ngauss=10;
            break;
        case GMIX_BD:
            ngauss=16;
            break;
        case GMIX_TURB:
            ngauss=3;
            break;
        default:
            fprintf(stderr, "bad simple gmix model type: %u", model);
            *flags |= GMIX_BAD_MODEL;
            ngauss=-1;
            break;
    }

    return ngauss;
}

long gmix_get_coellip_ngauss(long npars, long *flags)
{
    long ngauss=0;

    if ( ((npars-4) % 2) != 0) {
        fprintf(stderr, "bad npars for coellip: %ld", npars);
        ngauss = -1;
        *flags |= GMIX_WRONG_NPARS; 
    } else {
        ngauss = (npars-4)/2;
    }
    return ngauss;

}
long gmix_get_full_ngauss(long npars, long *flags)
{
    long ngauss=0;

    if ( (npars % 6) != 0) {
        fprintf(stderr, "bad npars for full: %ld", npars);
        ngauss = -1;
        *flags |= GMIX_WRONG_NPARS; 
    } else {
        ngauss = npars/6;
    }
    return ngauss;

}


struct gmix* gmix_new(size_t ngauss, long *flags)
{
    struct gmix*self=NULL;
    if (ngauss == 0) {
        wlog("number of gaussians must be > 0\n");
        *flags |= GMIX_ZERO_GAUSS;
        return NULL;
    }

    self = calloc(1, sizeof(struct gmix));
    if (self==NULL) {
        wlog("could not allocate struct gmix\n");
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

struct gmix* gmix_new_empty_simple(enum gmix_model model, long *flags)
{
    struct gmix*self=NULL;
    long ngauss=0;

    ngauss=gmix_get_simple_ngauss(model, flags);

    if (ngauss <= 0) {
        return NULL;
    }

    self=gmix_new(ngauss,flags);
    return self;
}

struct gmix* gmix_new_empty_coellip(long npars, long *flags)
{
    struct gmix *self=NULL;
    long ngauss=0;
    ngauss=gmix_get_coellip_ngauss(npars, flags);
    if (ngauss <= 0) {
        return NULL;
    }
    self = gmix_new(ngauss, flags);
    return self;
}

struct gmix* gmix_new_empty_full(long npars, long *flags)
{
    struct gmix *self=NULL;
    long ngauss=0;
    ngauss=gmix_get_full_ngauss(npars, flags);
    if (ngauss <= 0) {
        return NULL;
    }
    self = gmix_new(ngauss, flags);
    return self;
}


struct gmix* gmix_new_model(enum gmix_model model, double *pars, long npars, long *flags)
{
    struct gmix *self=NULL;

    if (model==GMIX_COELLIP) {
        self = gmix_new_empty_coellip(npars, flags);
    } else if (model==GMIX_FULL) {
        self = gmix_new_empty_full(npars, flags);
    } else {
        self = gmix_new_empty_simple(model, flags);
    }
    if (!self) {
        return NULL;
    }

    gmix_fill_model(self, model, pars, npars, flags);
    if (*flags) {
        self=gmix_free(self);
    }

    return self;
}

/*
   fill is provided so we aren't constantly hitting the heap
*/
void gmix_fill_model(struct gmix *self,
                     enum gmix_model model,
                     double *pars,
                     long npars,
                     long *flags)
{
    switch (model) {
        case GMIX_EXP:
            gmix_fill_exp6(self, pars, npars, flags);
            break;
        case GMIX_DEV:
            gmix_fill_dev10(self, pars, npars, flags);
            break;
        case GMIX_BD:
            gmix_fill_bd(self, pars, npars, flags);
            break;
        case GMIX_COELLIP:
            gmix_fill_coellip(self, pars, npars, flags);
            break;
        case GMIX_FULL:
            gmix_fill_full(self, pars, npars, flags);
            break;
        default:
            fprintf(stderr, "bad simple gmix model type: %u\n", model);
            *flags |= GMIX_BAD_MODEL;
            break;
    }

    return;
}



struct gmix *gmix_free(struct gmix *self)
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
               double icc,
               long *flags)
{
    self->det = irr*icc - irc*irc;

    if (self->det <= 0) {
        *flags |= GMIX_ERROR_NEGATIVE_DET; 
    }

    self->p   = p;
    self->row = row;
    self->col = col;
    self->irr = irr;
    self->irc = irc;
    self->icc = icc;

    self->drr = self->irr/self->det;
    self->drc = self->irc/self->det;
    self->dcc = self->icc/self->det;
    self->norm = 1./(M_TWO_PI*sqrt(self->det));

    self->pnorm = p*self->norm;
}
void gmix_set_dets(struct gmix *self)
{
    struct gauss *gauss = NULL;
    size_t i=0;
    for (i=0; i<self->size; i++) {
        gauss = &self->data[i];
        gauss->det = gauss->irr*gauss->icc - gauss->irc*gauss->irc;
    }
}

long gmix_verify(const struct gmix *self)
{
    size_t i=0;
    const struct gauss *gauss=NULL;

    if (!self || !self->data) {
        DBG wlog("gmix is not initialized\n");
        return 0;
    }

    for (i=0; i<self->size; i++) {
        gauss=&self->data[i];
        if (gauss->det <= 0) {
            DBG wlog("gmix_verify found det: %.16g\n", gauss->det);
            return 0;
        }
    }
    return 1;
}


void gmix_copy(const struct gmix *self, struct gmix* dest, long *flags)
{
    if (dest->size != self->size) {
        wlog("gmix are not same size\n");
        *flags |= GMIX_MISMATCH_SIZE;
        return;
    }
    memcpy(dest->data, self->data, self->size*sizeof(struct gauss));
    return;
}
struct gmix *gmix_newcopy(const struct gmix *self, long *flags)
{
    struct gmix *dest=NULL;

    dest=gmix_new(self->size, flags);
    if (dest) {
        memcpy(dest->data, self->data, self->size*sizeof(struct gauss));
    }

    return dest;
}


void gmix_print(const struct gmix *self, FILE* fptr)
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

double gmix_wmomsum(const struct gmix* self)
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

void gmix_get_cen(const struct gmix *self, double *row, double *col)
{
    long i=0;
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

void gmix_set_cen(struct gmix *self, double row, double col)
{
    long i=0;
    struct gauss *gauss=NULL;

    double row_cur=0, col_cur=0;
    double row_shift=0, col_shift=0;

    gmix_get_cen(self, &row_cur, &col_cur);

    row_shift = row - row_cur;
    col_shift = col - col_cur;

    for (i=0; i<self->size; i++) {
        gauss=&self->data[i];

        gauss->row += row_shift;
        gauss->col += col_shift;
    }
}


double gmix_get_T(const struct gmix *self)
{
    long i=0;
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
double gmix_get_psum(const struct gmix *self)
{
    long i=0;
    const struct gauss *gauss=NULL;
    double psum=0;

    for (i=0; i<self->size; i++) {
        gauss=&self->data[i];

        psum += gauss->p;
    }
    return psum;
}
void gmix_set_psum(struct gmix *self, double psum)
{
    long i=0;
    double psum_cur=0, rat=0;
    struct gauss *gauss=NULL;

    psum_cur=gmix_get_psum(self);
    rat=psum/psum_cur;

    for (i=0; i<self->size; i++) {
        gauss=&self->data[i];

        gauss->p *= rat;
    }
}

void gmix_set_total_moms(struct gmix *self)
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


void gmix_convolve_fill(struct gmix *self, 
                        const struct gmix *obj_gmix, 
                        const struct gmix *psf_gmix,
                        long *flags)
{
    struct gauss *psf=NULL, *obj=NULL, *comb=NULL;

    long ntot=0, iobj=0, ipsf=0;
    double irr=0, irc=0, icc=0, psum=0;
    double row=0, col=0;
    double psf_rowcen=0, psf_colcen=0;

    ntot = obj_gmix->size*psf_gmix->size;
    if (ntot != self->size) {
        fprintf(stderr,
                "gmix (%lu) wrong size to accept convolution %ld\n",
                self->size, ntot);
        *flags |= GMIX_MISMATCH_SIZE;
        return;
    }

    gmix_get_cen(psf_gmix, &psf_rowcen, &psf_colcen);

    for (ipsf=0; ipsf<psf_gmix->size; ipsf++) {
        psf = &psf_gmix->data[ipsf];
        psum += psf->p;
    }

    comb = self->data;
    for (iobj=0; iobj<obj_gmix->size; iobj++) {
        obj = &obj_gmix->data[iobj];

        for (ipsf=0; ipsf<psf_gmix->size; ipsf++) {
            psf = &psf_gmix->data[ipsf];

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
                      irr, irc, icc,
                      flags);
            if (*flags) {
                return;
            }

            comb++;
        }
    }

    return;
}

struct gmix *gmix_convolve(const struct gmix *obj_gmix,
                           const struct gmix *psf_gmix,
                           long *flags)
{
    struct gmix *self=NULL;
    long ntot=0;
    ntot = obj_gmix->size*psf_gmix->size;
    self= gmix_new(ntot,flags);
    if (!self) {
        return NULL;
    }

    gmix_convolve_fill(self, obj_gmix, psf_gmix,flags);
    if (*flags) {
        self=gmix_free(self);
    }
    return self;
}



void gmix_fill_full(struct gmix *self, double *pars, long npars, long *flags)
{
    long ngauss=0;
    struct gauss *gauss=NULL;

    long i=0, beg=0;

    ngauss=gmix_get_full_ngauss(npars,flags);
    if (ngauss <= 0) {
        *flags |= GMIX_ZERO_GAUSS;
        return;
    }
    if (self->size != ngauss) {
        fprintf(stderr,"full gmix (%lu) has wrong size, expected %ld\n",
                self->size, ngauss);
        *flags |= GMIX_MISMATCH_SIZE;
        return;
    }

    for (i=0; i<ngauss; i++) {
        gauss = &self->data[i];

        beg = i*6;

        gauss_set(gauss,
                  pars[beg+0],  // p
                  pars[beg+1],  // row
                  pars[beg+2],  // col
                  pars[beg+3],  // irr
                  pars[beg+4],  // irc
                  pars[beg+5],
                  flags); // icc
        if (*flags) {
            return;
        }
    }

    return;
}


void gmix_fill_coellip(struct gmix *self, double *pars, long npars, long *flags)
{
    long ngauss=0, Tstart=0, Astart=0;
    double row=0, col=0, g1=0, g2=0, e1=0, e2=0, Ti=0, Ai=0;
    struct gauss *gauss=NULL;

    long i=0;

    ngauss=gmix_get_coellip_ngauss(npars,flags);
    if (ngauss <= 0) {
        *flags |= GMIX_ZERO_GAUSS;
        return;
    }
    if (self->size != ngauss) {
        fprintf(stderr,"coellip gmix (%lu) has wrong size, expected %ld\n",
                self->size, ngauss);
        *flags |= GMIX_MISMATCH_SIZE;
        return;
    }

    row=pars[0];
    col=pars[1];
    g1 = pars[2];
    g2 = pars[3];

    Tstart=4;
    Astart=Tstart+ngauss;

    if (!g1g2_to_e1e2(g1,g2,&e1,&e2)) {
        *flags |= GMIX_ERROR_G_RANGE;
        return;
    }

    for (i=0; i<self->size; i++) {
        gauss = &self->data[i];

        Ti = pars[Tstart+i];
        Ai = pars[Astart+i];

        gauss_set(gauss,
                  Ai,
                  row, 
                  col, 
                  (Ti/2.)*(1-e1),
                  (Ti/2.)*e2,
                  (Ti/2.)*(1+e1),
                  flags);
        if (*flags) {
            return;
        }
    }

    return;
}



struct gmix *gmix_new_coellip(double *pars, long npars, long *flags)
{
    struct gmix *self=NULL;

    self=gmix_new_empty_coellip(npars,flags);
    if (self) {
        gmix_fill_coellip(self, pars, npars,flags);
        if (*flags) {
            self=gmix_free(self);
        }
    }
    return self;
}


/* no error checking except on shape */
static void _gmix_fill_simple(struct gmix *self,
                              double *pars, 
                              const double *Fvals, 
                              const double *pvals,
                              long *flags)
{
    double row=0, col=0, g1=0, g2=0, e1=0, e2=0;
    double T=0, T_i=0;
    double counts=0, counts_i=0;

    struct gauss *gauss=NULL;

    long i=0;

    row=pars[0];
    col=pars[1];
    g1=pars[2];
    g2=pars[3];
    T=pars[4];
    counts=pars[5];

    if (!g1g2_to_e1e2(g1,g2,&e1,&e2)) {
        *flags |= GMIX_ERROR_G_RANGE;
        return;
    }

    for (i=0; i<self->size; i++) {
        gauss=&self->data[i];

        T_i = T*Fvals[i];
        counts_i=counts*pvals[i];

        gauss_set(gauss,
                  counts_i,
                  row, col, 
                  (T_i/2.)*(1-e1), 
                  (T_i/2.)*e2,
                  (T_i/2.)*(1+e1),
                  flags);
        if (*flags) {
            return;
        }
    }

    return;
}

void gmix_fill_bd(struct gmix *self, double *pars, long npars, long *flags)
{
    static const long 
        npars_expected=8,
        npars_exp=6, npars_dev=6, ngauss_exp=6, ngauss_dev=10,
        ngauss_expected=16;

    double pars_exp[6], pars_dev[6];
    struct gmix *gmix_exp=NULL, *gmix_dev=NULL;

    if (npars != npars_expected) {
        fprintf(stderr,"wrong par len for bulge+disk: %ld\n", npars);
        *flags |= GMIX_WRONG_NPARS;
        return;
    }
    if (self->size != ngauss_expected) {
        fprintf(stderr,"bulge+disk gmix size %lu != %ld\n", self->size, ngauss_expected);
        *flags |= GMIX_MISMATCH_SIZE; 
        return;
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

    gmix_exp = gmix_new_model(GMIX_EXP,pars_exp,npars_exp, flags);
    if (*flags) {
        return;
    }
    gmix_dev = gmix_new_model(GMIX_DEV,pars_dev,npars_dev, flags);
    if (*flags) {
        return;
    }

    memcpy(self->data,
           gmix_exp->data,
           ngauss_exp*sizeof(struct gauss));
    memcpy(self->data+ngauss_exp,
           gmix_dev->data,
           ngauss_dev*sizeof(struct gauss));

    gmix_exp=gmix_free(gmix_exp);
    gmix_dev=gmix_free(gmix_dev);

    return;
}



void gmix_fill_dev10(struct gmix *self, double *pars, long npars, long *flags)
{
    static const long npars_expected=6, ngauss_expected=10;
    if (npars != npars_expected) {
        fprintf(stderr,"dev10: expected npars=%ld, got %ld\n", npars_expected, npars);
        *flags |= GMIX_WRONG_NPARS;
        return;
    }
    if (self->size != ngauss_expected) {
        fprintf(stderr,"dev10: expected ngauss=%ld, got %lu\n", ngauss_expected, self->size);
        *flags |= GMIX_MISMATCH_SIZE; 
        return;
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

    _gmix_fill_simple(self, pars, Fvals, pvals, flags);
}



void gmix_fill_exp6(struct gmix *self, double *pars, long npars, long *flags)
{
    static const long npars_expected=6, ngauss_expected=6;
    if (npars != npars_expected) {
        fprintf(stderr,"exp6: expected npars=%ld, got %ld\n", npars_expected, npars);
        *flags |= GMIX_WRONG_NPARS;
        return;
    }
    if (self->size != ngauss_expected) {
        fprintf(stderr,"exp6: expected ngauss=%ld, got %lu\n", ngauss_expected, self->size);
        *flags |= GMIX_MISMATCH_SIZE; 
        return;
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

    _gmix_fill_simple(self, pars, Fvals, pvals, flags);
}



void gmix_fill_turb3(struct gmix *self, double *pars, long npars, long *flags)
{
    static const long npars_expected=6, ngauss_expected=3;
    if (npars != npars_expected) {
        fprintf(stderr,"turb3: expected npars=%ld, got %ld\n", npars_expected, npars);
        *flags |= GMIX_WRONG_NPARS;
        return;
    }
    if (self->size != ngauss_expected) {
        fprintf(stderr,"turb3: expected ngauss=%ld, got %lu\n", ngauss_expected, self->size);
        *flags |= GMIX_MISMATCH_SIZE; 
        return;
    }

    static const double Fvals[3] = 
        {0.5793612389470884,1.621860687127999,7.019347162356363};
    static const double pvals[3] = 
        {0.596510042804182,0.4034898268889178,1.303069003078001e-07};

    _gmix_fill_simple(self, pars, Fvals, pvals, flags);
}


