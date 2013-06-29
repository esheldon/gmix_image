#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "dist.h"
#include "defs.h"

void dist_gauss_fill(struct dist_gauss *self, double mean, double sigma)
{
    self->mean=mean;
    self->sigma=sigma;
    self->ivar=1./(sigma*sigma);
}
struct dist_gauss *dist_gauss_new(double mean, double sigma)
{
    struct dist_gauss *self=calloc(1, sizeof(struct dist_gauss));
    if (!self) {
        fprintf(stderr,"Could not allocate struct dist_gauss\n");
        exit(1);
    }

    dist_gauss_fill(self, mean, sigma);
    return self;
}
double dist_gauss_lnprob(const struct dist_gauss *self, double x)
{
    double lnp=0.0;

    // -0.5*self->ivar*(x-self->mean)**2
    lnp = x;
    lnp -= self->mean;
    lnp *= lnp;
    lnp *= self->ivar;
    lnp *= (-0.5);
    return lnp;
}
void dist_gauss_print(const struct dist_gauss *self, FILE *stream)
{
    fprintf(stream,"guass dist\n");
    fprintf(stream,"    mean: %g\n", self->mean);
    fprintf(stream,"    sigma: %g\n", self->sigma);
}



void dist_lognorm_fill(struct dist_lognorm *self, double mean, double sigma)
{
    self->mean=mean;
    self->sigma=sigma;

    self->logmean = log(mean) - 0.5*log( 1 + sigma*sigma/(mean*mean) );
    self->logivar = 1./(  log(1 + sigma*sigma/(mean*mean) ) );
}

struct dist_lognorm *dist_lognorm_new(double mean, double sigma)
{
    struct dist_lognorm *self=calloc(1, sizeof(struct dist_lognorm));
    if (!self) {
        fprintf(stderr,"Could not allocate struct dist_lognorm\n");
        exit(1);
    }

    dist_lognorm_fill(self, mean, sigma);
    return self;
}
double dist_lognorm_lnprob(const struct dist_lognorm *self, double x)
{
    double lnp=0.0, logx=0;

    //chi2 = self.logivar*(logx-self.logmean)**2;
    //lnprob = - 0.5*chi2 - logx;

    if (x < LOG_MINARG) {
        lnp = LOG_LOWVAL;
    } else {
        logx=log(x);
        lnp = logx;

        lnp -= self->logmean;
        lnp *= lnp;
        lnp *= self->logivar;

        lnp *= (-0.5);
        lnp -= logx;
    }
    return lnp;
}
void dist_lognorm_print(const struct dist_lognorm *self, FILE *stream)
{
    fprintf(stream,"lognorm dist\n");
    fprintf(stream,"    mean: %g\n", self->mean);
    fprintf(stream,"    sigma: %g\n", self->sigma);
}




void dist_g_ba_fill(struct dist_g_ba *self, double sigma)
{
    self->sigma=sigma;
    self->ivar=1./(sigma*sigma);
}


struct dist_g_ba *dist_g_ba_new(double sigma)
{
    struct dist_g_ba *self=calloc(1, sizeof(struct dist_g_ba));
    if (!self) {
        fprintf(stderr,"Could not allocate struct dist_g_ba\n");
        exit(1);
    }

    dist_g_ba_fill(self, sigma);
    return self;
}

double dist_g_ba_lnprob(const struct dist_g_ba *self, double g1, double g2)
{
    double lnp=0, gsq=0, tmp=0;

    gsq = g1*g1 + g2*g2;

    tmp = 1-gsq;
    if ( tmp < LOG_MINARG ) {
        lnp = LOG_LOWVAL;
    } else {

        //p= (1-g**2)**2*exp(-0.5 * g**2 * ivar)
        // log(p) = 2*log(1-g^2) - 0.5*g^2 * ivar

        // should do a fast math version; I suspect this
        // will be a bottleneck
        lnp = log(tmp);

        lnp *= 2;
        
        tmp = 0.5;
        tmp *= gsq;
        tmp *= self->ivar;
        lnp -= tmp;
    }
    return lnp;

}

// p= (1-g**2)**2*exp(-0.5 * g**2 * ivar)
double dist_g_ba_prob(const struct dist_g_ba *self, double g1, double g2)
{
    double prob=0, gsq=0, chi2=0, tmp=0;

    gsq = g1*g1 + g2*g2;

    tmp = 1-gsq;
    if (tmp > 0) {
        tmp *= tmp;

        chi2 = gsq;
        chi2 *= self->ivar;

        prob = exp(-0.5*chi2);

        prob *= tmp;
    }
    return prob;

}


void dist_g_ba_print(const struct dist_g_ba *self, FILE *stream)
{
    fprintf(stream,"g dist BA13\n");
    fprintf(stream,"    sigma: %g\n", self->sigma);
}
