#include <stdlib.h>
#include <stdio.h>
#include "dist.h"

struct dist_gauss *dist_gauss_new(double mean, double sigma)
{
    struct dist_gauss *self=calloc(1, sizeof(struct dist_gauss));
    if (!self) {
        fprintf(stderr,"Could not allocate struct dist_gauss\n");
        exit(1);
    }

    self->mean=mean;
    self->sigma=sigma;
    self->ivar=1./(sigma*sigma);

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

struct dist_lognorm *dist_lognorm_new(double mean, double sigma)
{
    struct dist_lognorm *self=calloc(1, sizeof(struct dist_lognorm));
    if (!self) {
        fprintf(stderr,"Could not allocate struct dist_lognorm\n");
        exit(1);
    }

    self->mean=mean;
    self->sigma=sigma;

    self->logmean = log(mean) - 0.5*log( 1 + sigma*sigma/(mean*mean) );
    self->logivar = 1./(  log(1 + sigma*sigma/(mean*mean) ) );

    return self;
}
double dist_lognorm_lnprob(const struct dist_lognorm *self, double x)
{
    double lnp=0.0;

    //chi2 = self.logivar*(logx-self.logmean)**2;
    //lnprob = - 0.5*chi2 - logx;

    if (x < LOG_MINARG) {
        lnp = LOG_LOWVAL;
    } else {
        lnp = log(x);

        lnp -= self->logmean;
        lnp *= lnp;
        lnp *= self->logivar;

        lnp *= (-0.5);
        lnp -= logx;
    }
    return lnp;
}

struct dist_g_ba *dist_g_ba_new(double sigma)
{
    struct dist_g_ba *self=calloc(1, sizeof(struct dist_g_ba));
    if (!self) {
        fprintf(stderr,"Could not allocate struct dist_g_ba\n");
        exit(1);
    }

    self->sigma=sigma;
    self->ivar=1./(sigma*sigma);

    return self;
}

double dist_g_ba_lnprob(const struct dist_g_ba *self, double g1, double g2)
{
    double lnp=0, g=0, g2=0, tmp=0;

    g=sqrt(g1**2 + g2**2);
    g2=g*g;

    tmp = 1-g2;
    if ( tmp < LOG_MINARG) {
        lnp = LOG_LOWVAL;
    } else {

        //p= (1-g2)**2*exp(-0.5 * g2 * ivar)
        // log(p) = 2*log(1-g^2) - 0.5*g^2 * ivar

        lnp = log(tmp);

        lnp *= 2;
        
        tmp = 0.5;
        tmp *= g2;
        tmp *= self->ivar;
        lnp -= tmp;
    }
    return lnp;

}
