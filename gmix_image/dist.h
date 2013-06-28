/*
   A set of distributions
*/
#ifndef _PRIORS_HEADER_GUARD
#define _PRIORS_HEADER_GUARD

enum dist {
    DIST_GAUSS,
    DIST_LOGNORMAL,
    DIST_G_BA
};

struct dist_gauss {
    double mean;
    double sigma;
    double ivar;
};

struct dist_lognorm {
    double mean;
    double sigma;

    double logmean;
    double logivar;
};

struct dist_g_ba {
    double sigma;
    double ivar;
};


struct dist_gauss *dist_gauss_new(double mean, double sigma);
double dist_gauss_lnprob(const struct dist_gauss *self, double x);

struct dist_lognorm *dist_lognorm_new(double mean, double sigma);
double dist_lognorm_lnprob(const struct dist_gauss *self, double x);

struct dist_g_ba *dist_g_ba_new(double sigma);
double dist_g_ba_lnprob(const struct dist_g_ba *self, double g1, double g2);

#endif

