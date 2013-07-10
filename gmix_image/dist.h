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

// 3 round gaussians centered at zero
// evaluation is easy,
//    prob = sum_i( N_i*exp( -0.5*ivar*( eta_1^2 + eta_2^2 ) )
// where N_i is amp_i/(2*PI)*ivar

struct dist_gmix3_eta {
    double gauss1_ivar;
    double gauss1_pnorm; // amp*norm = amp*ivar/(2*PI)

    double gauss2_ivar;
    double gauss2_pnorm;

    double gauss3_ivar;
    double gauss3_pnorm;
};

// eta_sq = eta1**2 + eta2**2
/*
#define DIST_GMIX3_ETA_EVAL(dist, eta_sq) ({                               \
    double _prob=0;                                                        \
                                                                           \
    _prob += (dist)->gauss1_pnorm*exp(-0.5*(dist)->gauss1_ivar*(eta_sq) ); \
    _prob += (dist)->gauss2_pnorm*exp(-0.5*(dist)->gauss2_ivar*(eta_sq) ); \
    _prob += (dist)->gauss3_pnorm*exp(-0.5*(dist)->gauss3_ivar*(eta_sq) ); \
                                                                           \
    _prob;                                                                 \
})
*/



struct dist_gauss *dist_gauss_new(double mean, double sigma);
void dist_gauss_fill(struct dist_gauss *self, double mean, double sigma);
double dist_gauss_lnprob(const struct dist_gauss *self, double x);
void dist_gauss_print(const struct dist_gauss *self, FILE *stream);

struct dist_lognorm *dist_lognorm_new(double mean, double sigma);
void dist_lognorm_fill(struct dist_lognorm *self, double mean, double sigma);
double dist_lognorm_lnprob(const struct dist_lognorm *self, double x);
void dist_lognorm_print(const struct dist_lognorm *self, FILE *stream);

struct dist_g_ba *dist_g_ba_new(double sigma);
void dist_g_ba_fill(struct dist_g_ba *self, double sigma);
double dist_g_ba_lnprob(const struct dist_g_ba *self, double g1, double g2);
double dist_g_ba_prob(const struct dist_g_ba *self, double g1, double g2);
void dist_g_ba_print(const struct dist_g_ba *self, FILE *stream);

void dist_gmix3_eta_fill(struct dist_gmix3_eta *self,
                         double ivar1, double p1,
                         double ivar2, double p2,
                         double ivar3, double p3);



struct dist_gmix3_eta *dist_gmix3_eta_new(double ivar1, double p1,
                                          double ivar2, double p2,
                                          double ivar3, double p3);
double dist_gmix3_eta_lnprob(const struct dist_gmix3_eta *self, double eta1, double eta2);
double dist_gmix3_eta_prob(const struct dist_gmix3_eta *self, double eta1, double eta2);
void dist_gmix3_eta_print(const struct dist_gmix3_eta *self, FILE *stream);

#endif

