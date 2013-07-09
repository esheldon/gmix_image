/*
   C only handling of posterior, including priors
*/
#ifndef _PROB_HEADER_GUARD
#define _PROB_HEADER_GUARD

#include "image.h"
#include "jacobian.h"
#include "gmix.h"
#include "dist.h"
#include "obs.h"

enum prob_type {
    PROB_BA13=1,
    PROB_NOSPLIT_ETA=2
};

// BA13 g prior
// log normal priors on T and flux
// gaussian prior on center
struct prob_data_simple_ba {
    const struct obs_list *obs_list;

    enum gmix_model model;
    struct gmix *obj0;
    struct gmix *obj;

    // priors

    // currently cen prior is always gaussian in both directions
    struct dist_gauss cen1_prior;
    struct dist_gauss cen2_prior;

    struct dist_g_ba g_prior;

    struct dist_lognorm T_prior;
    struct dist_lognorm counts_prior;
};

struct prob_data_simple_ba *prob_data_simple_ba_new(enum gmix_model model,
                                                    const struct obs_list *obs_list,

                                                    const struct dist_gauss *cen1_prior,
                                                    const struct dist_gauss *cen2_prior,

                                                    const struct dist_g_ba *g_prior,

                                                    const struct dist_lognorm *T_prior,
                                                    const struct dist_lognorm *counts_prior,
                                                    long *flags);
 
struct prob_data_simple_ba *prob_data_simple_ba_free(struct prob_data_simple_ba *self);
                                                 
void prob_simple_ba_calc_priors(struct prob_data_simple_ba *self,
                                double *pars, long npars,
                                double *lnprob,
                                long *flags);


void prob_simple_ba_calc_likelihood(struct prob_data_simple_ba *self,
                                    double *pars,
                                    long npars,
                                    double *s2n_numer,
                                    double *s2n_denom,
                                    double *loglike,
                                    long *flags);

// calculate the lnprob for the input pars
// also running s/n values
void prob_simple_ba_calc(struct prob_data_simple_ba *self,
                         double *pars, long npars,
                         double *s2n_numer, double *s2n_denom,
                         double *lnprob,
                         long *flags);


// using gaussian mixture in eta space
struct prob_gmix_eta_simple {
    const struct obs_list *obs_list;

    enum gmix_model model;
    struct gmix *obj0;
    struct gmix *obj;

    // priors

    // currently cen prior is always gaussian in both directions
    struct dist_gauss cen1_prior;
    struct dist_gauss cen2_prior;

    struct dist_gmix3_eta eta_prior;

    struct dist_lognorm T_prior;
    struct dist_lognorm counts_prior;
};


#endif
