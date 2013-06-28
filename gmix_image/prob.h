/*
   C only handling of posterior, including priors
*/
#ifndef _RENDER_HEADER_GUARD
#define _RENDER_HEADER_GUARD

#include "image.h"
#include "jacobian.h"
#include "gvec.h"
#include "dist.h"


// BA13 g prior
// log normal priors on T and flux
// gaussian prior on center

struct observation {
    const struct image *image;
    const struct image *weight;
    const struct jacobian *jacob;
    const struct gvec *psf;
};

struct observations {
    size_t size;
    struct prob_observation *data;
};

struct prob_data_simple_ba {
    const struct observations *obs_list;

    enum gvec_model model;
    struct gvec *obj0;
    struct gvec *obj;

    // priors

    // currently cen prior is always gaussian in both directions
    const struct dist_gauss *cen1_prior;
    const struct dist_gauss *cen2_prior;

    const struct dist_g_ba *g_prior;

    const struct dist_lognorm *T_prior;
    const struct dist_lognorm *counts_prior;
};

struct prob_data_simple_ba *prob_data_simple_ba_new(enum gvec_model model,
                                                    const struct observations *obs

                                                    const struct dist_gauss *cen1_prior,
                                                    const struct dist_gauss *cen2_prior,

                                                    const struct dist_g_ba *g_prior,

                                                    const struct dist_lognorm *T_prior,
                                                    const struct dist_lognorm *counts_prior);
 
struct prob_data_simple_ba *prob_data_simple_ba_free(struct prob_data_simple_ba *self);
                                                 
#endif
