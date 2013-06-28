#include <stdlib.h>
#include <stdio.h>
#include "prob.h"
#include "render.h"

struct prob_data_simple_ba *prob_data_simple_ba_new(enum gvec_model model,
                                                    const struct observations *obs

                                                    const struct dist_gauss *cen1_prior,
                                                    const struct dist_gauss *cen2_prior,

                                                    const struct dist_g_ba *g_prior,

                                                    const struct dist_lognorm *T_prior,
                                                    const struct dist_lognorm *counts_prior)


{

    long ngauss_tot=0;

    struct prob_data_simple_ba *data=calloc(1, sizeof(struct prob_data_simple_ba));
    if (!prob_data) {
        fprintf(stderr,"could not allocate struct prob_data_simple_ba\n");
        exit(EXIT_FAILURE);
    }

    data->model=model;
    data->obs_list = obs;

    data->obj0 = gvec_new_empty_simple(model);
    ngauss_tot = psf_list->data[0].size * data->obj0->size;

    data->obj = gvec_new(ngauss_tot);

    data->cen1_prior = cen1_prior;
    data->cen2_prior = cen2_prior;
    data->g_prior = g_prior;
    data->T_prior = T_prior;
    data->counts_prior = counts_prior;
}

struct prob_data_simple_ba *prob_data_simple_ba_free(struct prob_data_simple_ba *self)
{
    if (self) {
        self->obj0=gvec_free(self->obj0);
        self->obj=gvec_free(self->obj);
        free(self);
    }
    return NULL;
}

static long prob_simple_ba_calc_likelihood(struct prob_data_simple_ba *self,
                                           double *pars, long npars,
                                           double *s2n_numer, double *s2n_denom,
                                           long *loglike)
{

    long i=0, flags=0;
    double t_likelihood=0, t_s2n_numer=0, t_s2n_denom=0;
    struct observation *obj=NULL;

    *loglike=0;
    *s2n_numer=0;
    *s2n_denom=0;

    if (!gvec_fill_model(self->obj0,self->model,pars,npars)) {
        flags = GMIX_ERROR_MODEL;
        goto _prob_simple_ba_calc_like_bail;
    }

    for (i=0; i<self->obs_list->size; i++) {
        obs=&self->obs_list->data[i];

        if (!gvec_convolve_fill(self->obj, self->obj0, obs->psf)) {
            flags = GMIX_ERROR_MODEL;
            goto _prob_simple_ba_calc_like_bail;
        }
        flags= calculate_loglike_wt_jacob(obs->image, 
                                          obs->weight,
                                          obs->jacob,
                                          self->obj,
                                          t_s2n_numer,
                                          t_s2n_denom,
                                          &t_loglike);
        if (flags != 0) {
            goto _prob_simple_ba_calc_like_bail;
        }

        (*loglike) += loglike;

        *s2n_numer += t_s2n_numer;
        *s2n_denom += t_s2n_denom;
    }

_prob_simple_ba_calc_like_bail:
    if (flags != 0) {
        lnprob = LOG_LOWVAL;
        *s2n_numer=0;
        *s2n_denom=0;
    }

    return flags;
}

long prob_simple_ba_calc_priors(struct prob_data_simple_ba *self,
                                double *pars, long npars,
                                long *lnprob)
{
    (*lnprob) = 0;

    (*lnprob) += dist_gauss_lnprob(self->cen1_prior,pars[0]);
    (*lnprob) += dist_gauss_lnprob(self->cen2_prior,pars[1]);

    (*lnprob) += dist_g_ba_lnprob(self->g_prior,pars[2],pars[3]);

    (*lnprob) += dist_lognorm_lnprob(self->T_prior,pars[4]);
    (*lnprob) += dist_lognorm_lnprob(self->counts_prior,pars[5]);

    return 0;
}

long prob_simple_ba_calc(struct prob_data_simple_ba *self,
                           double *pars, long npars,
                           double *s2n_numer, double *s2n_denom,
                           long *lnprob)
{

    long flags=0;
    double loglike=0, priors_lnprob=0;

    flags=prob_simple_ba_calc_likelihood(self, pars, npars,
                                         s2n_numer, s2n_denom,
                                         &loglike);
    if (flags != 0) {
        goto _prob_simple_ba_calc_bail;
    }

    (*lnprob) += loglike;

    flags = prob_simple_ba_calc_priors(self, pars, npars, &priors_lnprob);
    if (flags != 0) {
        goto _prob_simple_ba_calc_bail;
    }

    (*lnprob) += priors_lnprob;

_prob_simple_ba_calc_bail:
    if (flags != 0) {
        (*lnprob) = LOG_LOWVAL;
        *s2n_numer=0;
        *s2n_denom=0;
    }
    return flags;
}