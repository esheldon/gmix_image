#ifndef _OBSERVATION_HEADER_GUARD
#define _OBSERVATION_HEADER_GUARD

struct observation {
    const struct image *image;
    const struct image *weight;
    const struct jacobian *jacob;
    const struct gmix *psf;
};

struct observations {
    size_t size;
    struct prob_observation *data;
};


#endif
