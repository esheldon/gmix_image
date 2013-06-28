#include "observation.h"

struct observation *observation_new(const struct image *image,
                                    const struct image *weight,;
                                    const struct jacobian *jacob,
                                    const struct gmix *psf)
{
    struct observation *self=calloc(1,sizeof(struct observation));
    if (!self) {
        fprintf(stderr,"could not allocate struct observation\n");
        exit(1);
    }

    self->image=image;
    self->weight=weight;
    self->jacob=jacob;
    self->psf=psf;

    return self;
}


struct observation *observation_free(struct observation *self)
{
    free(self);
    return NULL;
}
