#ifndef _GMIX_IMAGE_CONVOLVED_HEADER_GUARD
#define _GMIX_IMAGE_CONVOLVED_HEADER_GUARD

#include "image.h"
#include "gvec.h"


int gmix_image_convolved(struct gmix* self,
                         struct image *image, 
                         struct gvec *gvec,
                         struct gvec *gvec_psf,
                         size_t *iter,
                         double *fdiff);

int gmix_get_sums_convolved(struct image *image,
                            struct gvec *gvec,
                            struct gvec *gvec_psf,
                            struct iter* iter);


#endif
