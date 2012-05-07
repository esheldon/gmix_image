#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "image.h"
#include "bound.h"
#include "defs.h"

int main(int argc, char** argv)
{
    struct image *image=NULL;
    struct bound bound={0};
    double sky=100.0;
    size_t row=0, col=0, i=0;
    size_t nrows=10, ncols=10;
    double counts=0, masked_counts=0, added_counts=0;
    image = image_new(nrows,ncols);

    image_add_scalar(image, sky);
    
    assert(sky*IM_SIZE(image) == IM_COUNTS(image));
    assert(!IM_HAS_MASK(image));

    bound_set(&bound,3,5,4,8);

    image_add_mask(image, &bound);

    assert(IM_HAS_MASK(image));

    assert(IM_SIZE(image) == (5-3+1)*(8-4+1));
    assert(IM_NROWS(image) == (5-3+1));
    assert(IM_NCOLS(image) == (8-4+1));
    assert(sky*IM_SIZE(image) == IM_COUNTS(image));

    // check parent is still consistent
    assert(IM_PARENT_SIZE(image) == nrows*ncols);
    assert(IM_PARENT_NROWS(image) == nrows);
    assert(IM_PARENT_NCOLS(image) == ncols);
    assert(sky*IM_PARENT_SIZE(image) == IM_PARENT_COUNTS(image));

    i=0;
    counts=IM_PARENT_COUNTS(image);
    masked_counts=IM_COUNTS(image);
    added_counts=0;
    for (row=0; row<IM_NROWS(image); row++) {
        for (col=0; col<IM_NCOLS(image); col++) {
            added_counts += i - IM_GET(image,row,col);
            IM_SET(image, row, col, i);
            i++;
        }
    }

    assert((masked_counts+added_counts) == IM_COUNTS(image));
    assert((counts+added_counts) == IM_PARENT_COUNTS(image));

    printf("sub image\n");
    image_write(image,stdout);

    IM_UNMASK(image);
    printf("image\n");
    image_write(image,stdout);

    image = image_free(image);
}
