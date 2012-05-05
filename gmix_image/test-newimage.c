#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "newimage.h"
#include "bound.h"
#include "defs.h"

int main(int argc, char** argv)
{
    struct image *image=NULL;
    struct bound bound={0};
    double sky=10.0;
    double *col=NULL, *end=NULL;
    size_t row=0, i=0;

    image = image_new(10,10);
    assert(IM_IS_OWNER(image));

    image_add_scalar(image, sky);
    
    assert(sky*IM_SIZE(image) == IM_COUNTS(image));


    bound_set(&bound,3,5,4,8);

    struct image *imsub = image_sub(image, &bound);

    assert(IM_PARENT(imsub) == image);
    assert(!IM_IS_OWNER(imsub));
    assert(IM_SIZE(imsub) == (5-3+1)*(8-4+1));
    assert(IM_NROWS(imsub) == (5-3+1));
    assert(IM_NCOLS(imsub) == (8-4+1));
    assert(sky*IM_SIZE(imsub) == IM_COUNTS(imsub));


    i=0;
    for (row=0; row<IM_NROWS(image); row++) {
        col = IM_ROW_ITER(image,row);
        end = IM_ROW_END(image,row);
        for (; col!=end; col++) {
            *col = i;
            i++;
        }
    }


    printf("image\n");
    image_write(image,stdout);
    printf("sub image\n");
    image_write(imsub,stdout);

    image = image_free(image);
    imsub = image_free(imsub);
}
