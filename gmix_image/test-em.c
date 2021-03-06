#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "image.h"
#include "gvec.h"
#include "gmix_em.h"
#include "defs.h"

int main(int argc, char** argv)
{
    size_t ngauss=2;
    struct gvec *ginit, *gvec;
    struct image *image;
    struct gmix gmix;
    struct gauss *gptr=NULL;
    size_t niter;
    struct timespec ts_start;
    struct timespec ts_end;

    if (argc != 2) {
        printf("usage: test image\n");
        exit(1);
    }

    gmix.maxiter=1000;
    gmix.tol = 1.e-6;
    //gmix.tol = 1.e-5;
    gmix.fixsky = 1;
    gmix.verbose=0;

    ginit = gvec_new(ngauss);
    gptr = ginit->data;

    gptr[0].p = 0.6;
    gptr[0].irr=2.0;
    gptr[0].irc=0.0;
    gptr[0].icc=2.0;
    gptr[0].row = 15;
    gptr[0].col = 15;

    gptr[1].p = 0.4;
    gptr[1].irr=1.5;
    gptr[1].irc=0.3;
    gptr[1].icc=4.0;
    gptr[1].row = 10;
    gptr[1].col = 8;
 
    // set the deteminants
    gvec_set_dets(ginit);

    // read the data
    image = image_read_text(argv[1]);

    if (image==NULL)
        exit(EXIT_FAILURE);

    wlog("nrows: %lu\n", IM_NROWS(image));
    wlog("ncols: %lu\n", IM_NCOLS(image));
    wlog("sky: %.16g\n", IM_SKY(image));
    wlog("counts: %.16g\n", IM_COUNTS(image));
    wlog("image[7,9]: %.16g\n", IM_GET(image, 7, 9));
    wlog("image[9,7]: %.16g\n", IM_GET(image, 9, 7));

    gvec = gvec_new(ginit->size);
    gvec_copy(ginit, gvec);

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    double fdiff=0;
    int flags = gmix_em(&gmix, image, gvec, NULL, &niter, &fdiff);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double sec = 
        ((double)ts_end.tv_sec-ts_start.tv_sec)
        +
        1.e-9*((double)(ts_end.tv_nsec-ts_start.tv_nsec));
    wlog("time: %.7lf sec\n", sec);


    wlog("numiter: %lu\n", niter);
    if (flags != 0) {
        wlog("failure with flags: %d\n", flags);
    } else {

        gptr = ginit->data;
        wlog("input\n");
        gvec_print(ginit,stderr);

        wlog("meas\n");
        gvec_print(gvec,stderr);
    }
}
