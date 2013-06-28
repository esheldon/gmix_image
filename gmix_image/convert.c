#include <math.h>
#include "convert.h"

long g1g2_to_e1e2(double g1, double g2, double *e1, double *e2)
{
    double g=0,e=0,fac=0;

    g = sqrt(g1*g1 + g2*g2);

    if (g == 0) {
        *e1=0;
        *e2=0;
    } else {

        if (g >= 1.) {
            *e1=-9999;
            *e2=-9999;
            return 0;
        }

        e = tanh(2*atanh(g));
        if (e >= 1.) {
            // round off error most likely
            e = 0.99999999;
        }
        fac = e/g;

        *e1 = fac*g1;
        *e2 = fac*g2;
    }
    return 1;
}


