#ifndef _CONVERT_HEADER_GUARD
#define _CONVERT_HEADER_GUARD
// out of range inputs are scaled to be in range
long g1g2_to_e1e2(double g1, double g2, double *e1, double *e2);
long eta1eta2_to_g1g2(double eta1, double eta2, double *g1, double *g2);

#endif
