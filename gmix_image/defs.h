#ifndef _GMIX_DEFINITIONS_H
#define _GMIX_DEFINITIONS_H

#include <math.h>

// these are the same
#define GMIX_ERROR_NEGATIVE_DET 0x1
#define GMIX_ERROR_G_RANGE 0x1
#define GMIX_ERROR_MAXIT 0x2
#define GMIX_ERROR_NEGATIVE_DET_COCENTER 0x4
#define GMIX_ERROR_MODEL 0x8
#define GMIX_BAD_MODEL 0x10
#define GMIX_WRONG_PROB_TYPE 0x20
#define GMIX_ZERO_GAUSS 0x40
#define GMIX_WRONG_NPARS 0x80
#define GMIX_MISMATCH_SIZE 0x100


#ifndef isfinite
# define isfinite(x) ((x) * 0 == 0)
#endif

#define wlog(...) fprintf(stderr, __VA_ARGS__)

// when arg is too small or negative, return this
#define LOG_LOWVAL -9.999e9
#define LOG_MINARG 1.0e-10

# define M_TWO_PI   6.28318530717958647693

// this is for exp(-0.5*chi2) and corresponds to
// about sqrt(200) ~ 14 sigma
//#define EXP_MAX_CHI2 200
// 4 sigma
//#define EXP_MAX_CHI2 16
// 5 sigma
#define EXP_MAX_CHI2 25
// 6 sigma
//#define EXP_MAX_CHI2 36
// 8 sigma
//#define EXP_MAX_CHI2 64

#define GMIX_IMAGE_BIGNUM 9.999e9

//#define DEBUG
//#define DEBUG2

#ifdef DEBUG
 #define DBG if(1) 
#else
 #define DBG if(0) 
#endif
#ifdef DEBUG2
 #define DBG2 if(1) 
#else
 #define DBG2 if(0) 
#endif



#endif
