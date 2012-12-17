#ifndef _GMIX_DEFINITIONS_H
#define _GMIX_DEFINITIONS_H

#define GMIX_ERROR_NEGATIVE_DET 0x1
#define GMIX_ERROR_MAXIT 0x2
#define GMIX_ERROR_NEGATIVE_DET_COCENTER 0x4

#define wlog(...) fprintf(stderr, __VA_ARGS__)

# define M_TWO_PI   6.28318530717958647693

#define EXP_MAX_CHI2 200

#define DEBUG
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
