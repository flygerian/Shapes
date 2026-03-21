#ifndef shapes_blas_h
#define shapes_blas_h

#include "cblas.h"

#define BLAS_DOT(dt, n, A, B, result)                                                              \
  do {                                                                                             \
    switch (dt) {                                                                                  \
      case F64: *((double *)(result)) = cblas_ddot(n, (double *)(A), 1, (double *)(B), 1); break;  \
      case F32:                                                                                    \
      case F16: *((float *)(result)) = cblas_sdot(n, (float *)(A), 1, (float *)(B), 1); break;     \
      default: break;                                                                              \
    }                                                                                              \
  } while (0)

#endif
