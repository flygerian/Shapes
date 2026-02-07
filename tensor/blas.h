#ifndef shapes_blas_h
#define shapes_blas_h

#include "cblas.h"

#define BLAS_GEMM(dt, A, B, C, m, n, k)                                                            \
  do {                                                                                             \
    switch (dt) {                                                                                  \
      case F64:                                                                                    \
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, (double *)(A), k,     \
                    (double *)(B), n, 0.0, (double *)(C), n);                                      \
        break;                                                                                     \
      case F32:                                                                                    \
      case F16:                                                                                    \
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, (float *)(A), k,     \
                    (float *)(B), n, 0.0f, (float *)(C), n);                                       \
        break;                                                                                     \
      default: break;                                                                              \
    }                                                                                              \
  } while (0)

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
