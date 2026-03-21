#include "../common.h"
#include "tensor_internal.h"
#include <sched.h>

static cublasOperation_t toCudaTranspose(CBLAS_TRANSPOSE trans) {
  switch (trans) {
  case CblasNoTrans:
    return CUBLAS_OP_N;
  case CblasTrans:
    return CUBLAS_OP_T;
  default:
    return CUBLAS_OP_C;
  }
}

void runCpuGemm(Dtype dtype, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n, int k,
                const void *a, int lda, const void *b, int ldb, bool accumulate, void *c,
                int ldc) {

  if (dtype == F64) {
    cblas_dgemm(CblasRowMajor, transA, transB, m, n, k, 1.0, (const double *)a, lda,
                (const double *)b, ldb, accumulate ? 1.0 : 0.0, (double *)c, ldc);
    return;
  }

  cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, 1.0f, (const float *)a, lda, (const float *)b,
              ldb, accumulate ? 1.0f : 0.0f, (float *)c, ldc);
}

void runGemm(Context *ctx, Dtype dtype, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int m, int n,
             int k, const void *a, int lda, const void *b, int ldb, bool accumulate, void *c,
             int ldc) {

  if (ctx->device == NULL) {
    return runCpuGemm(dtype, transA, transB, m, n, k, a, lda, b, ldb, accumulate, c, ldc);
  }

  switch (ctx->device->type) {
  case CPU:
    return runCpuGemm(dtype, transA, transB, m, n, k, a, lda, b, ldb, accumulate, c, ldc);
    break;

  case CUDA:

    // cuBLAS assumes column-major storage; swapping operands/op flags makes it
    // compute the same result as the row-major CBLAS entry points used elsewhere.
    return runCudaGemm(ctx, dtype, toCudaTranspose(transB), toCudaTranspose(transA), n, m, k, b,
                       ldb, a, lda, accumulate, c, ldc);

  default:
    return runCpuGemm(dtype, transA, transB, m, n, k, a, lda, b, ldb, accumulate, c, ldc);
  }
}
