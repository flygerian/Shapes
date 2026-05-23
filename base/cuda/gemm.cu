#include "../types.h"
#include "../utils_lib/utils_lib.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stddef.h>

extern "C" void runCudaGemm(Context *ctx, Dtype dtype, cublasOperation_t transA,
                            cublasOperation_t transB, int m, int n, int k,
                            const void *a, int lda, const void *b, int ldb,
                            bool accumulate, void *c, int ldc) {
  if (dtype == F64) {
    double alpha = 1.0;
    double beta = accumulate ? 1.0 : 0.0;

    cublasDgemm(ctx->handle, transA, transB, m, n, k, &alpha, (const double *)a,
                lda, (const double *)b, ldb, &beta, (double *)c, ldc);
    return;
  }

  float alpha = 1.0f;
  float beta = accumulate ? 1.0f : 0.0f;

  cublasSgemm(ctx->handle, transA, transB, m, n, k, &alpha, (const float *)a,
              lda, (const float *)b, ldb, &beta, (float *)c, ldc);
}
