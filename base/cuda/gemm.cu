#include "../common.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stddef.h>

extern "C" void runCudaGemm(Context *ctx, Dtype dtype, cublasOperation_t transA,
                            cublasOperation_t transB, int m, int n, int k,
                            const void *a, int lda, const void *b, int ldb,
                            bool accumulate, void *c, int ldc) {
  if (dtype == F64) {
    f64 *dA;
    f64 *dB;
    f64 *dC;

    size_t aSize = (size_t)m * k * sizeof(f64);
    size_t bSize = (size_t)k * n * sizeof(f64);
    size_t cSize = (size_t)m * n * sizeof(f64);

    cudaMalloc(&dA, aSize);
    cudaMalloc(&dB, bSize);
    cudaMalloc(&dC, cSize);
    if (accumulate) {
      cudaMemcpy(dC, c, cSize, cudaMemcpyHostToDevice);
    }

    cudaMemcpy(dA, a, aSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b, bSize, cudaMemcpyHostToDevice);

    double alpha = 1.0;
    double beta = accumulate ? 1.0 : 0.0;

    cublasDgemm(ctx->handle, transA, transB, m, n, k, &alpha, dA, lda, dB, ldb,
                &beta, dC, ldc);
    cudaDeviceSynchronize();

    cudaMemcpy(c, dC, cSize, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return;
  }

  f32 *dA;
  f32 *dB;
  f32 *dC;

  size_t aSize = (size_t)m * k * sizeof(f32);
  size_t bSize = (size_t)k * n * sizeof(f32);
  size_t cSize = (size_t)m * n * sizeof(f32);

  cudaMalloc(&dA, aSize);
  cudaMalloc(&dB, bSize);
  cudaMalloc(&dC, cSize);
  if (accumulate) {
    cudaMemcpy(dC, c, cSize, cudaMemcpyHostToDevice);
  }

  cudaMemcpy(dA, a, aSize, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, b, bSize, cudaMemcpyHostToDevice);

  float alpha = 1.0f;
  float beta = accumulate ? 1.0f : 0.0f;

  cublasSgemm(ctx->handle, transA, transB, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc);
  cudaDeviceSynchronize();

  cudaMemcpy(c, dC, cSize, cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}
