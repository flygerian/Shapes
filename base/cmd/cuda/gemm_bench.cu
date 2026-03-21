#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double now_ms() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

int main() {
  int M = 4096, N = 4096, K = 4096;

  float *h_A = (float *)malloc(M * K * sizeof(float));
  float *h_B = (float *)malloc(K * N * sizeof(float));
  float *h_C = (float *)malloc(M * N * sizeof(float));

  for (int i = 0; i < M * K; i++)
    h_A[i] = (float)rand() / RAND_MAX;
  for (int i = 0; i < K * N; i++)
    h_B[i] = (float)rand() / RAND_MAX;

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(float));
  cudaMalloc(&d_B, K * N * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));

  cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f, beta = 0.0f;

  // warmup
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K,
              &beta, d_C, M);
  cudaDeviceSynchronize();

  double start = now_ms();
  for (int i = 0; i < 10; i++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B,
                K, &beta, d_C, M);
  }
  cudaDeviceSynchronize();
  double elapsed = now_ms() - start;

  printf("cuBLAS: %.2f ms/iter\n", elapsed / 10);

  // CPU comparison with a naive loop would be unfair, but you can compare
  // against your existing cblas_sgemm timing on the same size

  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(h_A);
  free(h_B);
  free(h_C);
  return 0;
}
