#ifndef shapes_cuda_compat_h
#define shapes_cuda_compat_h

#ifdef SHAPES_HAS_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <driver_types.h>

#else // !SHAPES_HAS_CUDA

#include <stddef.h>
#include <stdint.h>

#if !defined(__CUDA_RUNTIME_H__)

// cudaError_t and constants
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1
#define cudaErrorMemoryAllocation 2

// cudaMemcpyKind
typedef int cudaMemcpyKind;
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 1
#define cudaMemcpyDeviceToDevice 2

// cudaPointerAttributes
struct cudaPointerAttributes {
  int type;
  int memoryType;
};

#define cudaMemoryTypeDevice 2

// CUDART_VERSION
#ifndef CUDART_VERSION
#define CUDART_VERSION 0
#endif

// Stub inline functions
static inline const char *cudaGetErrorString(cudaError_t err) {
  (void)err;
  return "CUDA support is not compiled in";
}

static inline cudaError_t cudaGetLastError(void) { return cudaSuccess; }

static inline cudaError_t cudaDeviceReset(void) { return cudaSuccess; }

static inline cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }

static inline cudaError_t cudaGetDeviceCount(int *count) {
  *count = 0;
  return cudaSuccess;
}

static inline cudaError_t cudaSetDevice(int device) {
  (void)device;
  return cudaSuccess;
}

static inline cudaError_t cudaFree(void *ptr) {
  (void)ptr;
  return cudaSuccess;
}

static inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                                     cudaMemcpyKind kind) {
  (void)dst;
  (void)src;
  (void)count;
  (void)kind;
  return cudaSuccess;
}

static inline cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  (void)devPtr;
  (void)value;
  (void)count;
  return cudaSuccess;
}

static inline cudaError_t cudaPointerGetAttributes(
    struct cudaPointerAttributes *attrs, const void *ptr) {
  (void)attrs;
  (void)ptr;
  return cudaErrorInvalidValue;
}

static inline cudaError_t cudaMalloc(void **ptr, size_t size) {
  (void)size;
  *ptr = NULL;
  return cudaErrorMemoryAllocation;
}

#endif // !__CUDA_RUNTIME_H__

#if !defined(CUBLAS_API_H_)

// cublas types
typedef void *cublasHandle_t;
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_STATUS_NOT_INITIALIZED 1

typedef int cublasOperation_t;
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1
#define CUBLAS_OP_C 2

static inline cublasStatus_t cublasCreate(cublasHandle_t *handle) {
  *handle = NULL;
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static inline cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  (void)handle;
  return CUBLAS_STATUS_SUCCESS;
}

#endif // !CUBLAS_API_H_

#endif // SHAPES_HAS_CUDA

#endif // shapes_cuda_compat_h
