#include "olib.h"
#include "result.h"
#include "shapes_common_types.h"
#include <cuda_runtime.h>
#include <stddef.h>

template <typename Dest, typename Src>
__global__ static void castKernel(const Src *src, Dest *dest, size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  dest[idx] = (Dest)src[idx];
}

template <typename Src, typename Dest>
static Result launchCastKernel(const void *src, void *dest, size_t n) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  castKernel<<<blocks, threadsPerBlock>>>((const Src *)src, (Dest *)dest, n);

  cudaError_t launchError = cudaGetLastError();
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

template <typename Src>
static Result dispatchCudaCastTarget(shapes_Dtype targetDtype, const void *src,
                                     void *dest, size_t n) {
  switch (targetDtype) {
  case BOOL:
    return launchCastKernel<Src, bool>(src, dest, n);
  case U8:
    return launchCastKernel<Src, u8>(src, dest, n);
  case U16:
    return launchCastKernel<Src, u16>(src, dest, n);
  case U32:
    return launchCastKernel<Src, u32>(src, dest, n);
  case U64:
    return launchCastKernel<Src, u64>(src, dest, n);
  case I8:
    return launchCastKernel<Src, i8>(src, dest, n);
  case I16:
    return launchCastKernel<Src, i16>(src, dest, n);
  case I32:
    return launchCastKernel<Src, i32>(src, dest, n);
  case I64:
    return launchCastKernel<Src, i64>(src, dest, n);
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    return launchCastKernel<Src, f32>(src, dest, n);
  case F64:
    return launchCastKernel<Src, f64>(src, dest, n);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result runCudaCast(shapes_Dtype sourceDtype, const void *src,
                              shapes_Dtype targetDtype, void *dest, size_t n) {
  switch (sourceDtype) {
  case BOOL:
    return dispatchCudaCastTarget<bool>(targetDtype, src, dest, n);
  case U8:
    return dispatchCudaCastTarget<u8>(targetDtype, src, dest, n);
  case U16:
    return dispatchCudaCastTarget<u16>(targetDtype, src, dest, n);
  case U32:
    return dispatchCudaCastTarget<u32>(targetDtype, src, dest, n);
  case U64:
    return dispatchCudaCastTarget<u64>(targetDtype, src, dest, n);
  case I8:
    return dispatchCudaCastTarget<i8>(targetDtype, src, dest, n);
  case I16:
    return dispatchCudaCastTarget<i16>(targetDtype, src, dest, n);
  case I32:
    return dispatchCudaCastTarget<i32>(targetDtype, src, dest, n);
  case I64:
    return dispatchCudaCastTarget<i64>(targetDtype, src, dest, n);
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    return dispatchCudaCastTarget<f32>(targetDtype, src, dest, n);
  case F64:
    return dispatchCudaCastTarget<f64>(targetDtype, src, dest, n);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}
