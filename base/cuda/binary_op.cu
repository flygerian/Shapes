#include "../common.h"
#include "../result/result.h"
#include <cuda_runtime.h>
#include <stddef.h>

template <typename T>
__global__ static void binaryOpKernel(const T *a, const T *b, T *dest, size_t n,
                                      OpType opType) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  switch (opType) {
  case OP_ADD:
    dest[idx] = a[idx] + b[idx];
    break;
  case OP_SUBTRACT:
    dest[idx] = a[idx] - b[idx];
    break;
  case OP_MULTIPLY:
    dest[idx] = a[idx] * b[idx];
    break;
  default:
    break;
  }
}

template <typename T>
static Result launchBinaryOpKernel(const void *a, const void *b, void *dest,
                                   size_t n, OpType opType) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  binaryOpKernel<<<blocks, threadsPerBlock>>>((const T *)a, (const T *)b,
                                              (T *)dest, n, opType);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

extern "C" Result runCudaBinaryOp(Context *ctx, Dtype dtype, OpType opType,
                                  const void *a, const void *b, void *dest,
                                  tensor_size_t n) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  if (opType != OP_ADD && opType != OP_SUBTRACT && opType != OP_MULTIPLY) {
    return ERR_NOT_A_BINOP;
  }

  switch (dtype) {
  case BOOL:
    return launchBinaryOpKernel<bool>(a, b, dest, n, opType);
  case U8:
    return launchBinaryOpKernel<u8>(a, b, dest, n, opType);
  case U16:
    return launchBinaryOpKernel<u16>(a, b, dest, n, opType);
  case U32:
    return launchBinaryOpKernel<u32>(a, b, dest, n, opType);
  case U64:
    return launchBinaryOpKernel<u64>(a, b, dest, n, opType);
  case I8:
    return launchBinaryOpKernel<i8>(a, b, dest, n, opType);
  case I16:
    return launchBinaryOpKernel<i16>(a, b, dest, n, opType);
  case I32:
    return launchBinaryOpKernel<i32>(a, b, dest, n, opType);
  case I64:
    return launchBinaryOpKernel<i64>(a, b, dest, n, opType);
  case F16:
  case F32:
    return launchBinaryOpKernel<f32>(a, b, dest, n, opType);
  case F64:
    return launchBinaryOpKernel<f64>(a, b, dest, n, opType);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}
