#include "olib.h"
#include "result.h"
#include "types.h"
#include <cuda_runtime.h>
#include <stddef.h>

// ============================================================================
// Element-wise arithmetic kernels
// ============================================================================

template <typename T>
__global__ static void addKernel(const T *a, const T *b, T *dest, size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dest[idx] = a[idx] + b[idx];
}

template <typename T>
__global__ static void subtractKernel(const T *a, const T *b, T *dest,
                                      size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dest[idx] = a[idx] - b[idx];
}

template <typename T>
__global__ static void multiplyKernel(const T *a, const T *b, T *dest,
                                      size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dest[idx] = a[idx] * b[idx];
}

// ============================================================================
// Element-wise comparison kernels (output bool)
// ============================================================================

template <typename T>
__global__ static void greaterKernel(const T *a, const T *b, bool *dest,
                                     size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dest[idx] = a[idx] > b[idx];
}

template <typename T>
__global__ static void greaterOrEqualKernel(const T *a, const T *b, bool *dest,
                                            size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dest[idx] = a[idx] >= b[idx];
}

template <typename T>
__global__ static void lessKernel(const T *a, const T *b, bool *dest,
                                  size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dest[idx] = a[idx] < b[idx];
}

template <typename T>
__global__ static void lessOrEqualKernel(const T *a, const T *b, bool *dest,
                                         size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dest[idx] = a[idx] <= b[idx];
}

template <typename T>
__global__ static void equalKernel(const T *a, const T *b, bool *dest,
                                   size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dest[idx] = a[idx] == b[idx];
}

// ============================================================================
// Broadcast arithmetic kernels
// ============================================================================

template <typename T>
__global__ static void broadcastAddKernel(const T *larger, const T *smaller,
                                          T *dest, size_t broadcastDimSize,
                                          size_t innerDimSize) {
  size_t outer = blockIdx.x;
  size_t broadcast = blockIdx.y * blockDim.y + threadIdx.y;
  size_t inner = blockIdx.z * blockDim.x + threadIdx.x;

  if (broadcast >= broadcastDimSize || inner >= innerDimSize) {
    return;
  }

  size_t idx = outer * broadcastDimSize * innerDimSize +
               broadcast * innerDimSize + inner;
  dest[idx] = larger[idx] + smaller[outer * innerDimSize + inner];
}

template <typename T>
__global__ static void
broadcastSubtractKernel(const T *larger, const T *smaller, T *dest,
                        size_t broadcastDimSize, size_t innerDimSize) {
  size_t outer = blockIdx.x;
  size_t broadcast = blockIdx.y * blockDim.y + threadIdx.y;
  size_t inner = blockIdx.z * blockDim.x + threadIdx.x;

  if (broadcast >= broadcastDimSize || inner >= innerDimSize) {
    return;
  }

  size_t idx = outer * broadcastDimSize * innerDimSize +
               broadcast * innerDimSize + inner;
  dest[idx] = larger[idx] - smaller[outer * innerDimSize + inner];
}

template <typename T>
__global__ static void
broadcastMultiplyKernel(const T *larger, const T *smaller, T *dest,
                        size_t broadcastDimSize, size_t innerDimSize) {
  size_t outer = blockIdx.x;
  size_t broadcast = blockIdx.y * blockDim.y + threadIdx.y;
  size_t inner = blockIdx.z * blockDim.x + threadIdx.x;

  if (broadcast >= broadcastDimSize || inner >= innerDimSize) {
    return;
  }

  size_t idx = outer * broadcastDimSize * innerDimSize +
               broadcast * innerDimSize + inner;
  dest[idx] = larger[idx] * smaller[outer * innerDimSize + inner];
}

// ============================================================================
// Broadcast comparison kernels (output bool)
// ============================================================================

template <typename T>
__global__ static void
broadcastGreaterKernel(const T *larger, const T *smaller, bool *dest,
                       size_t broadcastDimSize, size_t innerDimSize) {
  size_t outer = blockIdx.x;
  size_t broadcast = blockIdx.y * blockDim.y + threadIdx.y;
  size_t inner = blockIdx.z * blockDim.x + threadIdx.x;

  if (broadcast >= broadcastDimSize || inner >= innerDimSize) {
    return;
  }

  size_t idx = outer * broadcastDimSize * innerDimSize +
               broadcast * innerDimSize + inner;
  dest[idx] = larger[idx] > smaller[outer * innerDimSize + inner];
}

template <typename T>
__global__ static void
broadcastGreaterOrEqualKernel(const T *larger, const T *smaller, bool *dest,
                              size_t broadcastDimSize, size_t innerDimSize) {
  size_t outer = blockIdx.x;
  size_t broadcast = blockIdx.y * blockDim.y + threadIdx.y;
  size_t inner = blockIdx.z * blockDim.x + threadIdx.x;

  if (broadcast >= broadcastDimSize || inner >= innerDimSize) {
    return;
  }

  size_t idx = outer * broadcastDimSize * innerDimSize +
               broadcast * innerDimSize + inner;
  dest[idx] = larger[idx] >= smaller[outer * innerDimSize + inner];
}

template <typename T>
__global__ static void broadcastLessKernel(const T *larger, const T *smaller,
                                           bool *dest, size_t broadcastDimSize,
                                           size_t innerDimSize) {
  size_t outer = blockIdx.x;
  size_t broadcast = blockIdx.y * blockDim.y + threadIdx.y;
  size_t inner = blockIdx.z * blockDim.x + threadIdx.x;

  if (broadcast >= broadcastDimSize || inner >= innerDimSize) {
    return;
  }

  size_t idx = outer * broadcastDimSize * innerDimSize +
               broadcast * innerDimSize + inner;
  dest[idx] = larger[idx] < smaller[outer * innerDimSize + inner];
}

template <typename T>
__global__ static void
broadcastLessOrEqualKernel(const T *larger, const T *smaller, bool *dest,
                           size_t broadcastDimSize, size_t innerDimSize) {
  size_t outer = blockIdx.x;
  size_t broadcast = blockIdx.y * blockDim.y + threadIdx.y;
  size_t inner = blockIdx.z * blockDim.x + threadIdx.x;

  if (broadcast >= broadcastDimSize || inner >= innerDimSize) {
    return;
  }

  size_t idx = outer * broadcastDimSize * innerDimSize +
               broadcast * innerDimSize + inner;
  dest[idx] = larger[idx] <= smaller[outer * innerDimSize + inner];
}

template <typename T>
__global__ static void broadcastEqualKernel(const T *larger, const T *smaller,
                                            bool *dest, size_t broadcastDimSize,
                                            size_t innerDimSize) {
  size_t outer = blockIdx.x;
  size_t broadcast = blockIdx.y * blockDim.y + threadIdx.y;
  size_t inner = blockIdx.z * blockDim.x + threadIdx.x;

  if (broadcast >= broadcastDimSize || inner >= innerDimSize) {
    return;
  }

  size_t idx = outer * broadcastDimSize * innerDimSize +
               broadcast * innerDimSize + inner;
  dest[idx] = larger[idx] == smaller[outer * innerDimSize + inner];
}

// ============================================================================
// Launch wrappers
// ============================================================================

static inline void checkCudaLaunch(cudaError_t err) {
  PANIC_WITH_MSG_IF(err != cudaSuccess, cudaGetErrorString(err));
}

template <typename T>
static Result launchAddKernel(Context *, const void *a, const void *b,
                              void *dest, tensor_size_t n) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  addKernel<<<blocks, threadsPerBlock>>>((const T *)a, (const T *)b, (T *)dest,
                                         (size_t)n);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result launchSubtractKernel(Context *, const void *a, const void *b,
                                   void *dest, tensor_size_t n) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  subtractKernel<<<blocks, threadsPerBlock>>>((const T *)a, (const T *)b,
                                              (T *)dest, (size_t)n);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result launchMultiplyKernel(Context *, const void *a, const void *b,
                                   void *dest, tensor_size_t n) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  multiplyKernel<<<blocks, threadsPerBlock>>>((const T *)a, (const T *)b,
                                              (T *)dest, (size_t)n);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result launchGreaterKernel(Context *, const void *a, const void *b,
                                  void *dest, tensor_size_t n) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  greaterKernel<<<blocks, threadsPerBlock>>>((const T *)a, (const T *)b,
                                             (bool *)dest, (size_t)n);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result launchGreaterOrEqualKernel(Context *, const void *a,
                                         const void *b, void *dest,
                                         tensor_size_t n) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  greaterOrEqualKernel<<<blocks, threadsPerBlock>>>((const T *)a, (const T *)b,
                                                    (bool *)dest, (size_t)n);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result launchLessKernel(Context *, const void *a, const void *b,
                               void *dest, tensor_size_t n) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  lessKernel<<<blocks, threadsPerBlock>>>((const T *)a, (const T *)b,
                                          (bool *)dest, (size_t)n);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result launchLessOrEqualKernel(Context *, const void *a, const void *b,
                                      void *dest, tensor_size_t n) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  lessOrEqualKernel<<<blocks, threadsPerBlock>>>((const T *)a, (const T *)b,
                                                 (bool *)dest, (size_t)n);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result launchEqualKernel(Context *, const void *a, const void *b,
                                void *dest, tensor_size_t n) {
  int threadsPerBlock = 256;
  int blocks =
      (int)((n + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  equalKernel<<<blocks, threadsPerBlock>>>((const T *)a, (const T *)b,
                                           (bool *)dest, (size_t)n);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result
launchBroadcastAddKernel(Context *, const void *larger, const void *smaller,
                         void *dest, size_t outerDimSize,
                         size_t broadcastDimSize, size_t innerDimSize) {
  dim3 block(32, 32, 1);
  dim3 grid(outerDimSize, (broadcastDimSize + block.y - 1) / block.y,
            (innerDimSize + block.x - 1) / block.x);
  broadcastAddKernel<<<grid, block>>>((const T *)larger, (const T *)smaller,
                                      (T *)dest, broadcastDimSize,
                                      innerDimSize);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result launchBroadcastSubtractKernel(Context *, const void *larger,
                                            const void *smaller, void *dest,
                                            size_t outerDimSize,
                                            size_t broadcastDimSize,
                                            size_t innerDimSize) {
  dim3 block(32, 32, 1);
  dim3 grid(outerDimSize, (broadcastDimSize + block.y - 1) / block.y,
            (innerDimSize + block.x - 1) / block.x);
  broadcastSubtractKernel<<<grid, block>>>((const T *)larger,
                                           (const T *)smaller, (T *)dest,
                                           broadcastDimSize, innerDimSize);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result launchBroadcastMultiplyKernel(Context *, const void *larger,
                                            const void *smaller, void *dest,
                                            size_t outerDimSize,
                                            size_t broadcastDimSize,
                                            size_t innerDimSize) {
  dim3 block(32, 32, 1);
  dim3 grid(outerDimSize, (broadcastDimSize + block.y - 1) / block.y,
            (innerDimSize + block.x - 1) / block.x);
  broadcastMultiplyKernel<<<grid, block>>>((const T *)larger,
                                           (const T *)smaller, (T *)dest,
                                           broadcastDimSize, innerDimSize);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result
launchBroadcastGreaterKernel(Context *, const void *larger, const void *smaller,
                             void *dest, size_t outerDimSize,
                             size_t broadcastDimSize, size_t innerDimSize) {
  dim3 block(32, 32, 1);
  dim3 grid(outerDimSize, (broadcastDimSize + block.y - 1) / block.y,
            (innerDimSize + block.x - 1) / block.x);
  broadcastGreaterKernel<<<grid, block>>>((const T *)larger, (const T *)smaller,
                                          (bool *)dest, broadcastDimSize,
                                          innerDimSize);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result launchBroadcastGreaterOrEqualKernel(
    Context *, const void *larger, const void *smaller, void *dest,
    size_t outerDimSize, size_t broadcastDimSize, size_t innerDimSize) {
  dim3 block(32, 32, 1);
  dim3 grid(outerDimSize, (broadcastDimSize + block.y - 1) / block.y,
            (innerDimSize + block.x - 1) / block.x);
  broadcastGreaterOrEqualKernel<<<grid, block>>>(
      (const T *)larger, (const T *)smaller, (bool *)dest, broadcastDimSize,
      innerDimSize);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result
launchBroadcastLessKernel(Context *, const void *larger, const void *smaller,
                          void *dest, size_t outerDimSize,
                          size_t broadcastDimSize, size_t innerDimSize) {
  dim3 block(32, 32, 1);
  dim3 grid(outerDimSize, (broadcastDimSize + block.y - 1) / block.y,
            (innerDimSize + block.x - 1) / block.x);
  broadcastLessKernel<<<grid, block>>>((const T *)larger, (const T *)smaller,
                                       (bool *)dest, broadcastDimSize,
                                       innerDimSize);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result launchBroadcastLessOrEqualKernel(Context *, const void *larger,
                                               const void *smaller, void *dest,
                                               size_t outerDimSize,
                                               size_t broadcastDimSize,
                                               size_t innerDimSize) {
  dim3 block(32, 32, 1);
  dim3 grid(outerDimSize, (broadcastDimSize + block.y - 1) / block.y,
            (innerDimSize + block.x - 1) / block.x);
  broadcastLessOrEqualKernel<<<grid, block>>>((const T *)larger,
                                              (const T *)smaller, (bool *)dest,
                                              broadcastDimSize, innerDimSize);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

template <typename T>
static Result
launchBroadcastEqualKernel(Context *, const void *larger, const void *smaller,
                           void *dest, size_t outerDimSize,
                           size_t broadcastDimSize, size_t innerDimSize) {
  dim3 block(32, 32, 1);
  dim3 grid(outerDimSize, (broadcastDimSize + block.y - 1) / block.y,
            (innerDimSize + block.x - 1) / block.x);
  broadcastEqualKernel<<<grid, block>>>((const T *)larger, (const T *)smaller,
                                        (bool *)dest, broadcastDimSize,
                                        innerDimSize);
  checkCudaLaunch(cudaGetLastError());
  return OK;
}

// ============================================================================
// Dispatch tables
// ============================================================================

// Dtype enum order: F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64, BOOL
// Indices:          0    1    2    3   4    5    6    7   8    9    10   11

typedef Result (*CudaBinaryOpFn)(Context *, const void *, const void *, void *,
                                 tensor_size_t);
typedef Result (*CudaBroadcastBinaryOpFn)(Context *, const void *, const void *,
                                          void *, size_t, size_t, size_t);

static const CudaBinaryOpFn addTable[12] = {
    NULL,                  // F16
    launchAddKernel<f32>,  // F32
    launchAddKernel<f64>,  // F64
    launchAddKernel<u8>,   // U8
    launchAddKernel<u16>,  // U16
    launchAddKernel<u32>,  // U32
    launchAddKernel<u64>,  // U64
    launchAddKernel<i8>,   // I8
    launchAddKernel<i16>,  // I16
    launchAddKernel<i32>,  // I32
    launchAddKernel<i64>,  // I64
    launchAddKernel<bool>, // BOOL
};

static const CudaBinaryOpFn subtractTable[12] = {
    NULL,                       // F16
    launchSubtractKernel<f32>,  // F32
    launchSubtractKernel<f64>,  // F64
    launchSubtractKernel<u8>,   // U8
    launchSubtractKernel<u16>,  // U16
    launchSubtractKernel<u32>,  // U32
    launchSubtractKernel<u64>,  // U64
    launchSubtractKernel<i8>,   // I8
    launchSubtractKernel<i16>,  // I16
    launchSubtractKernel<i32>,  // I32
    launchSubtractKernel<i64>,  // I64
    launchSubtractKernel<bool>, // BOOL
};

static const CudaBinaryOpFn multiplyTable[12] = {
    NULL,                       // F16
    launchMultiplyKernel<f32>,  // F32
    launchMultiplyKernel<f64>,  // F64
    launchMultiplyKernel<u8>,   // U8
    launchMultiplyKernel<u16>,  // U16
    launchMultiplyKernel<u32>,  // U32
    launchMultiplyKernel<u64>,  // U64
    launchMultiplyKernel<i8>,   // I8
    launchMultiplyKernel<i16>,  // I16
    launchMultiplyKernel<i32>,  // I32
    launchMultiplyKernel<i64>,  // I64
    launchMultiplyKernel<bool>, // BOOL
};

static const CudaBinaryOpFn greaterTable[12] = {
    NULL,                      // F16
    launchGreaterKernel<f32>,  // F32
    launchGreaterKernel<f64>,  // F64
    launchGreaterKernel<u8>,   // U8
    launchGreaterKernel<u16>,  // U16
    launchGreaterKernel<u32>,  // U32
    launchGreaterKernel<u64>,  // U64
    launchGreaterKernel<i8>,   // I8
    launchGreaterKernel<i16>,  // I16
    launchGreaterKernel<i32>,  // I32
    launchGreaterKernel<i64>,  // I64
    launchGreaterKernel<bool>, // BOOL
};

static const CudaBinaryOpFn greaterOrEqualTable[12] = {
    NULL,                             // F16
    launchGreaterOrEqualKernel<f32>,  // F32
    launchGreaterOrEqualKernel<f64>,  // F64
    launchGreaterOrEqualKernel<u8>,   // U8
    launchGreaterOrEqualKernel<u16>,  // U16
    launchGreaterOrEqualKernel<u32>,  // U32
    launchGreaterOrEqualKernel<u64>,  // U64
    launchGreaterOrEqualKernel<i8>,   // I8
    launchGreaterOrEqualKernel<i16>,  // I16
    launchGreaterOrEqualKernel<i32>,  // I32
    launchGreaterOrEqualKernel<i64>,  // I64
    launchGreaterOrEqualKernel<bool>, // BOOL
};

static const CudaBinaryOpFn lessTable[12] = {
    NULL,                   // F16
    launchLessKernel<f32>,  // F32
    launchLessKernel<f64>,  // F64
    launchLessKernel<u8>,   // U8
    launchLessKernel<u16>,  // U16
    launchLessKernel<u32>,  // U32
    launchLessKernel<u64>,  // U64
    launchLessKernel<i8>,   // I8
    launchLessKernel<i16>,  // I16
    launchLessKernel<i32>,  // I32
    launchLessKernel<i64>,  // I64
    launchLessKernel<bool>, // BOOL
};

static const CudaBinaryOpFn lessOrEqualTable[12] = {
    NULL,                          // F16
    launchLessOrEqualKernel<f32>,  // F32
    launchLessOrEqualKernel<f64>,  // F64
    launchLessOrEqualKernel<u8>,   // U8
    launchLessOrEqualKernel<u16>,  // U16
    launchLessOrEqualKernel<u32>,  // U32
    launchLessOrEqualKernel<u64>,  // U64
    launchLessOrEqualKernel<i8>,   // I8
    launchLessOrEqualKernel<i16>,  // I16
    launchLessOrEqualKernel<i32>,  // I32
    launchLessOrEqualKernel<i64>,  // I64
    launchLessOrEqualKernel<bool>, // BOOL
};

static const CudaBinaryOpFn equalTable[12] = {
    NULL,                    // F16
    launchEqualKernel<f32>,  // F32
    launchEqualKernel<f64>,  // F64
    launchEqualKernel<u8>,   // U8
    launchEqualKernel<u16>,  // U16
    launchEqualKernel<u32>,  // U32
    launchEqualKernel<u64>,  // U64
    launchEqualKernel<i8>,   // I8
    launchEqualKernel<i16>,  // I16
    launchEqualKernel<i32>,  // I32
    launchEqualKernel<i64>,  // I64
    launchEqualKernel<bool>, // BOOL
};

static const CudaBinaryOpFn *cudaBinaryOpTable[9] = {
    NULL,                // OP_NONE (0)
    addTable,            // OP_ADD (1)
    subtractTable,       // OP_SUBTRACT (2)
    multiplyTable,       // OP_MULTIPLY (3)
    greaterTable,        // OP_GREATER (4)
    greaterOrEqualTable, // OP_GREATER_OR_EQUAL (5)
    lessTable,           // OP_LESS (6)
    lessOrEqualTable,    // OP_LESS_OR_EQUAL (7)
    equalTable,          // OP_EQUAL (8)
};

static const CudaBroadcastBinaryOpFn broadcastAddTable[12] = {
    NULL,                           // F16
    launchBroadcastAddKernel<f32>,  // F32
    launchBroadcastAddKernel<f64>,  // F64
    launchBroadcastAddKernel<u8>,   // U8
    launchBroadcastAddKernel<u16>,  // U16
    launchBroadcastAddKernel<u32>,  // U32
    launchBroadcastAddKernel<u64>,  // U64
    launchBroadcastAddKernel<i8>,   // I8
    launchBroadcastAddKernel<i16>,  // I16
    launchBroadcastAddKernel<i32>,  // I32
    launchBroadcastAddKernel<i64>,  // I64
    launchBroadcastAddKernel<bool>, // BOOL
};

static const CudaBroadcastBinaryOpFn broadcastSubtractTable[12] = {
    NULL,                                // F16
    launchBroadcastSubtractKernel<f32>,  // F32
    launchBroadcastSubtractKernel<f64>,  // F64
    launchBroadcastSubtractKernel<u8>,   // U8
    launchBroadcastSubtractKernel<u16>,  // U16
    launchBroadcastSubtractKernel<u32>,  // U32
    launchBroadcastSubtractKernel<u64>,  // U64
    launchBroadcastSubtractKernel<i8>,   // I8
    launchBroadcastSubtractKernel<i16>,  // I16
    launchBroadcastSubtractKernel<i32>,  // I32
    launchBroadcastSubtractKernel<i64>,  // I64
    launchBroadcastSubtractKernel<bool>, // BOOL
};

static const CudaBroadcastBinaryOpFn broadcastMultiplyTable[12] = {
    NULL,                                // F16
    launchBroadcastMultiplyKernel<f32>,  // F32
    launchBroadcastMultiplyKernel<f64>,  // F64
    launchBroadcastMultiplyKernel<u8>,   // U8
    launchBroadcastMultiplyKernel<u16>,  // U16
    launchBroadcastMultiplyKernel<u32>,  // U32
    launchBroadcastMultiplyKernel<u64>,  // U64
    launchBroadcastMultiplyKernel<i8>,   // I8
    launchBroadcastMultiplyKernel<i16>,  // I16
    launchBroadcastMultiplyKernel<i32>,  // I32
    launchBroadcastMultiplyKernel<i64>,  // I64
    launchBroadcastMultiplyKernel<bool>, // BOOL
};

static const CudaBroadcastBinaryOpFn broadcastGreaterTable[12] = {
    NULL,                               // F16
    launchBroadcastGreaterKernel<f32>,  // F32
    launchBroadcastGreaterKernel<f64>,  // F64
    launchBroadcastGreaterKernel<u8>,   // U8
    launchBroadcastGreaterKernel<u16>,  // U16
    launchBroadcastGreaterKernel<u32>,  // U32
    launchBroadcastGreaterKernel<u64>,  // U64
    launchBroadcastGreaterKernel<i8>,   // I8
    launchBroadcastGreaterKernel<i16>,  // I16
    launchBroadcastGreaterKernel<i32>,  // I32
    launchBroadcastGreaterKernel<i64>,  // I64
    launchBroadcastGreaterKernel<bool>, // BOOL
};

static const CudaBroadcastBinaryOpFn broadcastGreaterOrEqualTable[12] = {
    NULL,                                      // F16
    launchBroadcastGreaterOrEqualKernel<f32>,  // F32
    launchBroadcastGreaterOrEqualKernel<f64>,  // F64
    launchBroadcastGreaterOrEqualKernel<u8>,   // U8
    launchBroadcastGreaterOrEqualKernel<u16>,  // U16
    launchBroadcastGreaterOrEqualKernel<u32>,  // U32
    launchBroadcastGreaterOrEqualKernel<u64>,  // U64
    launchBroadcastGreaterOrEqualKernel<i8>,   // I8
    launchBroadcastGreaterOrEqualKernel<i16>,  // I16
    launchBroadcastGreaterOrEqualKernel<i32>,  // I32
    launchBroadcastGreaterOrEqualKernel<i64>,  // I64
    launchBroadcastGreaterOrEqualKernel<bool>, // BOOL
};

static const CudaBroadcastBinaryOpFn broadcastLessTable[12] = {
    NULL,                            // F16
    launchBroadcastLessKernel<f32>,  // F32
    launchBroadcastLessKernel<f64>,  // F64
    launchBroadcastLessKernel<u8>,   // U8
    launchBroadcastLessKernel<u16>,  // U16
    launchBroadcastLessKernel<u32>,  // U32
    launchBroadcastLessKernel<u64>,  // U64
    launchBroadcastLessKernel<i8>,   // I8
    launchBroadcastLessKernel<i16>,  // I16
    launchBroadcastLessKernel<i32>,  // I32
    launchBroadcastLessKernel<i64>,  // I64
    launchBroadcastLessKernel<bool>, // BOOL
};

static const CudaBroadcastBinaryOpFn broadcastLessOrEqualTable[12] = {
    NULL,                                   // F16
    launchBroadcastLessOrEqualKernel<f32>,  // F32
    launchBroadcastLessOrEqualKernel<f64>,  // F64
    launchBroadcastLessOrEqualKernel<u8>,   // U8
    launchBroadcastLessOrEqualKernel<u16>,  // U16
    launchBroadcastLessOrEqualKernel<u32>,  // U32
    launchBroadcastLessOrEqualKernel<u64>,  // U64
    launchBroadcastLessOrEqualKernel<i8>,   // I8
    launchBroadcastLessOrEqualKernel<i16>,  // I16
    launchBroadcastLessOrEqualKernel<i32>,  // I32
    launchBroadcastLessOrEqualKernel<i64>,  // I64
    launchBroadcastLessOrEqualKernel<bool>, // BOOL
};

static const CudaBroadcastBinaryOpFn broadcastEqualTable[12] = {
    NULL,                             // F16
    launchBroadcastEqualKernel<f32>,  // F32
    launchBroadcastEqualKernel<f64>,  // F64
    launchBroadcastEqualKernel<u8>,   // U8
    launchBroadcastEqualKernel<u16>,  // U16
    launchBroadcastEqualKernel<u32>,  // U32
    launchBroadcastEqualKernel<u64>,  // U64
    launchBroadcastEqualKernel<i8>,   // I8
    launchBroadcastEqualKernel<i16>,  // I16
    launchBroadcastEqualKernel<i32>,  // I32
    launchBroadcastEqualKernel<i64>,  // I64
    launchBroadcastEqualKernel<bool>, // BOOL
};

static const CudaBroadcastBinaryOpFn *cudaBroadcastBinaryOpTable[9] = {
    NULL,                         // OP_NONE (0)
    broadcastAddTable,            // OP_ADD (1)
    broadcastSubtractTable,       // OP_SUBTRACT (2)
    broadcastMultiplyTable,       // OP_MULTIPLY (3)
    broadcastGreaterTable,        // OP_GREATER (4)
    broadcastGreaterOrEqualTable, // OP_GREATER_OR_EQUAL (5)
    broadcastLessTable,           // OP_LESS (6)
    broadcastLessOrEqualTable,    // OP_LESS_OR_EQUAL (7)
    broadcastEqualTable,          // OP_EQUAL (8)
};

// ============================================================================
// Public API
// ============================================================================

extern "C" Result
runCudaBroadcastBinaryOp(Context *ctx, Dtype dtype, OpType opType, void *larger,
                         void *smaller, void *dest, size_t outerDimSize,
                         size_t broadcastDimSize, size_t innerDimSize) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  if (opType < OP_ADD || opType > OP_EQUAL) {
    return ERR_NOT_A_BINOP;
  }

  const CudaBroadcastBinaryOpFn *table = cudaBroadcastBinaryOpTable[opType];
  if (table == NULL) {
    return ERR_NOT_A_BINOP;
  }

  if (dtype < F16 || dtype > BOOL) {
    return ERR_DTYPE_MISMATCH;
  }

  CudaBroadcastBinaryOpFn fn = table[dtype];
  if (fn == NULL) {
    return ERR_DTYPE_MISMATCH;
  }

  return fn(ctx, larger, smaller, dest, outerDimSize, broadcastDimSize,
            innerDimSize);
}

extern "C" Result runCudaBinaryOp(Context *ctx, Dtype dtype, OpType opType,
                                  const void *a, const void *b, void *dest,
                                  tensor_size_t n) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  if (opType < OP_ADD || opType > OP_EQUAL) {
    return ERR_NOT_A_BINOP;
  }

  const CudaBinaryOpFn *table = cudaBinaryOpTable[opType];
  if (table == NULL) {
    return ERR_NOT_A_BINOP;
  }

  if (dtype < F16 || dtype > BOOL) {
    return ERR_DTYPE_MISMATCH;
  }

  CudaBinaryOpFn fn = table[dtype];
  if (fn == NULL) {
    return ERR_DTYPE_MISMATCH;
  }

  return fn(ctx, a, b, dest, n);
}
