#include "olib.h"
#include "result.h"
#include "shapes_common_types.h"
#include <cuda_runtime.h>
#include <math.h>

static __device__ f32 deviceExpValue(f32 value) { return expf(value); }

static __device__ f64 deviceExpValue(f64 value) { return exp(value); }

static __device__ f32 deviceLogValue(f32 value) { return logf(value); }

static __device__ f64 deviceLogValue(f64 value) { return log(value); }

template <typename T>
__global__ static void
crossEntropyForwardKernel(const T *yGround, const T *logits, size_t rows,
                          size_t classCount, T *probs, T *loss) {
  size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  size_t base = row * classCount;
  T rowMax = logits[base];
  for (size_t c = 1; c < classCount; c++) {
    T candidate = logits[base + c];
    if (candidate > rowMax) {
      rowMax = candidate;
    }
  }

  T sumExp = (T)0;
  for (size_t c = 0; c < classCount; c++) {
    T expValue = deviceExpValue(logits[base + c] - rowMax);
    probs[base + c] = expValue;
    sumExp += expValue;
  }

  T rowLoss = (T)0;
  for (size_t c = 0; c < classCount; c++) {
    T probability = probs[base + c] / sumExp;
    probs[base + c] = probability;
    rowLoss += yGround[base + c] * deviceLogValue(probability);
  }

  atomicAdd(loss, (T)(-rowLoss / (T)rows));
}

template <typename T>
__global__ static void
crossEntropyBackwardKernel(const T *yGround, const T *probs, const T *gradOut,
                           size_t size, size_t rows, bool scalarGradOut,
                           T *dLogits) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  T gradScale = scalarGradOut ? gradOut[0] : gradOut[idx];
  T localGrad = (probs[idx] - yGround[idx]) / (T)rows;
  dLogits[idx] = localGrad * gradScale;
}

static Result finishCrossEntropyLaunch() {
  cudaError_t launchError = cudaGetLastError();
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

template <typename T>
static Result launchCrossEntropyForward(const void *yGround, const void *logits,
                                        size_t rows, size_t classCount,
                                        void *probs, void *loss) {
  if (rows == 0) {
    return OK;
  }

  int threadsPerBlock = 256;
  int blocks =
      (int)((rows + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  crossEntropyForwardKernel<<<blocks, threadsPerBlock>>>(
      (const T *)yGround, (const T *)logits, rows, classCount, (T *)probs,
      (T *)loss);
  return finishCrossEntropyLaunch();
}

template <typename T>
static Result launchCrossEntropyBackward(const void *yGround, const void *probs,
                                         const void *gradOut, size_t size,
                                         size_t rows, bool scalarGradOut,
                                         void *dLogits) {
  if (size == 0) {
    return OK;
  }

  int threadsPerBlock = 256;
  int blocks =
      (int)((size + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  crossEntropyBackwardKernel<<<blocks, threadsPerBlock>>>(
      (const T *)yGround, (const T *)probs, (const T *)gradOut, size, rows,
      scalarGradOut, (T *)dLogits);
  return finishCrossEntropyLaunch();
}

extern "C" Result shapescuda_CrossEntropyForward(shapes_Dtype dtype,
                                                 const void *yGround,
                                                 const void *logits,
                                                 size_t rows, size_t classCount,
                                                 void *probs, void *loss) {
  switch (dtype) {
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    return launchCrossEntropyForward<f32>(yGround, logits, rows, classCount,
                                          probs, loss);
  case F64:
    return launchCrossEntropyForward<f64>(yGround, logits, rows, classCount,
                                          probs, loss);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result
shapescuda_CrossEntropyBackward(shapes_Dtype dtype, const void *yGround,
                                const void *probs, const void *gradOut,
                                size_t rows, size_t classCount,
                                bool scalarGradOut, void *dLogits) {

  size_t size = rows * classCount;
  switch (dtype) {
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    return launchCrossEntropyBackward<f32>(yGround, probs, gradOut, size, rows,
                                           scalarGradOut, dLogits);
  case F64:
    return launchCrossEntropyBackward<f64>(yGround, probs, gradOut, size, rows,
                                           scalarGradOut, dLogits);
  default:
    return ERR_DTYPE_MISMATCH;
  }
}
