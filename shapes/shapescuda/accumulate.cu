#include "olib.h"
#include "result.h"
#include "shapes_common_types.h"
#include <cuda_runtime.h>
#include <stddef.h>

template <typename T>
__device__ static void atomicAccumulate(T *address, T value);

template <>
__device__ void atomicAccumulate<float>(float *address, float value) {
  atomicAdd(address, value);
}

template <>
__device__ void atomicAccumulate<double>(double *address, double value) {
  atomicAdd(address, value);
}

template <typename IndexType>
__device__ static size_t loadIndex(const IndexType *indices, size_t idx) {
  return (size_t)indices[idx];
}

template <typename ValueType, typename IndexType>
__global__ static void
indexAccumulate1dKernel(ValueType *dest, const IndexType *indices,
                        const ValueType *srcGrad, size_t numIndices,
                        size_t sliceSize) {
  size_t flatIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = numIndices * sliceSize;
  if (flatIdx >= total) {
    return;
  }

  size_t indexPos = flatIdx / sliceSize;
  size_t sliceOffset = flatIdx % sliceSize;
  size_t targetIndex = loadIndex(indices, indexPos);
  size_t destOffset = (size_t)targetIndex * sliceSize + sliceOffset;
  atomicAccumulate(&dest[destOffset], srcGrad[flatIdx]);
}

template <typename ValueType, typename RowIndexType, typename ColIndexType>
__global__ static void indexAccumulate2dKernel(ValueType *dest, size_t destDim1,
                                               const RowIndexType *rowIndices,
                                               const ColIndexType *colIndices,
                                               const ValueType *srcGrad,
                                               size_t numIndices,
                                               size_t sliceSize) {
  size_t flatIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = numIndices * sliceSize;
  if (flatIdx >= total) {
    return;
  }

  size_t indexPos = flatIdx / sliceSize;
  size_t sliceOffset = flatIdx % sliceSize;
  size_t row = loadIndex(rowIndices, indexPos);
  size_t col = loadIndex(colIndices, indexPos);
  size_t destOffset =
      ((size_t)row * destDim1 + (size_t)col) * sliceSize + sliceOffset;
  atomicAccumulate(&dest[destOffset], srcGrad[flatIdx]);
}

template <typename ValueType>
__global__ static void
sliceAccumulateKernel(ValueType *dest, size_t destNumDims,
                      const size_t *destMultipliers, const shapes_Range *ranges,
                      const ValueType *srcGrad, const size_t *srcDims,
                      size_t srcNumDims, size_t srcSize) {
  size_t flatIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (flatIdx >= srcSize) {
    return;
  }

  size_t remaining = flatIdx;
  size_t destOffset = 0;
  for (size_t d = 0; d < srcNumDims; d++) {
    size_t srcCoord = remaining % srcDims[d];
    remaining /= srcDims[d];
    destOffset += (srcCoord + ranges[d].start) * destMultipliers[d];
  }

  atomicAccumulate(&dest[destOffset], srcGrad[flatIdx]);
}

template <typename ValueType, typename IndexType>
static Result launchIndexAccumulate1d(const void *dest, const void *indices,
                                      const void *srcGrad, size_t numIndices,
                                      size_t sliceSize) {
  int threadsPerBlock = 256;
  size_t total = numIndices * sliceSize;
  int blocks =
      (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  indexAccumulate1dKernel<<<blocks, threadsPerBlock>>>(
      (ValueType *)dest, (const IndexType *)indices, (const ValueType *)srcGrad,
      numIndices, sliceSize);

  cudaError_t launchError = cudaGetLastError();
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

template <typename ValueType, typename RowIndexType, typename ColIndexType>
static Result launchIndexAccumulate2d(const void *dest, size_t destDim1,
                                      const void *rowIndices,
                                      const void *colIndices,
                                      const void *srcGrad, size_t numIndices,
                                      size_t sliceSize) {
  int threadsPerBlock = 256;
  size_t total = numIndices * sliceSize;
  int blocks =
      (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  indexAccumulate2dKernel<<<blocks, threadsPerBlock>>>(
      (ValueType *)dest, destDim1, (const RowIndexType *)rowIndices,
      (const ColIndexType *)colIndices, (const ValueType *)srcGrad, numIndices,
      sliceSize);

  cudaError_t launchError = cudaGetLastError();
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

template <typename ValueType>
static Result launchSliceAccumulate(const void *dest, size_t destNumDims,
                                    const size_t *destMultipliers,
                                    const shapes_Range *ranges,
                                    const void *srcGrad, const size_t *srcDims,
                                    size_t srcNumDims, size_t srcSize) {
  size_t *deviceMultipliers = NULL;
  shapes_Range *deviceRanges = NULL;
  size_t *deviceSrcDims = NULL;

  cudaError_t allocError =
      cudaMalloc(&deviceMultipliers, sizeof(size_t) * destNumDims);
  PANIC_WITH_MSG_IF(allocError != cudaSuccess, cudaGetErrorString(allocError));
  allocError = cudaMalloc(&deviceRanges, sizeof(shapes_Range) * srcNumDims);
  PANIC_WITH_MSG_IF(allocError != cudaSuccess, cudaGetErrorString(allocError));
  allocError = cudaMalloc(&deviceSrcDims, sizeof(size_t) * srcNumDims);
  PANIC_WITH_MSG_IF(allocError != cudaSuccess, cudaGetErrorString(allocError));

  cudaError_t memcpyErr =
      cudaMemcpy(deviceMultipliers, destMultipliers,
                 sizeof(size_t) * destNumDims, cudaMemcpyHostToDevice);
  PANIC_WITH_MSG_IF(memcpyErr != cudaSuccess, cudaGetErrorString(memcpyErr));
  memcpyErr =
      cudaMemcpy(deviceRanges, ranges, sizeof(shapes_Range) * srcNumDims,
                 cudaMemcpyHostToDevice);
  PANIC_WITH_MSG_IF(memcpyErr != cudaSuccess, cudaGetErrorString(memcpyErr));
  memcpyErr = cudaMemcpy(deviceSrcDims, srcDims, sizeof(size_t) * srcNumDims,
                         cudaMemcpyHostToDevice);
  PANIC_WITH_MSG_IF(memcpyErr != cudaSuccess, cudaGetErrorString(memcpyErr));

  int threadsPerBlock = 256;
  int blocks =
      (int)((srcSize + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  sliceAccumulateKernel<<<blocks, threadsPerBlock>>>(
      (ValueType *)dest, destNumDims, deviceMultipliers, deviceRanges,
      (const ValueType *)srcGrad, deviceSrcDims, srcNumDims, srcSize);

  cudaError_t launchError = cudaGetLastError();
  cudaFree(deviceMultipliers);
  cudaFree(deviceRanges);
  cudaFree(deviceSrcDims);
  PANIC_WITH_MSG_IF(launchError != cudaSuccess,
                    cudaGetErrorString(launchError));

  return OK;
}

static Result dispatchIndexDtype1d(shapes_Dtype valueDtype, const void *dest,
                                   const void *indices, shapes_Dtype indexDtype,
                                   const void *srcGrad, size_t numIndices,
                                   size_t sliceSize) {
  switch (valueDtype) {
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    switch (indexDtype) {
    case U8:
      return launchIndexAccumulate1d<float, u8>(dest, indices, srcGrad,
                                                numIndices, sliceSize);
    case U16:
      return launchIndexAccumulate1d<float, u16>(dest, indices, srcGrad,
                                                 numIndices, sliceSize);
    case U32:
      return launchIndexAccumulate1d<float, u32>(dest, indices, srcGrad,
                                                 numIndices, sliceSize);
    case U64:
      return launchIndexAccumulate1d<float, u64>(dest, indices, srcGrad,
                                                 numIndices, sliceSize);
    case I8:
      return launchIndexAccumulate1d<float, i8>(dest, indices, srcGrad,
                                                numIndices, sliceSize);
    case I16:
      return launchIndexAccumulate1d<float, i16>(dest, indices, srcGrad,
                                                 numIndices, sliceSize);
    case I32:
      return launchIndexAccumulate1d<float, i32>(dest, indices, srcGrad,
                                                 numIndices, sliceSize);
    case I64:
      return launchIndexAccumulate1d<float, i64>(dest, indices, srcGrad,
                                                 numIndices, sliceSize);
    default:
      return ERR_NO_OP;
    }
  case F64:
    switch (indexDtype) {
    case U8:
      return launchIndexAccumulate1d<double, u8>(dest, indices, srcGrad,
                                                 numIndices, sliceSize);
    case U16:
      return launchIndexAccumulate1d<double, u16>(dest, indices, srcGrad,
                                                  numIndices, sliceSize);
    case U32:
      return launchIndexAccumulate1d<double, u32>(dest, indices, srcGrad,
                                                  numIndices, sliceSize);
    case U64:
      return launchIndexAccumulate1d<double, u64>(dest, indices, srcGrad,
                                                  numIndices, sliceSize);
    case I8:
      return launchIndexAccumulate1d<double, i8>(dest, indices, srcGrad,
                                                 numIndices, sliceSize);
    case I16:
      return launchIndexAccumulate1d<double, i16>(dest, indices, srcGrad,
                                                  numIndices, sliceSize);
    case I32:
      return launchIndexAccumulate1d<double, i32>(dest, indices, srcGrad,
                                                  numIndices, sliceSize);
    case I64:
      return launchIndexAccumulate1d<double, i64>(dest, indices, srcGrad,
                                                  numIndices, sliceSize);
    default:
      return ERR_NO_OP;
    }
  default:
    return ERR_NO_OP;
  }
}

static Result
dispatchIndexDtype2d(shapes_Dtype valueDtype, const void *dest, size_t destDim1,
                     const void *rowIndices, shapes_Dtype rowIndexDtype,
                     const void *colIndices, shapes_Dtype colIndexDtype,
                     const void *srcGrad, size_t numIndices, size_t sliceSize) {
  if (rowIndexDtype != colIndexDtype) {
    return ERR_DTYPE_MISMATCH;
  }

  switch (valueDtype) {
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    switch (rowIndexDtype) {
    case U8:
      return launchIndexAccumulate2d<float, u8, u8>(dest, destDim1, rowIndices,
                                                    colIndices, srcGrad,
                                                    numIndices, sliceSize);
    case U16:
      return launchIndexAccumulate2d<float, u16, u16>(
          dest, destDim1, rowIndices, colIndices, srcGrad, numIndices,
          sliceSize);
    case U32:
      return launchIndexAccumulate2d<float, u32, u32>(
          dest, destDim1, rowIndices, colIndices, srcGrad, numIndices,
          sliceSize);
    case U64:
      return launchIndexAccumulate2d<float, u64, u64>(
          dest, destDim1, rowIndices, colIndices, srcGrad, numIndices,
          sliceSize);
    case I8:
      return launchIndexAccumulate2d<float, i8, i8>(dest, destDim1, rowIndices,
                                                    colIndices, srcGrad,
                                                    numIndices, sliceSize);
    case I16:
      return launchIndexAccumulate2d<float, i16, i16>(
          dest, destDim1, rowIndices, colIndices, srcGrad, numIndices,
          sliceSize);
    case I32:
      return launchIndexAccumulate2d<float, i32, i32>(
          dest, destDim1, rowIndices, colIndices, srcGrad, numIndices,
          sliceSize);
    case I64:
      return launchIndexAccumulate2d<float, i64, i64>(
          dest, destDim1, rowIndices, colIndices, srcGrad, numIndices,
          sliceSize);
    default:
      return ERR_NO_OP;
    }
  case F64:
    switch (rowIndexDtype) {
    case U8:
      return launchIndexAccumulate2d<double, u8, u8>(dest, destDim1, rowIndices,
                                                     colIndices, srcGrad,
                                                     numIndices, sliceSize);
    case U16:
      return launchIndexAccumulate2d<double, u16, u16>(
          dest, destDim1, rowIndices, colIndices, srcGrad, numIndices,
          sliceSize);
    case U32:
      return launchIndexAccumulate2d<double, u32, u32>(
          dest, destDim1, rowIndices, colIndices, srcGrad, numIndices,
          sliceSize);
    case U64:
      return launchIndexAccumulate2d<double, u64, u64>(
          dest, destDim1, rowIndices, colIndices, srcGrad, numIndices,
          sliceSize);
    case I8:
      return launchIndexAccumulate2d<double, i8, i8>(dest, destDim1, rowIndices,
                                                     colIndices, srcGrad,
                                                     numIndices, sliceSize);
    case I16:
      return launchIndexAccumulate2d<double, i16, i16>(
          dest, destDim1, rowIndices, colIndices, srcGrad, numIndices,
          sliceSize);
    case I32:
      return launchIndexAccumulate2d<double, i32, i32>(
          dest, destDim1, rowIndices, colIndices, srcGrad, numIndices,
          sliceSize);
    case I64:
      return launchIndexAccumulate2d<double, i64, i64>(
          dest, destDim1, rowIndices, colIndices, srcGrad, numIndices,
          sliceSize);
    default:
      return ERR_NO_OP;
    }
  default:
    return ERR_NO_OP;
  }
}

extern "C" Result shapescuda_IndexAccumulate1d(shapes_Dtype dtype, void *dest,
                                               const void *indices,
                                               shapes_Dtype indexDtype,
                                               const void *srcGrad,
                                               size_t numIndices,
                                               size_t sliceSize) {
  return dispatchIndexDtype1d(dtype, dest, indices, indexDtype, srcGrad,
                              numIndices, sliceSize);
}

extern "C" Result
shapescuda_IndexAccumulate2d(shapes_Dtype dtype, void *dest, size_t destDim1,
                             const void *rowIndices, shapes_Dtype rowIndexDtype,
                             const void *colIndices, shapes_Dtype colIndexDtype,
                             const void *srcGrad, size_t numIndices,
                             size_t sliceSize) {

  return dispatchIndexDtype2d(dtype, dest, destDim1, rowIndices, rowIndexDtype,
                              colIndices, colIndexDtype, srcGrad, numIndices,
                              sliceSize);
}

extern "C" Result
shapescuda_SliceAccumulate(shapes_Dtype dtype, void *dest, size_t destNumDims,
                           const size_t *destMultipliers,
                           const shapes_Range *ranges, const void *srcGrad,
                           const size_t *srcDims, size_t srcNumDims,
                           size_t srcSize) {
  switch (dtype) {
  case F16:
    return ERR_DTYPE_MISMATCH;
  case F32:
    return launchSliceAccumulate<float>(dest, destNumDims, destMultipliers,
                                        ranges, srcGrad, srcDims, srcNumDims,
                                        srcSize);
  case F64:
    return launchSliceAccumulate<double>(dest, destNumDims, destMultipliers,
                                         ranges, srcGrad, srcDims, srcNumDims,
                                         srcSize);
  default:
    return ERR_NO_OP;
  }
}
