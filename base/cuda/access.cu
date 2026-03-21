#include "../common.h"
#include <cuda_runtime.h>
#include <stddef.h>

template <typename IndexType>
__device__ static dim_t loadGatherIndex(const IndexType *indices, size_t idx) {
  return (dim_t)indices[idx];
}

template <typename ValueType, typename IndexType>
__global__ static void indexSelect1dKernel(const ValueType *src, const IndexType *indices,
                                           ValueType *dest, size_t numIndices,
                                           size_t sliceSize) {
  size_t flatIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = numIndices * sliceSize;
  if (flatIdx >= total) {
    return;
  }

  size_t indexPos = flatIdx / sliceSize;
  size_t sliceOffset = flatIdx % sliceSize;
  dim_t sourceIndex = loadGatherIndex(indices, indexPos);
  size_t srcOffset = (size_t)sourceIndex * sliceSize + sliceOffset;
  dest[flatIdx] = src[srcOffset];
}

template <typename ValueType, typename RowIndexType, typename ColIndexType>
__global__ static void indexSelect2dKernel(const ValueType *src, size_t sourceDim1,
                                           const RowIndexType *rowIndices,
                                           const ColIndexType *colIndices, ValueType *dest,
                                           size_t numIndices, size_t sliceSize) {
  size_t flatIdx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = numIndices * sliceSize;
  if (flatIdx >= total) {
    return;
  }

  size_t indexPos = flatIdx / sliceSize;
  size_t sliceOffset = flatIdx % sliceSize;
  dim_t row = loadGatherIndex(rowIndices, indexPos);
  dim_t col = loadGatherIndex(colIndices, indexPos);
  size_t srcOffset = ((size_t)row * sourceDim1 + (size_t)col) * sliceSize + sliceOffset;
  dest[flatIdx] = src[srcOffset];
}

template <typename ValueType, typename IndexType>
static Result launchIndexSelect1d(const void *src, const void *indices, void *dest,
                                  size_t numIndices, size_t sliceSize) {
  int threadsPerBlock = 256;
  size_t total = numIndices * sliceSize;
  int blocks = (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  indexSelect1dKernel<<<blocks, threadsPerBlock>>>((const ValueType *)src, (const IndexType *)indices,
                                                   (ValueType *)dest, numIndices, sliceSize);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

template <typename ValueType, typename RowIndexType, typename ColIndexType>
static Result launchIndexSelect2d(const void *src, size_t sourceDim1, const void *rowIndices,
                                  const void *colIndices, void *dest, size_t numIndices,
                                  size_t sliceSize) {
  int threadsPerBlock = 256;
  size_t total = numIndices * sliceSize;
  int blocks = (int)((total + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  indexSelect2dKernel<<<blocks, threadsPerBlock>>>((const ValueType *)src, sourceDim1,
                                                   (const RowIndexType *)rowIndices,
                                                   (const ColIndexType *)colIndices,
                                                   (ValueType *)dest, numIndices, sliceSize);

  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess) {
    return ERR_NO_OP;
  }

  return OK;
}

template <typename IndexType>
static Result dispatchIndexSelect1dByValueDtype(Dtype valueDtype, const void *src,
                                                const void *indices, void *dest,
                                                size_t numIndices, size_t sliceSize) {
  switch (valueDtype) {
    case BOOL: return launchIndexSelect1d<bool, IndexType>(src, indices, dest, numIndices, sliceSize);
    case U8: return launchIndexSelect1d<u8, IndexType>(src, indices, dest, numIndices, sliceSize);
    case U16: return launchIndexSelect1d<u16, IndexType>(src, indices, dest, numIndices, sliceSize);
    case U32: return launchIndexSelect1d<u32, IndexType>(src, indices, dest, numIndices, sliceSize);
    case U64: return launchIndexSelect1d<u64, IndexType>(src, indices, dest, numIndices, sliceSize);
    case I8: return launchIndexSelect1d<i8, IndexType>(src, indices, dest, numIndices, sliceSize);
    case I16: return launchIndexSelect1d<i16, IndexType>(src, indices, dest, numIndices, sliceSize);
    case I32: return launchIndexSelect1d<i32, IndexType>(src, indices, dest, numIndices, sliceSize);
    case I64: return launchIndexSelect1d<i64, IndexType>(src, indices, dest, numIndices, sliceSize);
    case F16:
    case F32: return launchIndexSelect1d<f32, IndexType>(src, indices, dest, numIndices, sliceSize);
    case F64: return launchIndexSelect1d<f64, IndexType>(src, indices, dest, numIndices, sliceSize);
    default: return ERR_DTYPE_MISMATCH;
  }
}

template <typename RowIndexType, typename ColIndexType>
static Result dispatchIndexSelect2dByValueDtype(Dtype valueDtype, const void *src,
                                                size_t sourceDim1, const void *rowIndices,
                                                const void *colIndices, void *dest,
                                                size_t numIndices, size_t sliceSize) {
  switch (valueDtype) {
    case BOOL:
      return launchIndexSelect2d<bool, RowIndexType, ColIndexType>(src, sourceDim1, rowIndices,
                                                                   colIndices, dest, numIndices,
                                                                   sliceSize);
    case U8:
      return launchIndexSelect2d<u8, RowIndexType, ColIndexType>(src, sourceDim1, rowIndices,
                                                                 colIndices, dest, numIndices,
                                                                 sliceSize);
    case U16:
      return launchIndexSelect2d<u16, RowIndexType, ColIndexType>(src, sourceDim1, rowIndices,
                                                                  colIndices, dest, numIndices,
                                                                  sliceSize);
    case U32:
      return launchIndexSelect2d<u32, RowIndexType, ColIndexType>(src, sourceDim1, rowIndices,
                                                                  colIndices, dest, numIndices,
                                                                  sliceSize);
    case U64:
      return launchIndexSelect2d<u64, RowIndexType, ColIndexType>(src, sourceDim1, rowIndices,
                                                                  colIndices, dest, numIndices,
                                                                  sliceSize);
    case I8:
      return launchIndexSelect2d<i8, RowIndexType, ColIndexType>(src, sourceDim1, rowIndices,
                                                                 colIndices, dest, numIndices,
                                                                 sliceSize);
    case I16:
      return launchIndexSelect2d<i16, RowIndexType, ColIndexType>(src, sourceDim1, rowIndices,
                                                                  colIndices, dest, numIndices,
                                                                  sliceSize);
    case I32:
      return launchIndexSelect2d<i32, RowIndexType, ColIndexType>(src, sourceDim1, rowIndices,
                                                                  colIndices, dest, numIndices,
                                                                  sliceSize);
    case I64:
      return launchIndexSelect2d<i64, RowIndexType, ColIndexType>(src, sourceDim1, rowIndices,
                                                                  colIndices, dest, numIndices,
                                                                  sliceSize);
    case F16:
    case F32:
      return launchIndexSelect2d<f32, RowIndexType, ColIndexType>(src, sourceDim1, rowIndices,
                                                                  colIndices, dest, numIndices,
                                                                  sliceSize);
    case F64:
      return launchIndexSelect2d<f64, RowIndexType, ColIndexType>(src, sourceDim1, rowIndices,
                                                                  colIndices, dest, numIndices,
                                                                  sliceSize);
    default: return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result runCudaIndexSelect1d(Context *ctx, Dtype dtype, const void *src,
                                       const void *indices, Dtype indexDtype, void *dest,
                                       tensor_size_t numIndices, tensor_size_t sliceSize) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (indexDtype) {
    case U8: return dispatchIndexSelect1dByValueDtype<u8>(dtype, src, indices, dest, numIndices, sliceSize);
    case U16: return dispatchIndexSelect1dByValueDtype<u16>(dtype, src, indices, dest, numIndices, sliceSize);
    case U32: return dispatchIndexSelect1dByValueDtype<u32>(dtype, src, indices, dest, numIndices, sliceSize);
    case U64: return dispatchIndexSelect1dByValueDtype<u64>(dtype, src, indices, dest, numIndices, sliceSize);
    case I8: return dispatchIndexSelect1dByValueDtype<i8>(dtype, src, indices, dest, numIndices, sliceSize);
    case I16: return dispatchIndexSelect1dByValueDtype<i16>(dtype, src, indices, dest, numIndices, sliceSize);
    case I32: return dispatchIndexSelect1dByValueDtype<i32>(dtype, src, indices, dest, numIndices, sliceSize);
    case I64: return dispatchIndexSelect1dByValueDtype<i64>(dtype, src, indices, dest, numIndices, sliceSize);
    default: return ERR_DTYPE_MISMATCH;
  }
}

extern "C" Result runCudaIndexSelect2d(Context *ctx, Dtype dtype, const void *src,
                                       dim_t sourceDim1, const void *rowIndices,
                                       Dtype rowIndexDtype, const void *colIndices,
                                       Dtype colIndexDtype, void *dest, tensor_size_t numIndices,
                                       tensor_size_t sliceSize) {
  if (ctx == NULL || ctx->device == NULL || ctx->device->type != CUDA) {
    return ERR_NO_OP;
  }

  switch (rowIndexDtype) {
    case U8:
      switch (colIndexDtype) {
        case U8:
          return dispatchIndexSelect2dByValueDtype<u8, u8>(dtype, src, sourceDim1, rowIndices,
                                                           colIndices, dest, numIndices, sliceSize);
        case U16:
          return dispatchIndexSelect2dByValueDtype<u8, u16>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case U32:
          return dispatchIndexSelect2dByValueDtype<u8, u32>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case U64:
          return dispatchIndexSelect2dByValueDtype<u8, u64>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case I8:
          return dispatchIndexSelect2dByValueDtype<u8, i8>(dtype, src, sourceDim1, rowIndices,
                                                           colIndices, dest, numIndices, sliceSize);
        case I16:
          return dispatchIndexSelect2dByValueDtype<u8, i16>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case I32:
          return dispatchIndexSelect2dByValueDtype<u8, i32>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case I64:
          return dispatchIndexSelect2dByValueDtype<u8, i64>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        default: return ERR_DTYPE_MISMATCH;
      }
    case U16:
      switch (colIndexDtype) {
        case U8:
          return dispatchIndexSelect2dByValueDtype<u16, u8>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case U16:
          return dispatchIndexSelect2dByValueDtype<u16, u16>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case U32:
          return dispatchIndexSelect2dByValueDtype<u16, u32>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case U64:
          return dispatchIndexSelect2dByValueDtype<u16, u64>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I8:
          return dispatchIndexSelect2dByValueDtype<u16, i8>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case I16:
          return dispatchIndexSelect2dByValueDtype<u16, i16>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I32:
          return dispatchIndexSelect2dByValueDtype<u16, i32>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I64:
          return dispatchIndexSelect2dByValueDtype<u16, i64>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        default: return ERR_DTYPE_MISMATCH;
      }
    case U32:
      switch (colIndexDtype) {
        case U8:
          return dispatchIndexSelect2dByValueDtype<u32, u8>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case U16:
          return dispatchIndexSelect2dByValueDtype<u32, u16>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case U32:
          return dispatchIndexSelect2dByValueDtype<u32, u32>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case U64:
          return dispatchIndexSelect2dByValueDtype<u32, u64>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I8:
          return dispatchIndexSelect2dByValueDtype<u32, i8>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case I16:
          return dispatchIndexSelect2dByValueDtype<u32, i16>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I32:
          return dispatchIndexSelect2dByValueDtype<u32, i32>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I64:
          return dispatchIndexSelect2dByValueDtype<u32, i64>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        default: return ERR_DTYPE_MISMATCH;
      }
    case U64:
      switch (colIndexDtype) {
        case U8:
          return dispatchIndexSelect2dByValueDtype<u64, u8>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case U16:
          return dispatchIndexSelect2dByValueDtype<u64, u16>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case U32:
          return dispatchIndexSelect2dByValueDtype<u64, u32>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case U64:
          return dispatchIndexSelect2dByValueDtype<u64, u64>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I8:
          return dispatchIndexSelect2dByValueDtype<u64, i8>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case I16:
          return dispatchIndexSelect2dByValueDtype<u64, i16>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I32:
          return dispatchIndexSelect2dByValueDtype<u64, i32>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I64:
          return dispatchIndexSelect2dByValueDtype<u64, i64>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        default: return ERR_DTYPE_MISMATCH;
      }
    case I8:
      switch (colIndexDtype) {
        case U8:
          return dispatchIndexSelect2dByValueDtype<i8, u8>(dtype, src, sourceDim1, rowIndices,
                                                           colIndices, dest, numIndices, sliceSize);
        case U16:
          return dispatchIndexSelect2dByValueDtype<i8, u16>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case U32:
          return dispatchIndexSelect2dByValueDtype<i8, u32>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case U64:
          return dispatchIndexSelect2dByValueDtype<i8, u64>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case I8:
          return dispatchIndexSelect2dByValueDtype<i8, i8>(dtype, src, sourceDim1, rowIndices,
                                                           colIndices, dest, numIndices, sliceSize);
        case I16:
          return dispatchIndexSelect2dByValueDtype<i8, i16>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case I32:
          return dispatchIndexSelect2dByValueDtype<i8, i32>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case I64:
          return dispatchIndexSelect2dByValueDtype<i8, i64>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        default: return ERR_DTYPE_MISMATCH;
      }
    case I16:
      switch (colIndexDtype) {
        case U8:
          return dispatchIndexSelect2dByValueDtype<i16, u8>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case U16:
          return dispatchIndexSelect2dByValueDtype<i16, u16>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case U32:
          return dispatchIndexSelect2dByValueDtype<i16, u32>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case U64:
          return dispatchIndexSelect2dByValueDtype<i16, u64>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I8:
          return dispatchIndexSelect2dByValueDtype<i16, i8>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case I16:
          return dispatchIndexSelect2dByValueDtype<i16, i16>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I32:
          return dispatchIndexSelect2dByValueDtype<i16, i32>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I64:
          return dispatchIndexSelect2dByValueDtype<i16, i64>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        default: return ERR_DTYPE_MISMATCH;
      }
    case I32:
      switch (colIndexDtype) {
        case U8:
          return dispatchIndexSelect2dByValueDtype<i32, u8>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case U16:
          return dispatchIndexSelect2dByValueDtype<i32, u16>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case U32:
          return dispatchIndexSelect2dByValueDtype<i32, u32>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case U64:
          return dispatchIndexSelect2dByValueDtype<i32, u64>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I8:
          return dispatchIndexSelect2dByValueDtype<i32, i8>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case I16:
          return dispatchIndexSelect2dByValueDtype<i32, i16>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I32:
          return dispatchIndexSelect2dByValueDtype<i32, i32>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I64:
          return dispatchIndexSelect2dByValueDtype<i32, i64>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        default: return ERR_DTYPE_MISMATCH;
      }
    case I64:
      switch (colIndexDtype) {
        case U8:
          return dispatchIndexSelect2dByValueDtype<i64, u8>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case U16:
          return dispatchIndexSelect2dByValueDtype<i64, u16>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case U32:
          return dispatchIndexSelect2dByValueDtype<i64, u32>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case U64:
          return dispatchIndexSelect2dByValueDtype<i64, u64>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I8:
          return dispatchIndexSelect2dByValueDtype<i64, i8>(dtype, src, sourceDim1, rowIndices,
                                                            colIndices, dest, numIndices, sliceSize);
        case I16:
          return dispatchIndexSelect2dByValueDtype<i64, i16>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I32:
          return dispatchIndexSelect2dByValueDtype<i64, i32>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        case I64:
          return dispatchIndexSelect2dByValueDtype<i64, i64>(dtype, src, sourceDim1, rowIndices,
                                                             colIndices, dest, numIndices, sliceSize);
        default: return ERR_DTYPE_MISMATCH;
      }
    default: return ERR_DTYPE_MISMATCH;
  }
}
