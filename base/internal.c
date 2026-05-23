#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "shapes.h"
#include "nn/nn.h"
#include "result/result.h"
#include "types.h"
#include "tensor_internal.h"
#include "utils_lib/array.h"
#include "utils_lib/memory.h"
#include "utils_lib/cuda_memory.h"
#include "value.h"
#include <stdlib.h>

size_t getBytesForDtype(Dtype type) {
  switch (type) {
    case BOOL: return sizeof(bool);
    case U8: return sizeof(u8);
    case U16: return sizeof(u16);
    case U32: return sizeof(u32);
    case U64: return sizeof(u64);
    case I8: return sizeof(i8);
    case I16: return sizeof(i16);
    case I32: return sizeof(i32);
    case I64: return sizeof(i64);
    case F16:
    case F32: return sizeof(float);
    case F64: return sizeof(double);
    default: return 0;
  }
}

void PrintItem(Tensor *t) {
  dim_t zero[1] = {0};
  Dim zeroIdx = {.dims = zero, .numOfDims = 1};

  Value *val = GetAt(t, zeroIdx);

  PRINT_VALUE(*val);
}

char *GetItem(Context *ctx, Tensor *t) {
  dim_t zero[1] = {0};
  Dim zeroIdx = {.dims = zero, .numOfDims = 1};

  Value *val = GetAt(t, zeroIdx);

  size_t size = sizeof(char) * 32;
  char *valueStr = allocate(ctx->memory, size);
  VALUE_TO_STRING(*val, valueStr, size);

  return valueStr;
}

Result CopyShape(Tensor *t, dim_t *destDims, u8 *numDims) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (numDims == NULL) {
    return ERR_NULL_PTR;
  }

  *numDims = t->shape.numOfDims;
  if (destDims == NULL || t->shape.numOfDims == 0) {
    return OK;
  }

  memcpy(destDims, t->shape.dims, sizeof(dim_t) * t->shape.numOfDims);
  return OK;
}

sizeAndMultipliers calculateSizeAndMultipliers(Context *ctx, dim_t *dims, u8 numOfDims) {
  PANIC_IF(ctx == NULL, NULL_CONTEXT);

  if (dims == NULL || numOfDims == 0) {
    return (sizeAndMultipliers){.size = 1, .multipliers = NULL};
  }

  multiplier_t *multipliers = allocate(ctx->memory, sizeof(multiplier_t) * numOfDims);
  tensor_size_t size = 1;

  for (int x = numOfDims - 1; x >= 0; x--) {
    multipliers[x] = size;
    size *= dims[x];
  }

  return (sizeAndMultipliers){.multipliers = multipliers, .size = size};
}

bool isSameContext(Context *a, Context *b) {
  return a == b;
}

Result clearTensorValues(Tensor *t) {
  if (t == NULL || t->values == NULL) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  size_t valueBytes = t->size * getBytesForDtype(t->dtype);
  if (t->context == NULL || t->context->device == NULL || t->context->device->type == CPU) {
    memset(t->values, 0, valueBytes);
    return OK;
  }

  cudaError_t clearResult = cudaMemset(t->values, 0, valueBytes);
  PANIC_WITH_MSG_IF(clearResult != cudaSuccess, cudaGetErrorString(clearResult));

  return OK;
}

u64 getContigousIdxFromCoord(Tensor *restrict t, dim_t *restrict idx) {
  u64 result = 0;

  for (u8 x = 0; x < t->shape.numOfDims; x++) {
    u64 coord = idx[x];
    if (t->isView && t->boundary) {
      coord += t->boundary[x].start;
    }
    result += coord * t->shape.multipliers[x];
  }

  return result;
}

Result readTensorValueAtFlatIndex(Tensor *t, u64 idx, Value *result) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (result == NULL) {
    return ERR_NULL_PTR;
  }

  if (!t->isView && idx >= t->size) {
    return ERR_OUT_OF_BOUNDS;
  }

  result->dtype = t->dtype;
  size_t valueBytes = getBytesForDtype(t->dtype);
  memcpy(&result->as, (char *)t->values + idx * valueBytes, valueBytes);
  return OK;
}

Result writeTensorValueAtFlatIndex(Tensor *t, u64 idx, Value value) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (t->dtype != value.dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (!t->isView && idx >= t->size) {
    return ERR_OUT_OF_BOUNDS;
  }

  size_t valueBytes = getBytesForDtype(t->dtype);
  memcpy((char *)t->values + idx * valueBytes, &value.as, valueBytes);
  return OK;
}

Tensor *copyToContiguous(Context *ctx, Tensor *source) {
  Tensor *copy = t_Zeros(ctx, source->shape, source->dtype);
  if (copy == NULL) {
    return NULL;
  }

  size_t *indices = allocate(ctx->memory, sizeof(size_t) * source->shape.numOfDims);
  if (indices == NULL) {
    return NULL;
  }
  memset(indices, 0, sizeof(size_t) * source->shape.numOfDims);

  for (tensor_size_t i = 0; i < source->size; i++) {
    Dim idx = {.dims = indices, .numOfDims = source->shape.numOfDims};
    Value *val = GetAt(source, idx);
    writeTensorValueAtFlatIndex(copy, i, *val);

    for (int d = source->shape.numOfDims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < source->shape.dims[d]) {
        break;
      }
      indices[d] = 0;
    }
  }

  copy->isContigousCopy = true;

  return copy;
}

Tensor *materializeTensorOnContext(Context *ctx, Tensor *src) {
  PANIC_IF(ctx == NULL || src == NULL, ERR_NULL_TENSOR_PROVIDED);

  // Check if tensors are on the same device (both NULL = CPU, or same device pointer)
  bool sameDevice = (ctx->device == NULL && src->context->device == NULL) ||
                    (ctx->device != NULL && src->context->device != NULL && ctx->device->type == src->context->device->type);
  PANIC_IF(!sameDevice, ERR_DIFFERENT_CTX_TENSORS_PASSED);

  Tensor *working = src;
  if (!src->isContigous) {
    Context *materializeCtx = src->context != NULL ? src->context : ctx;
    working = copyToContiguous(materializeCtx, src);
    PANIC_IF(working == NULL, ALLOCATION_FAILED);
  }

  return working;
}

bool areBroadcastable(Tensor *a, Tensor *b) {
  u8 maxDims = a->shape.numOfDims > b->shape.numOfDims ? a->shape.numOfDims : b->shape.numOfDims;

  bool pastBroadcastRegion = false;
  for (int d = 0; d < maxDims; d++) {
    int aIdx = a->shape.numOfDims - 1 - d;
    int bIdx = b->shape.numOfDims - 1 - d;

    dim_t aDim = aIdx >= 0 ? a->shape.dims[aIdx] : 1;
    dim_t bDim = bIdx >= 0 ? b->shape.dims[bIdx] : 1;

    if (aDim == bDim) {
      pastBroadcastRegion = true;
      continue;
    }

    if (aDim != 1 && bDim != 1) {
      return false;
    }

    if (pastBroadcastRegion) {
      return false;
    }
  }

  return true;
}

TensorPair padSmallerTensor(Context *ctx, Tensor *a, Tensor *b) {
  Tensor *smaller = a->shape.numOfDims < b->shape.numOfDims ? a : b;
  Tensor *larger = a->shape.numOfDims < b->shape.numOfDims ? b : a;
  u8 diff = larger->shape.numOfDims - smaller->shape.numOfDims;

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * larger->shape.numOfDims);
  if (newDims == NULL) {
    return (TensorPair){.a = a, .b = b};
  }
  for (u8 i = 0; i < diff; i++) {
    newDims[i] = 1;
  }
  for (u8 i = 0; i < smaller->shape.numOfDims; i++) {
    newDims[diff + i] = smaller->shape.dims[i];
  }

  Dim newShape = {.dims = newDims, .numOfDims = larger->shape.numOfDims};
  Tensor *reshapedSmaller = Reshape(ctx, smaller, newShape);

  if (smaller == a) {
    return (TensorPair){.a = reshapedSmaller, .b = b};
  }
  return (TensorPair){.a = a, .b = reshapedSmaller};
}

Result calculateNumElementsBeforeDim(Tensor *t, dim_t dim, tensor_size_t *result) {
  if (dim == 0) {
    *result = 1;
    return OK;
  }

  if (dim >= t->shape.numOfDims) {
    return ERR_OUT_OF_BOUNDS;
  }

  tensor_size_t totalElements = t->shape.dims[0];
  for (dim_t d = 1; d < t->shape.numOfDims; d++) {
    if (d == dim) {
      break;
    }
    totalElements *= t->shape.dims[d];
  }

  *result = totalElements;
  return OK;
}

Result getDimsBefore(Context *ctx, Tensor *t, dim_t dim, Dim *result) {
  (void)ctx;
  if (dim == 0) {
    return OK;
  }

  if (dim >= t->shape.numOfDims) {
    return ERR_OUT_OF_BOUNDS;
  }

  for (dim_t d = 0; d < dim; d++) {
    result->dims[d] = t->shape.dims[d];
  }

  return OK;
}

Result calculateNumElementsAfterDim(Tensor *t, dim_t dim, tensor_size_t *result) {
  if (dim == t->shape.numOfDims - 1) {
    *result = 1;
    return OK;
  }

  if (dim >= t->size) {
    return ERR_OUT_OF_BOUNDS;
  }

  tensor_size_t numElements = t->shape.dims[dim + 1];
  for (dim_t d = dim + 2; d < t->shape.numOfDims; d++) {
    if (d >= t->shape.numOfDims) {
      break;
    }
    numElements *= t->shape.dims[d];
  }

  *result = numElements;
  return OK;
}

void accumulateStridedByDtype(Dtype dtype, void *destValues, u64 destBase, u64 destStep, void *srcValues, u64 srcBase, u64 srcStep, u64 count) {
  switch (dtype) {
    case BOOL: {
      bool *d = (bool *)destValues;
      bool *s = (bool *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] = d[destBase + i * destStep] || s[srcBase + i * srcStep];
      }
      break;
    }
    case U8: {
      u8 *d = (u8 *)destValues;
      u8 *s = (u8 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case U16: {
      u16 *d = (u16 *)destValues;
      u16 *s = (u16 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case U32: {
      u32 *d = (u32 *)destValues;
      u32 *s = (u32 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case U64: {
      u64 *d = (u64 *)destValues;
      u64 *s = (u64 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case I8: {
      i8 *d = (i8 *)destValues;
      i8 *s = (i8 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case I16: {
      i16 *d = (i16 *)destValues;
      i16 *s = (i16 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case I32: {
      i32 *d = (i32 *)destValues;
      i32 *s = (i32 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case I64: {
      i64 *d = (i64 *)destValues;
      i64 *s = (i64 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case F16: {
      f16 *d = (f16 *)destValues;
      f16 *s = (f16 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case F32: {
      f32 *d = (f32 *)destValues;
      f32 *s = (f32 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
    case F64: {
      f64 *d = (f64 *)destValues;
      f64 *s = (f64 *)srcValues;
      for (u64 i = 0; i < count; i++) {
        d[destBase + i * destStep] += s[srcBase + i * srcStep];
      }
      break;
    }
  }
}

Array *shapes_Make_DynamicTensorArray(Memory *memory) {
  return MakeDynamicArray(memory, sizeof(Tensor *));
}

Array *shapes_Make_TensorArray(Memory *memory, size_t capacity) {
  return MakeArray(memory, sizeof(Tensor *), capacity);
}

void shapes_Array_AppendTensor(Array *array, Tensor *tensor) {
  PANIC_IF(array->elemSize != sizeof(Tensor *), ARRAY_ELEM_SIZE_MISMATCH);
  Array_Append(array, (void *)&tensor);
}

void shapes_Array_AppendTensorArray(Array *array, Array *tensorArray) {
  PANIC_IF(array == NULL, ERR_NULL_PTR);

  for (size_t i = 0; i < tensorArray->size; i++) {
    Tensor *tensor = shapes_Array_TensorIdx(tensorArray, i);
    PANIC_IF(tensor == NULL, ERR_NULL_TENSOR_PROVIDED);

    shapes_Array_AppendTensor(array, tensor);
  }
}

void shapes_Array_AppendLayer(Array *array, Layer *layer) {
  Array_Append(array, (void *)&layer);
}

Layer *shapes_Array_LayerIdx(Array *array, size_t idx) {
  return *(Layer **)Array_Idx(array, idx);
}

static void printTensorDim(Tensor *t, size_t *flatIdx, u8 dim, u8 indent) {
  printf("[");
  for (dim_t i = 0; i < t->shape.dims[dim]; i++) {
    if (i > 0) {
      if (dim == t->shape.numOfDims - 1) {
        printf(", ");
      } else {
        printf(",\n");
        for (u8 s = 0; s < indent + 1; s++) {
          printf(" ");
        }
      }
    }

    if (dim == t->shape.numOfDims - 1) {
      Value val;
      (void)readTensorValueAtFlatIndex(t, (*flatIdx)++, &val);
      switch (t->dtype) {
        case BOOL: printf("%s", val.as.boolean ? "true" : "false"); break;
        case U8: printf("%u", (unsigned int)val.as.u8); break;
        case U16: printf("%u", (unsigned int)val.as.u16); break;
        case U32: printf("%u", (unsigned int)val.as.u32); break;
        case U64: printf("%llu", (unsigned long long)val.as.u64); break;
        case I8: printf("%d", (int)val.as.i8); break;
        case I16: printf("%d", (int)val.as.i16); break;
        case I32: printf("%d", (int)val.as.i32); break;
        case I64: printf("%lld", (long long)val.as.i64); break;
        case F16: printf("%.4f", (double)val.as.f16); break;
        case F32: printf("%.4f", (double)val.as.f32); break;
        case F64: printf("%.4f", (double)val.as.f64); break;
      }
    } else {
      printTensorDim(t, flatIdx, dim + 1, indent + 1);
    }
  }
  printf("]");
}

void PrintTensor(Tensor *tensor) {
  if (tensor == NULL || tensor->values == NULL) {
    printf("tensor([])\n");
    return;
  }

  if (tensor->shape.numOfDims == 0) {
    printf("tensor(");
    Value val;
    (void)readTensorValueAtFlatIndex(tensor, 0, &val);
    switch (tensor->dtype) {
      case BOOL: printf("%s", val.as.boolean ? "true" : "false"); break;
      case U8: printf("%u", (unsigned int)val.as.u8); break;
      case U16: printf("%u", (unsigned int)val.as.u16); break;
      case U32: printf("%u", (unsigned int)val.as.u32); break;
      case U64: printf("%llu", (unsigned long long)val.as.u64); break;
      case I8: printf("%d", (int)val.as.i8); break;
      case I16: printf("%d", (int)val.as.i16); break;
      case I32: printf("%d", (int)val.as.i32); break;
      case I64: printf("%lld", (long long)val.as.i64); break;
      case F16: printf("%.4f", (double)val.as.f16); break;
      case F32: printf("%.4f", (double)val.as.f32); break;
      case F64: printf("%.4f", (double)val.as.f64); break;
    }
    printf(")\n");
    return;
  }

  printf("tensor<");
  for (u8 d = 0; d < tensor->shape.numOfDims; d++) {
    printf("%zu", (size_t)tensor->shape.dims[d]);
    if (d < tensor->shape.numOfDims - 1) {
      printf(",");
    }
  }
  printf(">(\n");

  printf("    ");
  size_t flatIdx = 0;
  printTensorDim(tensor, &flatIdx, 0, 4);
  printf("\n)\n");
}

void moveTensor(Context *destCtx, Tensor *t) {
  size_t valueBytes = t->size * getBytesForDtype(t->dtype);
  CudaBlock block = AllocateOnCuda(&destCtx->cudaMemory, destCtx->cudaMetadataMemory, valueBytes);
  PANIC_IF(block.ptr == NULL, ALLOCATION_FAILED);
  void *locationOnDest = block.ptr;
  Result copyResult = shapes_CopyBetweenDevices(t->context->device->type, destCtx->device->type, t->values, locationOnDest, valueBytes);

  PANIC_IF(copyResult != OK, ALLOCATION_FAILED);

  t->context = destCtx;
  t->values = locationOnDest;
}

void MoveTensorToHost(Context *destCtx, Tensor *t) {
  size_t valueBytes = t->size * getBytesForDtype(t->dtype);
  void *locationOnDest = allocate(destCtx->memory, valueBytes);
  PANIC_IF(locationOnDest == NULL, ALLOCATION_FAILED);

  Result copyResult =
      shapes_CopyBetweenDevices(t->context->device->type, destCtx->device->type,
                         t->values, locationOnDest, valueBytes);
  PANIC_IF(copyResult != OK, ALLOCATION_FAILED);

  t->context = destCtx;
  t->values = locationOnDest;
}

void MoveToCuda(Context *destCtx, Array *tensors) {
  PANIC_IF(destCtx == NULL, ERR_COPY_CTX_DEVICE_IS_NULL);
  PANIC_IF(destCtx->device == NULL || destCtx->device->type != CUDA,
           ERR_COPY_CTX_DEVICE_IS_NULL);
  PANIC_IF(tensors == NULL, ERR_NULL_PTR);

  for (size_t x = 0; x < tensors->size; x++) {
    Tensor *t = shapes_Array_TensorIdx(tensors, x);
    PANIC_IF(t == NULL || t->context == NULL, ERR_NULL_TENSOR_PROVIDED);
    PANIC_IF(!t->isContigous, NON_CONTIGOUS_MOVE_TENSOR);

    moveTensor(destCtx, t);
    moveTensor(destCtx, t->grad);
  }
}

void MoveToHost(Context *destCtx, Array *tensors) {
  PANIC_IF(destCtx == NULL, ERR_COPY_CTX_DEVICE_IS_NULL);
  PANIC_IF(destCtx->device != NULL && destCtx->device->type != CPU,
           ERR_COPY_CTX_DEVICE_IS_NULL);
  PANIC_IF(tensors == NULL, ERR_NULL_PTR);

  for (size_t x = 0; x < tensors->size; x++) {
    Tensor *t = shapes_Array_TensorIdx(tensors, x);
    PANIC_IF(t == NULL || t->context == NULL, ERR_NULL_TENSOR_PROVIDED);
    PANIC_IF(!t->isContigous, NON_CONTIGOUS_MOVE_TENSOR);

    MoveTensorToHost(destCtx, t);
    MoveTensorToHost(destCtx, t->grad);
  }
}
