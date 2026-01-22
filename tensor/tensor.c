#include "tensor.h"
#include <__stdarg_va_list.h>
#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <string.h>
#include "common.h"
#include "result/result.h"
#include "tensor.h"
#include "stdbool.h"
#include "cblas.h"

static size_t getBytesForDtype(Dtype type){
  switch(type) {
    case U8:
      return sizeof(u8);

    case U16:
      return sizeof(u16);

    case U32:
      return sizeof(u32);

    case U64:
      return sizeof(u64);

    case F32:
      return sizeof(float);

    default:
      return -1;
  }
}

static u64 calculateNumValuesAndMultipliers(Dim shape, u8 *multipliers) {
  u64 numberOfValues = 1;

  for (int x = shape.numOfDims - 1; x >= 0; x--) {
    // Nultipliers can be null if the caller is not interested
    if (multipliers != NULL) {
      multipliers[x] = numberOfValues;
    }

    numberOfValues *= shape.dims[x];
  }

  return numberOfValues;
}

static Tensor t_Zeros(Context *ctx, Dim shape, Dtype type) {
  Dim tShape = (Dim){.numOfDims=shape.numOfDims};

  // Dont want to hold on the the original memory space, so it can be freed
  tShape.dims = allocate(ctx->memory, sizeof(u32) * shape.numOfDims);
  tShape.multipliers = allocate(ctx->memory, sizeof(u8) * tShape.numOfDims);

  memcpy(tShape.dims, shape.dims, sizeof(u32) * shape.numOfDims); 
  tensor_size_t size = calculateNumValuesAndMultipliers(tShape, tShape.multipliers);
  size_t bytesRequired = size * getBytesForDtype(type);

  Tensor t = (Tensor){.dtype = type, .values=allocate(ctx->memory, bytesRequired), .shape=tShape, .size=size, .isContigous=true};
  memset(t.values, 0, bytesRequired);
  return t;
}

static u64 getIdx(Tensor *t, dim_t* idx) {
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

static bool isOutOfBounds(Tensor *t, Dim dim) {
  for (u8 i = 0; i < dim.numOfDims; i++) {
    if (dim.dims[i] >= t->shape.dims[i]) {
      return true;
    }
  }

  return false;
}

static bool isInvalidTensor(Tensor *t) {
  return t == NULL || t->values == NULL || t->shape.dims == NULL;
}

static void unravel_index(tensor_size_t flatIdx, Dim* shape, dim_t* destCoords) {
  for (int d = shape->numOfDims -1; d >= 0; d--) {
    destCoords[d] = flatIdx % shape->dims[d];
    flatIdx /= shape->dims[d];
  }
}

static Tensor copyToContiguous(Context *ctx, Tensor *source) {
  Tensor copy = t_Zeros(ctx, source->shape, source->dtype);

  u32 *indices = allocate(ctx->memory, sizeof(u32) * source->shape.numOfDims);
  memset(indices, 0, sizeof(u32) * source->shape.numOfDims);

  for (tensor_size_t i = 0; i < source->size; i++) {
    Dim idx = {.dims = indices, .numOfDims = source->shape.numOfDims};
    Value val;
    GetAt(source, idx, &val);
    VALUE_SET(copy.values, i, val);

    for (int d = source->shape.numOfDims - 1; d >= 0; d--) {
      indices[d]++;
      if (indices[d] < source->shape.dims[d]) {
        break;
      }
      indices[d] = 0;
    }
  }

  return copy;
}

static bool areBroadcastable(Tensor* a, Tensor* b) {
  u8 maxDims = a->shape.numOfDims > b->shape.numOfDims ? a->shape.numOfDims : b->shape.numOfDims;
  
  // Compare from the trailing dimensions, virtually prepending 1s
  for (int d = 0; d < maxDims; d++) {
    int aIdx = a->shape.numOfDims - 1 - d;
    int bIdx = b->shape.numOfDims - 1 - d;
    
    // Virtual dimension is 1 if out of bounds
    dim_t aDim = aIdx >= 0 ? a->shape.dims[aIdx] : 1;
    dim_t bDim = bIdx >= 0 ? b->shape.dims[bIdx] : 1;
    
    if (aDim != bDim && aDim != 1 && bDim != 1) {
      return false;
    }
  }
  
  return true;
}

// Binary ops
typedef enum {
  OP_ADD,
  OP_SUBTRACT,
  OP_MULTIPLY,
  OP_DIVIDE
} BinaryOpType;

static Result binaryOp(Context *ctx, Tensor *a, Tensor *b, Tensor *destination, BinaryOpType opType) {
  if (a->dtype != b->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (!areBroadcastable(a, b)) {
    return ERR_DIM_MISMATCH;
  }

  Tensor reshapedA, reshapedB;
  Tensor *opA = a;
  Tensor *opB = b;

  if (a->shape.numOfDims != b->shape.numOfDims) {
    Tensor *smaller = a->shape.numOfDims < b->shape.numOfDims ? a : b;
    Tensor *larger = a->shape.numOfDims < b->shape.numOfDims ? b : a;
    u8 diff = larger->shape.numOfDims - smaller->shape.numOfDims;

    dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * larger->shape.numOfDims);
    for (u8 i = 0; i < diff; i++) {
      newDims[i] = 1;
    }
    for (u8 i = 0; i < smaller->shape.numOfDims; i++) {
      newDims[diff + i] = smaller->shape.dims[i];
    }

    Dim newShape = {.dims = newDims, .numOfDims = larger->shape.numOfDims};
    Tensor *reshapedSmaller = (smaller == a) ? &reshapedA : &reshapedB;
    Result r = Reshape(ctx, smaller, reshapedSmaller, newShape);
    if (r != OK) return r;

    if (smaller == a) {
      opA = &reshapedA;
      opB = b;
    } else {
      opA = a;
      opB = &reshapedB;
    }
  }

  Tensor contiguousA, contiguousB;
  if (!opA->isContigous) {
    contiguousA = copyToContiguous(ctx, opA);
    opA = &contiguousA;
  }
  if (!opB->isContigous) {
    contiguousB = copyToContiguous(ctx, opB);
    opB = &contiguousB;
  }

  Dim outputShape;
  if (opA->size > opB->size) {
    outputShape = opA->shape;
  } else {
    outputShape = opB->shape;
  }

  Tensor output = t_Zeros(ctx, outputShape, opA->dtype);

  dim_t currentCoord[output.shape.numOfDims];
  dim_t aCoords[output.shape.numOfDims];
  dim_t bCoords[output.shape.numOfDims];

  for (tensor_size_t x = 0; x < output.size; x++) {
    unravel_index(x, &outputShape, currentCoord);

    for (u8 d = 0; d < output.shape.numOfDims; d++) {
      aCoords[d] = currentCoord[d] % opA->shape.dims[d];
      bCoords[d] = currentCoord[d] % opB->shape.dims[d];
    }

    Value aVal;
    u64 idx = getIdx(opA, aCoords);
    VALUE_GET_FROM_ARR(opA->values, idx, &aVal, opA->dtype);

    Value bVal;
    idx = getIdx(opB, bCoords);
    VALUE_GET_FROM_ARR(opB->values, idx, &bVal, opB->dtype);

    Value result;
    switch (opType) {
      case OP_ADD:      VALUE_BINOP(result, aVal, bVal, +); break;
      case OP_SUBTRACT: VALUE_BINOP(result, aVal, bVal, -); break;
      case OP_MULTIPLY: VALUE_BINOP(result, aVal, bVal, *); break;
      case OP_DIVIDE:   VALUE_BINOP(result, aVal, bVal, /); break;
    }
    VALUE_SET(output.values, x, result);
  }

  *destination = output;
  return OK;
}

Result Add(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_ADD);
}

Result Subtract(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_SUBTRACT);
}

Result Multiply(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_MULTIPLY);
}

Result Divide(Context *ctx, Tensor *a, Tensor *b, Tensor *destination) {
  return binaryOp(ctx, a, b, destination, OP_DIVIDE);
}

Result GetAt(Tensor *t, Dim dim, Value *result) {
  if (t == NULL || result == NULL) {
    return ERR_NULL_PTR;
  }

  if (dim.numOfDims != t->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  if (isOutOfBounds(t, dim)) {
    return ERR_OUT_OF_BOUNDS;
  }

  u64 idx = getIdx(t, dim.dims);
  VALUE_GET_FROM_ARR(t->values, idx, result, t->dtype);

  return OK;
}

Result AssignValueAt(Context *ctx, Tensor *t, Dim dim, Value value) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (dim.dims == NULL) {
    return ERR_NULL_SHAPE_PROVIDED;
  }

  if (t->dtype != value.dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (dim.numOfDims != t->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  if (isOutOfBounds(t, dim)) {
    return ERR_OUT_OF_BOUNDS;
  }

  // Apply offset here if view
  u64 idx = getIdx(t, dim.dims);
  VALUE_SET(t->values, idx, value);

  return OK;
}

Result Slice(Context *ctx, Tensor *source, Tensor *dest, ...) {
  Range *ranges = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);

  va_list args;
  va_start(args, dest);

  for(u8 x=0; x < source->shape.numOfDims; x++) {
    // Worried about this
    ranges[x] = va_arg(args, Range);

    if (ranges[x].end < ranges[x].start) {
      va_end(args);
      return ERR_INVALID_RANGE;
    }

    if (ranges[x].start < 0 || ranges[x].start > source->shape.dims[x] || ranges[x].end < 0 || ranges[x].end > source->shape.dims[x]) {
      va_end(args);
      return ERR_DIM_MISMATCH;
    }
  }
  va_end(args);

  // Compute the new shape
  // Since this would be a view and views indexes are converted to the base tensors index range using the boundaries. We keep the old multiplier
  Dim newShape = {.dims=allocate(ctx->memory, sizeof(Dim)* source->shape.numOfDims), .numOfDims=source->shape.numOfDims, .multipliers=source->shape.multipliers}; 
  Range *boundary = allocate(ctx->memory, sizeof(Range) * source->shape.numOfDims);

  for(u8 x=0; x < source->shape.numOfDims; x++) {
    Range r = ranges[x];
    u32 dimsize = (r.end - r.start);
    newShape.dims[x] = dimsize;

    if (source->isView) {
      boundary[x] = (Range) {.start=source->boundary->start + ranges[x].start, .end=source->boundary->start + ranges[x].end };
    } else {
      boundary[x] = ranges[x];
    }
  }
  
  tensor_size_t size = calculateNumValuesAndMultipliers(newShape, NULL);
  *dest = ((Tensor) {
      .isView = true, 
      .values=source->values, 
      .shape=newShape, 
      .dtype=source->dtype, 
      .boundary=boundary, 
      .size=size,
      .isContigous=false,
      // TODO: Probably need to comback to this, a slice does not autmattically mean discontingous
    });

  return OK;
}

Result Reshape(Context *ctx, Tensor *source, Tensor *dest, Dim newShape) {
  if (isInvalidTensor(source)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (newShape.dims == NULL) {
    return ERR_NULL_SHAPE_PROVIDED;
  }

  u8* multipliers = allocate(ctx->memory, sizeof(u8) * newShape.numOfDims);
  tensor_size_t proposedSize = calculateNumValuesAndMultipliers(newShape, multipliers);

  if (proposedSize != source->size) {
    return ERR_RESHAPE_DIM_MISMATCH;
  }

  void *values;
  bool isView = source->isView;
  Range *boundary = source->boundary;

  if (!source->isContigous) {
    Tensor contiguous = copyToContiguous(ctx, source);
    values = contiguous.values;
    isView = false;
    boundary = NULL;
  } else {
    values = source->values;
  }

  *dest = ((Tensor) {.isView = isView, .values = values, .dtype = source->dtype, .boundary = boundary, .size = source->size, .isContigous = true});
  dest->shape = (Dim) {.dims = newShape.dims, .numOfDims = newShape.numOfDims, .multipliers = multipliers};
  return OK;
}

Result Transpose(Context *ctx, Tensor *source, Tensor *dest, ...) {
  dim_t transposeDims[2];
  u8 expectedDims = 2;

  if (isInvalidTensor(source)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  if (source->size < 2) {
    return ERR_NO_OP;
  }
  
  va_list args;
  va_start(args, dest);

  for (u8 x = 0; x < expectedDims; x++) {
    transposeDims[x] = va_arg(args, dim_t);
  }
  va_end(args);

  if (transposeDims[0] >= source->shape.numOfDims || transposeDims[1] >= source->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }
  
  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * source->shape.numOfDims);
  memcpy(newDims, source->shape.dims, sizeof(dim_t) * source->shape.numOfDims);
 
  dim_t temp = newDims[transposeDims[0]]; // dim0
  newDims[transposeDims[0]] = newDims[transposeDims[1]];
  newDims[transposeDims[1]] = temp;

  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * source->shape.numOfDims);
  memcpy(newMultipliers, source->shape.multipliers, sizeof(multiplier_t) * source->shape.numOfDims);

  multiplier_t tempMultiplier = newMultipliers[transposeDims[0]];
  newMultipliers[transposeDims[0]] = newMultipliers[transposeDims[1]];
  newMultipliers[transposeDims[1]] = tempMultiplier;

  *dest = (Tensor) {
    .dtype  =source->dtype,
    .values =source->values,
    .size = source->size,
    .isView = true,
    .isContigous = false,
    .shape = {.dims=newDims, .numOfDims=source->shape.numOfDims, .multipliers=newMultipliers},
    .boundary=source->boundary
  };

  return OK;

}

// Creation
Tensor T_Zeros(Context *ctx, Dim shape) {
  return t_Zeros(ctx, shape, U8);
}
