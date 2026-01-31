#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "common.h"
#include "memory.h"
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

static Tensor* t_Zeros(Context *ctx, Dim shape, Dtype type) {
  Dim tShape = (Dim){.numOfDims=shape.numOfDims};

  // Dont want to hold on the the original memory space, so it can be freed
  tShape.dims = allocate(ctx->memory, sizeof(u32) * shape.numOfDims);
  tShape.multipliers = allocate(ctx->memory, sizeof(u8) * tShape.numOfDims);

  memcpy(tShape.dims, shape.dims, sizeof(u32) * shape.numOfDims); 
  tensor_size_t size = calculateNumValuesAndMultipliers(tShape, tShape.multipliers);
  size_t bytesRequired = size * getBytesForDtype(type);

  Tensor *t = allocate(ctx->memory, sizeof(Tensor));
  *t = (Tensor){.dtype = type, .values=allocate(ctx->memory, bytesRequired), .shape=tShape, .size=size, .isContigous=true};
  memset(t->values, 0, bytesRequired);
  return t;
}

// Take a shape coord say (3, 4, 2) and retuns the index in the tensor's contingous memory space
//
static u64 getContigousIdxFromCoord(Tensor *t, dim_t* idx) {
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

static Tensor* copyToContiguous(Context *ctx, Tensor *source) {
  Tensor *copy = t_Zeros(ctx, source->shape, source->dtype);

  u32 *indices = allocate(ctx->memory, sizeof(u32) * source->shape.numOfDims);
  memset(indices, 0, sizeof(u32) * source->shape.numOfDims);

  for (tensor_size_t i = 0; i < source->size; i++) {
    Dim idx = {.dims = indices, .numOfDims = source->shape.numOfDims};
    Value val;
    GetAt(source, idx, &val);
    VALUE_SET(copy->values, i, val);

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

static bool areBatchDimsBroadcastable(Tensor* a, Tensor* b) {
  u8 maxDims = a->shape.numOfDims > b->shape.numOfDims ? a->shape.numOfDims : b->shape.numOfDims;
  
  // For matmul, only check batch dimensions (exclude last 2)
  for (int d = 2; d < maxDims; d++) {
    int aIdx = a->shape.numOfDims - 1 - d;
    int bIdx = b->shape.numOfDims - 1 - d;
    
    dim_t aDim = aIdx >= 0 ? a->shape.dims[aIdx] : 1;
    dim_t bDim = bIdx >= 0 ? b->shape.dims[bIdx] : 1;
    
    if (aDim != bDim && aDim != 1 && bDim != 1) {
      return false;
    }
  }
  
  return true;
}

typedef struct {
  Tensor *a;
  Tensor *b;
} TensorPair;

TensorPair padSmallerTensor(Context *ctx, Tensor *a, Tensor *b) {
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
    Tensor *reshapedSmaller = allocate(ctx->memory, sizeof(Tensor));
    Reshape(ctx, smaller, reshapedSmaller, newShape);

    if (smaller == a) {
      return (TensorPair){.a = reshapedSmaller, .b = b};
    }
    return (TensorPair){.a = a, .b = reshapedSmaller};
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

  TensorPair ops = {.a = a, .b = b};
  if (a->shape.numOfDims != b->shape.numOfDims) {
    ops = padSmallerTensor(ctx, a, b);
  }

  Tensor *opA = ops.a;
  Tensor *opB = ops.b;

  if (!opA->isContigous) {
    opA = copyToContiguous(ctx, opA);
  }
  if (!opB->isContigous) {
    opB = copyToContiguous(ctx, opB);
  }

  Dim outputShape;
  if (opA->size > opB->size) {
    outputShape = opA->shape;
  } else {
    outputShape = opB->shape;
  }

  Tensor *output = t_Zeros(ctx, outputShape, opA->dtype);

  dim_t currentCoord[output->shape.numOfDims];
  dim_t aCoords[output->shape.numOfDims];
  dim_t bCoords[output->shape.numOfDims];

  for (tensor_size_t x = 0; x < output->size; x++) {
    unravel_index(x, &outputShape, currentCoord);

    for (u8 d = 0; d < output->shape.numOfDims; d++) {
      aCoords[d] = currentCoord[d] % opA->shape.dims[d];
      bCoords[d] = currentCoord[d] % opB->shape.dims[d];
    }

    Value aVal;
    u64 idx = getContigousIdxFromCoord(opA, aCoords);
    VALUE_GET_FROM_ARR(opA->values, idx, &aVal, opA->dtype);

    Value bVal;
    idx = getContigousIdxFromCoord(opB, bCoords);
    VALUE_GET_FROM_ARR(opB->values, idx, &bVal, opB->dtype);

    Value result;
    switch (opType) {
      case OP_ADD:      VALUE_BINOP(result, aVal, bVal, +); break;
      case OP_SUBTRACT: VALUE_BINOP(result, aVal, bVal, -); break;
      case OP_MULTIPLY: VALUE_BINOP(result, aVal, bVal, *); break;
      case OP_DIVIDE:   VALUE_BINOP(result, aVal, bVal, /); break;
    }
    VALUE_SET(output->values, x, result);
  }

  *destination = *output;
  return OK;
}

Result straightSum(Context *ctx, Tensor *t, Tensor* dest) {
  Value sums[MAX_PARALLEL_SUMS];

  tensor_size_t x = 0;
  while (x < t->size) {
    u8 numComputations = 0;
    for (u8 y = 0; y < MAX_PARALLEL_SUMS; y++) {
      if (x + y >= t->size) {
        break;
      }

      Value val;
      VALUE_GET_FROM_ARR(t->values, x + y, &val, t->dtype);
      VALUE_BINOP(sums[y], sums[y], val, +);
      numComputations++;
      x += numComputations;
    }
  }

  Value sum = VALUE(t->dtype, 0);
  for (u8 y = 0; y < MAX_PARALLEL_SUMS; y++) {
    VALUE_BINOP(sum, sum, sums[y], +);
  }

  dim_t dims[] = {1};
  Tensor *result = t_Zeros(ctx, (Dim) {.dims=dims, .numOfDims=1}, t->dtype);
  VALUE_UNBOX(sum, result->values);
  *dest = *result;
  
  return OK;
}

// Include offsets in calculation becuase you might be dealign with a view
Result calculateNumElementsBeforeDim(Tensor *t, dim_t dim, tensor_size_t* result) {
  if (dim == 0) {
    *result = 1;
    return OK;
  }

  if(dim >= t->shape.numOfDims) {
    return ERR_OUT_OF_BOUNDS;
  }

  tensor_size_t totalElements = t->shape.dims[0];
  dim_t totalOffset = 0;
  for (dim_t d=1; d < t->shape.numOfDims; d++) {
    if (d == dim) {
      break;
    }
    
    totalElements *= t->shape.dims[d];
  }

  *result = totalElements;

  return OK;
}

Result getDimsBefore(Context *ctx, Tensor *t, dim_t dim, Dim *result) {
  if (dim == 0) {
    return OK;
  }

  if(dim >= t->shape.numOfDims) {
    return ERR_OUT_OF_BOUNDS;
  }

  uint8_t numDimBefore = t->shape.numOfDims - dim - 1;
  for (dim_t d=0; d < dim; d++) {
    result->dims[d] = t->shape.dims[d];
  }

  return OK;
}



Result calculateNumElementsAfterDim(Tensor *t, dim_t dim, tensor_size_t* result) {
  if (dim == t->shape.numOfDims - 1) {
    *result = 1;
    return OK;
  }

  if(dim >= t->size) {
    return ERR_OUT_OF_BOUNDS;
  }

  tensor_size_t numElements = t->shape.dims[dim + 1];
  for (dim_t d=dim + 2; d < t->shape.numOfDims; d++) {
    if (d >= t->shape.numOfDims) {
      break;
    }
    
    numElements *= t->shape.dims[d];
  }

  *result = numElements;
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

  u64 idx = getContigousIdxFromCoord(t, dim.dims);
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
  u64 idx = getContigousIdxFromCoord(t, dim.dims);
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
    Tensor *contiguous = copyToContiguous(ctx, source);
    values = contiguous->values;
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

Result Sum(Context *ctx, Tensor *t, Tensor *dest, dim_t dim) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }
 
  if (dim >= t->shape.numOfDims) {
    return ERR_SUM_DIM_OUT_OF_BOUNDS;
  }

  Tensor *workingTensor = t;
  if (!t->isContigous) {
    workingTensor = copyToContiguous(ctx, t);
  }

  // Treate the tensor like a 3D (__, dim, __) shape
  // Bassically by flattening all the dimensions before and after, treating them as contingous
  tensor_size_t numBeforeDim = 0;
  Result numBeforeResult = calculateNumElementsBeforeDim(workingTensor, dim, &numBeforeDim);
  if (numBeforeResult != OK) {
    return numBeforeResult;
  }

  dim_t reduce = workingTensor->shape.dims[dim];

  tensor_size_t numAfterDim = 0;
  Result numAfterResult = calculateNumElementsAfterDim(workingTensor, dim, &numAfterDim);
  if (numAfterResult != OK) {
    return numBeforeResult;
  }

  tensor_size_t resultSize = numBeforeDim * numAfterDim;
  *dest = (Tensor) {
    .dtype=workingTensor->dtype, 
    .isContigous = true, 
    .isView = false, 
    .size = (resultSize), 
    .values=allocate(ctx->memory, getBytesForDtype(workingTensor->dtype) * resultSize)
  };

  for (tensor_size_t outer = 0; outer < numBeforeDim; outer++) {
    for (tensor_size_t inner = 0; inner < numAfterDim; inner++) {
      Value acc = VALUE(workingTensor->dtype, 0);
      for (dim_t r=0; r < reduce; r++) {
        tensor_size_t sourceIdx = outer * reduce * numAfterDim + r * numAfterDim + inner;
      
        Value v;
        VALUE_GET_FROM_ARR(workingTensor->values, sourceIdx, &v, workingTensor->dtype);
        
        VALUE_BINOP(acc, acc, v, +);
      } 
      // Indexing the outgoing tensor, we'll treat it as a tensor of shape (numBeforeResult, 1, numAfterResult)
      VALUE_UNBOX(acc, dest->values + (outer * numAfterDim + inner));
   }
 }

  dest->shape = (Dim) {.numOfDims = workingTensor->shape.numOfDims, .dims=allocate(ctx->memory, sizeof(dim_t) * workingTensor->shape.numOfDims)};
  memcpy(dest->shape.dims, workingTensor->shape.dims, sizeof(dim_t) * workingTensor->shape.numOfDims);
  // Set the dimension reduced to 1. If the caller want's to the keep the dims as there where;
  dest->shape.dims[dim] = 1;

 if (!t->isContigous) {
   // Means a working (contigous) tensor had to be created for this operation. 
   // It has no use after this operation, so we free it.
   Result freeRes = FreeTensor(ctx, workingTensor);
   if (freeRes != OK) {
     return freeRes;
   }
 }

  return OK;
}

Result Squeeze(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  // Count non-1 dimensions
  u8 newNumDims = 0;
  for (u8 i = 0; i < t->shape.numOfDims; i++) {
    if (t->shape.dims[i] != 1) {
      newNumDims++;
    }
  }

  // Edge case: all dims are 1, keep at least one
  if (newNumDims == 0) {
    newNumDims = 1;
  }

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
  u8 destIdx = 0;

  // If all dims were 1, just set single dim to 1
  if (newNumDims == 1 && t->shape.dims[0] == 1) {
    bool allOnes = true;
    for (u8 i = 0; i < t->shape.numOfDims; i++) {
      if (t->shape.dims[i] != 1) {
        allOnes = false;
        break;
      }
    }
    if (allOnes) {
      newDims[0] = 1;
      destIdx = 1;
    }
  }

  // Copy non-1 dimensions
  if (destIdx == 0) {
    for (u8 i = 0; i < t->shape.numOfDims; i++) {
      if (t->shape.dims[i] != 1) {
        newDims[destIdx++] = t->shape.dims[i];
      }
    }
  }

  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);
  calculateNumValuesAndMultipliers((Dim){.dims = newDims, .numOfDims = newNumDims}, newMultipliers);

  *dest = (Tensor) {
    .dtype = t->dtype,
    .values = t->values,
    .size = t->size,
    .isContigous = t->isContigous,
    .isView = true,
    .boundary = t->boundary,
    .shape = {
      .dims = newDims,
      .numOfDims = newNumDims,
      .multipliers = newMultipliers
    }
  };

  return OK;
}

Result UnSqueeze(Context *ctx, Tensor *t, Tensor *dest, dim_t dim) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  // dim can be 0 to numOfDims (inclusive - can insert at end)
  if (dim > t->shape.numOfDims) {
    return ERR_DIM_MISMATCH;
  }

  u8 newNumDims = t->shape.numOfDims + 1;
  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * newNumDims);
  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * newNumDims);

  // Copy dims and multipliers, inserting 1 and appropriate multiplier at position dim
  for (u8 i = 0; i < newNumDims; i++) {
    if (i < dim) {
      newDims[i] = t->shape.dims[i];
      newMultipliers[i] = t->shape.multipliers[i];

      continue;
    } 

    if (i == dim) {
      newDims[i] = 1;
      // For a dimension of size 1, multiplier doesn't matter for indexing
      // but we use the next dimension's multiplier (or 1 if at end)
      if (dim < t->shape.numOfDims) {
        newMultipliers[i] = t->shape.multipliers[dim];
      } else {
        newMultipliers[i] = 1;
      }
      continue;
    }

    newDims[i] = t->shape.dims[i - 1];
    newMultipliers[i] = t->shape.multipliers[i - 1];
  }

  *dest = (Tensor) {
    .dtype = t->dtype,
    .values = t->values,
    .size = t->size,
    .isContigous = t->isContigous,
    .isView = true,
    .boundary = t->boundary,
    .shape = {
      .dims = newDims,
      .numOfDims = newNumDims,
      .multipliers = newMultipliers
    }
  };

  return OK;
}

Result Clone(Context *ctx, Tensor *t, Tensor *dest) {
  if (isInvalidTensor(t)) {
    return ERR_NULL_TENSOR_PROVIDED;
  }

  Tensor *source = t;
  if (!t->isContigous) {
    source = copyToContiguous(ctx, t);
  }

  size_t valueBytes = getBytesForDtype(source->dtype) * source->size;
  void *newValues = allocate(ctx->memory, valueBytes);
  memcpy(newValues, source->values, valueBytes);

  dim_t *newDims = allocate(ctx->memory, sizeof(dim_t) * source->shape.numOfDims);
  memcpy(newDims, source->shape.dims, sizeof(dim_t) * source->shape.numOfDims);

  multiplier_t *newMultipliers = allocate(ctx->memory, sizeof(multiplier_t) * source->shape.numOfDims);
  memcpy(newMultipliers, source->shape.multipliers, sizeof(multiplier_t) * source->shape.numOfDims);

  *dest = (Tensor) {
    .dtype = source->dtype,
    .values = newValues,
    .size = source->size,
    .isContigous = true,
    .isView = false,
    .boundary = NULL,
    .shape = {
      .dims = newDims,
      .numOfDims = source->shape.numOfDims,
      .multipliers = newMultipliers
    }
  };

  return OK;
}

Result MatMul(Context *ctx, Tensor *a, Tensor *b, Tensor *result) {
  if (isInvalidTensor(a) || isInvalidTensor(b)) {
      return ERR_NULL_TENSOR_PROVIDED;
  }

  if ((a->shape.numOfDims < 2 || b->shape.numOfDims < 2)) {
    return ERR_MATMUL_MIN_2D;
  }

  if (a->dtype != b->dtype) {
    return ERR_DTYPE_MISMATCH;
  }

  if (a->dtype != F32 && a->dtype != F64 && a->dtype != F16) {
    return ERR_DTYPE_MISMATCH;
  }

  dim_t innerDimA = a->shape.dims[a->shape.numOfDims-1];
  dim_t innerDimB = b->shape.dims[b->shape.numOfDims-2];

  if (innerDimA != innerDimB) {
    return ERR_MATMUL_INNER_DIM_MISMATCH;
  }

  if (!areBatchDimsBroadcastable(a, b)) {
    return ERR_DIM_MISMATCH;
  }

  TensorPair ops = {.a = a, .b = b};
  if (a->shape.numOfDims != b->shape.numOfDims) {
    ops = padSmallerTensor(ctx, a, b);
  }

  Tensor *opA = ops.a;
  Tensor *opB = ops.b;

  if (!opA->isContigous) {
    opA = copyToContiguous(ctx, opA);
  }

  if (!opB->isContigous) {
    opB = copyToContiguous(ctx, opB);
  }

  dim_t m = opA->shape.dims[opA->shape.numOfDims-2];
  dim_t k = opB->shape.dims[opB->shape.numOfDims-2];
  dim_t n = opB->shape.dims[opB->shape.numOfDims-1];

  tensor_size_t batchSizeA, batchSizeB;
  dim_t batchDimIdx = opA->shape.numOfDims - 2;
  Result r = calculateNumElementsBeforeDim(opA, batchDimIdx, &batchSizeA);
  if (r != OK) return r;
  r = calculateNumElementsBeforeDim(opB, batchDimIdx, &batchSizeB);
  if (r != OK) return r;

  tensor_size_t batchSize = batchSizeA > batchSizeB ? batchSizeA : batchSizeB;

  Tensor *sentinel = batchSizeA >= batchSizeB ? opA : opB;
  Dim newDim = (Dim) {.dims = allocate(ctx->memory, sizeof(dim_t) * sentinel->shape.numOfDims), .numOfDims=sentinel->shape.numOfDims };
  r = getDimsBefore(ctx, sentinel, batchDimIdx, &newDim);
  if (r != OK) return r;

  newDim.dims[newDim.numOfDims - 2] = m; 
  newDim.dims[newDim.numOfDims - 1] = n; 

  newDim.multipliers = allocate(ctx->memory, sizeof(multiplier_t) * newDim.numOfDims);
  tensor_size_t size = calculateNumValuesAndMultipliers(newDim, newDim.multipliers);

  *result = (Tensor) {
    .dtype = a->dtype, 
    .isContigous = true,
    .isView = false,
    .shape=newDim,
    .size=size,
    .values=allocate(ctx->memory, getBytesForDtype(opA->dtype) * size)
  };

  size_t elemSize = getBytesForDtype(opA->dtype);
  for (tensor_size_t i = 0; i < batchSize; i++) {
    tensor_size_t aIdx = i % batchSizeA;
    tensor_size_t bIdx = i % batchSizeB;
    void *A_batch = (char*)opA->values + aIdx * (m * k) * elemSize;
    void *B_batch = (char*)opB->values + bIdx * (k * n) * elemSize;
    void *C_batch = (char*)result->values + i * (m * n) * elemSize;
    BLAS_GEMM(opA->dtype, A_batch, B_batch, C_batch, m, n, k);
  }

  return OK;
}

// Creation
Tensor* T_Zeros(Context *ctx, Dim shape) {
  return t_Zeros(ctx, shape, U8);
}

// Destruction
Result FreeTensor(Context *ctx, Tensor *t) {
  if (t == NULL) return ERR_NULL_TENSOR_PROVIDED;

  if (t->isView) {
   return ERR_CANNOT_FREE_VIEW_TENSOR; 
  }

  if (t->values != NULL) {
    freeAlloc(ctx->memory, t->values);
  }
  if (t->shape.dims != NULL) {
    freeAlloc(ctx->memory, t->shape.dims);
  }
  if (t->shape.multipliers != NULL) {
    freeAlloc(ctx->memory, t->shape.multipliers);
  }

  freeAlloc(ctx->memory, t);
  return OK;
}

