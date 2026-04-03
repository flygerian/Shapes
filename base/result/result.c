#include "result.h"

// Result creation functions

Result voidPtrResult(void *ptr) {
  return (Result){.err = NONE, .as.voidPtr = ptr};
}

Result tensorResult(Tensor *tensor) {
  return (Result){.err = NONE, .as.tensor = tensor};
}

Result tensorArgResult(TensorArg *arg) {
  return (Result){.err = NONE, .as.tensorarg = arg};
}

Result tensorPairResult(TensorPair *pair) {
  return (Result){.err = NONE, .as.tensorPair = pair};
}

Result dimsResult(dim_t *dims) {
  return (Result){.err = NONE, .as.dims = dims};
}

Result multiplierResult(multiplier_t *multipliers) {
  return (Result){.err = NONE, .as.multipliers = multipliers};
}

Result deviceResult(Device *device) {
  return (Result){.err = NONE, .as.device = device};
}

Result dimResult(Dim dim) {
  return (Result){.err = NONE, .as.dim = dim};
}

Result cudaCachedBlockResult(CudaCachedBlock *block) {
  return (Result){.err = NONE, .as.cudaCachedBlock = block};
}

Result contextResult(Context context) {
  return (Result){.err = NONE, .as.context = context};
}

Result tensorSizeResult(tensor_size_t size) {
  return (Result){.err = NONE, .as.tensorSize = size};
}

Result valueResult(Value value) {
  return (Result){.err = NONE, .as.value = value};
}

Result errorResult(Error err) {
  return (Result){.err = err, .as = {0}};
}

Result okResult(void) {
  return (Result){.err = NONE, .as = {0}};
}

// Result extraction functions

void *asVoidPtr(Result result) {
  return result.as.voidPtr;
}

Tensor *asTensor(Result result) {
  return result.as.tensor;
}

TensorArg *asTensorArg(Result result) {
  return result.as.tensorarg;
}

TensorPair *asTensorPair(Result result) {
  return result.as.tensorPair;
}

dim_t *asDims(Result result) {
  return result.as.dims;
}

Dim asDim(Result result) {
  return result.as.dim;
}

multiplier_t *asMultipliers(Result result) {
  return result.as.multipliers;
}

Device *asDevice(Result result) {
  return result.as.device;
}

CudaCachedBlock *asCudaCachedBlock(Result result) {
  return result.as.cudaCachedBlock;
}

Context asContext(Result result) {
  return result.as.context;
}

tensor_size_t asTensorSize(Result result) {
  return result.as.tensorSize;
}

Value asValue(Result result) {
  return result.as.value;
}

// Error checking functions

bool operationFailed(Result result) {
  return result.err != NONE;
}
