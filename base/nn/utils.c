#include "nn/nn.h"
#include "result/result.h"
#include "shapes.h"
#include "utils_lib/array.h"
#include <stddef.h>

FowardPassOp array_FowardPassOpIdx(Array *array, size_t idx) {
  PANIC_IF(array->elemSize != sizeof(FowardPassOp), ARRAY_ELEM_SIZE_MISMATCH);
  return *((FowardPassOp *)Array_Idx(array, idx));
}

void array_AppendFowardPassOp(Array *array, FowardPassOp *op) {
  PANIC_IF(array->elemSize != sizeof(FowardPassOp), ARRAY_ELEM_SIZE_MISMATCH);
  Array_Append(array, op);
}

void loadIntoTensor(Context *ctx, Tensor *dst, Tensor *src) {
  PANIC_IF(dst == NULL || src == NULL, ERR_NULL_TENSOR_PROVIDED);
  PANIC_IF(dst->dtype != src->dtype, ERR_DTYPE_MISMATCH);
  PANIC_IF(dst->shape.numOfDims != src->shape.numOfDims, ERR_DIM_MISMATCH);
  for (u8 d = 0; d < dst->shape.numOfDims; d++) {
    PANIC_IF(dst->shape.dims[d] != src->shape.dims[d], ERR_DIM_MISMATCH);
  }
  shapes_Copy(ctx, src, dst);
}

