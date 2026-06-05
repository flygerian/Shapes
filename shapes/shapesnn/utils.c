#include "nn.h"
#include "result.h"
#include "shapes.h"
#include "array.h"
#include <stddef.h>

FowardPassOp array_FowardPassOpIdx(olib_Array *array, size_t idx) {
  PANIC_IF(array->elemSize != sizeof(FowardPassOp), ARRAY_ELEM_SIZE_MISMATCH);
  return *((FowardPassOp *)olib_ArrayIdx(array, idx));
}

void array_AppendFowardPassOp(olib_Array *array, FowardPassOp *op) {
  PANIC_IF(array->elemSize != sizeof(FowardPassOp), ARRAY_ELEM_SIZE_MISMATCH);
  olib_ArrayAppend(array, op);
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

void shapes_Array_AppendLayer(olib_Array *array, Layer *layer) {
  olib_ArrayAppend(array, (void *)&layer);
}

Layer *shapes_Array_LayerIdx(olib_Array *array, size_t idx) {
  return *(Layer **)olib_ArrayIdx(array, idx);
}
