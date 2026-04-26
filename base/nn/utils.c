#include "nn/nn.h"
#include "result/result.h"
#include "utils_lib/array.h"
#include <stddef.h>

FowardPassOp *array_FowardPassOpIdx(Array *array, size_t idx) {
  PANIC_IF(array->elemSize != sizeof(FowardPassOp *), ARRAY_ELEM_SIZE_MISMATCH);
  return *((FowardPassOp **)Array_Idx(array, idx));
}

void array_AppendFowardPassOp(Array *array, FowardPassOp *op) {
  PANIC_IF(array->elemSize != sizeof(FowardPassOp *), ARRAY_ELEM_SIZE_MISMATCH);
  Array_Append(array, (void *)&op);
}
